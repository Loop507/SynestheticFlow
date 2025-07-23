import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time
import os
import subprocess
from numba import jit
import random

# --- CONFIGURAZIONI FORMATO ---
VIDEO_FORMATS = {
    "16:9 (Landscape) - 1280x720": (1280, 720),
    "1:1 (Square) - 720x720": (720, 720),
    "9:16 (Portrait) - 720x1280": (720, 1280)
}

# --- FUNZIONI DI SUPPORTO ---
def prepare_audio_file(uploaded_file, temp_dir):
    audio_path = f"{temp_dir}/input_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    return audio_path

def analyze_audio_minimal(audio_path):
    y, sr = librosa.load(audio_path, sr=11025)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return y, beat_times, tempo, sr

def get_optimal_settings(duration, width, height):
    fps = 20
    estimated_size = (width * height * fps * duration) / (1024 * 1024)  # rough estimate
    return width, height, fps, int(estimated_size)

def analyze_frequency_bands(freq_data):
    """Analizza le bande di frequenza (basse, medie, acute)"""
    if len(freq_data) == 0:
        return 0.0, 0.0, 0.0

    # Normalizza i dati di frequenza per evitare valori eccessivi
    max_val = np.max(freq_data)
    if max_val > 1e-6:
        freq_data_norm = freq_data / max_val
    else:
        return 0.0, 0.0, 0.0

    total_bins = len(freq_data_norm)
    low_end = max(1, total_bins // 3)
    mid_end = max(low_end + 1, (total_bins * 2) // 3)

    low_freq = np.mean(freq_data_norm[:low_end])
    mid_freq = np.mean(freq_data_norm[low_end:mid_end])
    high_freq = np.mean(freq_data_norm[mid_end:])

    return float(low_freq), float(mid_freq), float(high_freq)

def process_frame_data(audio_chunk):
    if len(audio_chunk) == 0:
        return 0.0, np.array([])
    
    rms = np.sqrt(np.mean(audio_chunk ** 2))

    # Applica una finestra per FFT per ridurre artefatti
    if len(audio_chunk) > 1:
        windowed_audio_chunk = audio_chunk * np.hanning(len(audio_chunk))
        freq_data = np.abs(np.fft.rfft(windowed_audio_chunk))
    else:
        freq_data = np.array([])

    return float(rms), freq_data

def hex_to_bgr(hex_color):
    """Converte colore hex in formato BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 0)
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # BGR format
    except ValueError:
        return (0, 0, 0)

# --- FUNZIONI DI SUPPORTO PER SINCRONIZZAZIONE BPM ---

def calculate_bpm_phase(current_time, tempo, sync_type, beat_times, bmp_settings):
    """Calcola la fase corrente basata sul BPM per sincronizzazione precisa"""
    if not bmp_settings['enabled'] or len(beat_times) == 0 or tempo <= 0:
        return 0.0, False, 1.0
    
    # Trova il beat più vicino
    beat_distances = np.abs(beat_times - current_time)
    nearest_beat_idx = np.argmin(beat_distances)
    nearest_beat_time = beat_times[nearest_beat_idx]
    
    # Calcola la distanza temporale dal beat più vicino
    time_from_beat = current_time - nearest_beat_time
    
    # Determina se siamo "su un beat"
    beat_window = 60.0 / (tempo * 8)
    is_on_beat = abs(time_from_beat) <= beat_window
    
    # Calcola la fase in base al tipo di sincronizzazione
    beat_duration = 60.0 / tempo
    
    sync_multipliers = {
        "Beat principale (1/1)": 1.0,
        "Mezzo beat (1/2)": 2.0,
        "Doppio beat (2/1)": 0.5,
        "Terzine (1/3)": 3.0
    }
    
    multiplier = sync_multipliers.get(sync_type, 1.0)
    phase_duration = beat_duration / multiplier
    
    # Calcola la fase (0 a 2π nel periodo del beat)
    if phase_duration > 0:
        time_in_cycle = (current_time % phase_duration)
        phase = (time_in_cycle / phase_duration) * 2 * np.pi
    else:
        phase = 0.0
    
    # Calcola l'intensità del beat
    if beat_window > 0:
        beat_intensity = max(0.1, 1.0 - (abs(time_from_beat) / beat_window)) if is_on_beat else 0.1
    else:
        beat_intensity = 0.1
    
    return float(phase), is_on_beat, float(beat_intensity)

def apply_bpm_movement_modulation(base_value, phase, beat_intensity, modulation_type, bmp_settings):
    """Applica modulazione basata sui BPM a un valore base"""
    if not bmp_settings['enabled']:
        return float(base_value)
    
    intensity = bmp_settings['beat_response_intensity']
    
    modulation = 0.0
    if modulation_type == 'sine':
        modulation = np.sin(phase) * intensity * beat_intensity
    elif modulation_type == 'cosine':
        modulation = np.cos(phase) * intensity * beat_intensity
    elif modulation_type == 'pulse':
        modulation = (np.sin(phase) ** 2) * intensity * beat_intensity
    elif modulation_type == 'sawtooth':
        normalized_phase = (phase / (2 * np.pi)) % 1
        modulation = (normalized_phase * 2 - 1) * intensity * beat_intensity
    
    if bmp_settings['smooth_transitions']:
        smooth_factor = 0.5
        modulation *= smooth_factor
    
    return float(base_value + modulation)

# --- FUNZIONI FRATTALI AVANZATE ---

@jit(nopython=True)
def mandelbrot_set_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    """Genera il set di Mandelbrot con influenza audio (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Coordinate complesse con zoom e movimento
            c_real = (x - width/2) / (zoom * width/4) + move_x
            c_imag = (y - height/2) / (zoom * height/4) + move_y

            # AUMENTATO DRASICAMENTE L'INFLUENZA AUDIO QUI per una trasformazione più marcata
            c_real += audio_influence * 0.5 * np.sin(x * 0.001) # AUMENTATO DA 0.2
            c_imag += audio_influence * 0.5 * np.cos(y * 0.001) # AUMENTATO DA 0.2

            z_real = 0.0
            z_imag = 0.0
            iteration = 0

            # Calcolo iterativo
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            # Colori basati sul numero di iterazioni
            if iteration == max_iter:
                fractal[y, x] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                t = iteration / max_iter
                r = int(9 * (1 - t) * t * t * t * 255)
                g = int(15 * (1 - t) * (1 - t) * t * t * 255)
                b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
                fractal[y, x] = np.array([b, g, r], dtype=np.uint8)

    return fractal

@jit(nopython=True)
def julia_set_numba(width, height, max_iter, c_real_base, c_imag_base, zoom, audio_mod):
    """Genera il set di Julia con parametri dinamici (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    # Modifica i parametri C in base all'audio
    # AUMENTATO DRASICAMENTE L'INFLUENZA AUDIO QUI
    c_real = c_real_base + audio_mod * 0.3 # AUMENTATO DA 0.15
    c_imag = c_imag_base + audio_mod * 0.35 # AUMENTATO DA 0.2

    for y in range(height):
        for x in range(width):
            z_real = (x - width/2) / (zoom * width/4)
            z_imag = (y - height/2) / (zoom * width/4)

            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            # Colori più vivaci per Julia
            if iteration == max_iter:
                fractal[y, x] = np.array([20, 20, 40], dtype=np.uint8)
            else:
                t = iteration / max_iter
                b = int(255 * (np.sin(t * 10 + 0) * 0.5 + 0.5))
                g = int(255 * (np.sin(t * 10 + 2 * np.pi / 3) * 0.5 + 0.5))
                r = int(255 * (np.sin(t * 10 + 4 * np.pi / 3) * 0.5 + 0.5))
                fractal[y, x] = np.array([b, g, r], dtype=np.uint8)

    return fractal

@jit(nopython=True)
def burning_ship_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    """Genera il frattale Burning Ship (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            c_real = (x - width/2) / (zoom * width/4) + move_x
            c_imag = (y - height/2) / (zoom * height/4) + move_y

            # AUMENTATO DRASICAMENTE L'INFLUENZA AUDIO QUI
            c_real += audio_influence * 0.08 * np.sin(x * 0.002 + y * 0.001) # AUMENTATO DA 0.015
            c_imag += audio_influence * 0.08 * np.cos(x * 0.001 + y * 0.002) # AUMENTATO DA 0.015

            z_real = 0.0
            z_imag = 0.0
            iteration = 0

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                # La differenza chiave: valori assoluti
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0 * abs(z_real) * abs(z_imag) + c_imag
                z_real = z_real_new
                iteration += 1

            # Colori fiammeggianti dinamici
            if iteration == max_iter:
                fractal[y, x] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                t = iteration / max_iter
                b = int(255 * (np.sin(t * 5 + 0) * 0.5 + 0.5))
                g = int(255 * (np.sin(t * 5 + 1.5) * 0.5 + 0.5))
                r = int(255 * (np.sin(t * 5 + 3) * 0.5 + 0.5))
                fractal[y, x] = np.array([b, g, r], dtype=np.uint8)

    return fractal

@jit(nopython=True)
def _remove_squares_numba(arr, level, x, y, size, fill_color_b, fill_color_g, fill_color_r):
    """Funzione ricorsiva per Sierpinski Carpet (Numba ottimizzato)"""
    if level == 0 or size < 3:
        return

    third = size // 3

    # Rimuovi il quadrato centrale
    for i in range(third):
        for j in range(third):
            cx = x + third + i
            cy = y + third + j
            if cx < arr.shape[0] and cy < arr.shape[1]:
                arr[cx, cy, 0] = fill_color_b
                arr[cx, cy, 1] = fill_color_g
                arr[cx, cy, 2] = fill_color_r

    # Ricorsione sui quadrati rimanenti
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:  # Salta il quadrato centrale
                _remove_squares_numba(arr, level-1, x + i*third, y + j*third, third, 
                                     fill_color_b, fill_color_g, fill_color_r)

def generate_sierpinski_carpet(width, height, iterations, audio_scale, base_color_bgr):
    """Genera il tappeto di Sierpinski con influenza audio"""
    size = min(width, height)
    
    # Inizia con un colore di base
    carpet = np.full((size, size, 3), base_color_bgr, dtype=np.uint8)

    # Numero di iterazioni basato sull'audio
    # AUMENTATO DRASICAMENTE L'INFLUENZA AUDIO QUI
    iter_count = max(1, min(6, int(iterations + audio_scale * 8))) # AUMENTATO DA 4

    # Colore "vuoto" (nero) per i fori
    _remove_squares_numba(carpet, iter_count, 0, 0, size, 0, 0, 0)

    # Ridimensiona il tappeto per adattarlo al frame
    fractal = cv2.resize(carpet, (width, height), interpolation=cv2.INTER_AREA)

    return fractal

def apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings):
    """Applica colori basati sulle frequenze al frattale esistente"""
    if not color_settings['use_frequency_colors']:
        return fractal

    colored_fractal = fractal.copy()

    low_bgr = np.array(hex_to_bgr(color_settings['low_freq_color']), dtype=np.float32)
    mid_bgr = np.array(hex_to_bgr(color_settings['mid_freq_color']), dtype=np.float32)
    high_bgr = np.array(hex_to_bgr(color_settings['high_freq_color']), dtype=np.float32)

    # Normalizza le influenze delle frequenze per il blending
    low_intensity = np.clip(low_freq * 5.0, 0.0, 1.0)
    mid_intensity = np.clip(mid_freq * 5.0, 0.0, 1.0)
    high_intensity = np.clip(high_freq * 5.0, 0.0, 1.0)

    # Crea un colore base mescolando i colori delle frequenze
    mixed_frequency_color = (
        low_bgr * low_intensity +
        mid_bgr * mid_intensity +
        high_bgr * high_intensity
    )
    
    sum_intensity = low_intensity + mid_intensity + high_intensity
    if sum_intensity > 1e-6:
        mixed_frequency_color = np.clip(mixed_frequency_color / sum_intensity, 0, 255)
    else:
        mixed_frequency_color = np.array([0, 0, 0], dtype=np.float32)

    # Applica il colore misto ai pixel del frattale che non sono neri
    mask = np.any(colored_fractal != [0, 0, 0], axis=-1)

    # Modula il colore del frattale con il colore misto delle frequenze
    for c in range(3):
        colored_fractal[mask, c] = np.clip(
            colored_fractal[mask, c].astype(np.float32) * (mixed_frequency_color[c] / 255.0) * 1.5, 
            0, 255
        ).astype(np.uint8)

    return colored_fractal

# --- FUNZIONI DI DISEGNO FRATTALE PER IL PROCESSING (ORA CON BPM SYNC) ---

def draw_mandelbrot_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings):
    """Mandelbrot con sincronizzazione BPM avanzata"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    # Calcola fase e intensità BPM
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bmp_settings['movement_sync_type'], beat_times, bmp_settings)
    
    # Modifica: Applica movement_scale_factor all'RMS e alle frequenze
    effective_rms = rms * movement_scale_factor
    effective_low_freq = low_freq * movement_scale_factor
    effective_mid_freq = mid_freq * movement_scale_factor
    effective_high_freq = high_freq * movement_scale_factor

    # Parametri dinamici con sincronizzazione BPM
    # AUMENTATI I COEFFICIENTI QUI
    base_max_iter = 80 + effective_rms * 350 # AUMENTATO DA 250
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'pulse', bmp_settings))
    max_iter = max(50, min(250, max_iter)) # AUMENTATO IL LIMITE SUPERIORE
    
    # Zoom sincronizzato sui BPM (maggiore influenza delle frequenze)
    base_zoom = 1.5
    bmp_zoom_modulation = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bmp_settings) * 0.5
    # AUMENTATI I COEFFICIENTI QUI
    audio_zoom_influence = effective_rms * 12 + effective_low_freq * 40 # AUMENTATO DA 8 e 30
    zoom = base_zoom + bmp_zoom_modulation + audio_zoom_influence * 0.9 # AUMENTATO DA 0.7
    
    # Movimento sincronizzato (maggiore influenza delle frequenze)
    base_move_x = -0.75
    base_move_y = 0.05
    
    bmp_move_x = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bmp_settings) * 0.1
    bmp_move_y = apply_bpm_movement_modulation(0, phase * 1.3, beat_intensity, 'cosine', bmp_settings) * 0.08
    
    # AUMENTATI I COEFFICIENTI QUI
    move_x = base_move_x + bmp_move_x + effective_mid_freq * 0.3 # AUMENTATO DA 0.2
    move_y = base_move_y + bmp_move_y + effective_high_freq * 0.25 # AUMENTATO DA 0.15
    
    # Influenza audio modulata dai BPM
    # AUMENTATI I COEFFICIENTI QUI
    base_audio_influence = (effective_rms * 4.0 + (effective_low_freq + effective_mid_freq + effective_high_freq) / 1.5) # AUMENTATO
    audio_influence = base_audio_influence * (1.0 + beat_intensity * 0.5)
    
    fractal = mandelbrot_set_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    
    if color_settings['use_frequency_colors'] and (movement_scale_factor > 0 or any(f > 0 for f in [low_freq, mid_freq, high_freq])): 
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)

    # Alpha blending reattivo ai BPM - AUMENTATA LA SENSIBILITA'
    base_alpha = 0.65
    beat_alpha_boost = 0
    if bmp_settings['enabled']: 
        beat_alpha_boost = bmp_settings['beat_response_intensity'] * 0.5 # AUMENTATO DA 0.4
    alpha = min(0.95, base_alpha + beat_alpha_boost)
    
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_julia_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings):
    """Julia Set con sincronizzazione BPM avanzata"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bmp_settings['movement_sync_type'], beat_times, bmp_settings)
    
    # Modifica: Applica movement_scale_factor all'RMS e alle frequenze
    effective_rms = rms * movement_scale_factor
    effective_low_freq = low_freq * movement_scale_factor
    effective_mid_freq = mid_freq * movement_scale_factor
    effective_high_freq = high_freq * movement_scale_factor

    # Iterazioni con modulazione BPM
    # AUMENTATI I COEFFICIENTI QUI
    base_max_iter = 70 + effective_rms * 180 # AUMENTATO DA 120
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'pulse', bmp_settings))
    max_iter = max(50, min(200, max_iter)) # AUMENTATO LIMITE SUPERIORE
    
    # Parametri C di Julia sincronizzati (maggiore influenza delle frequenze)
    base_c_real = -0.7
    base_c_imag = 0.27015
    
    bmp_c_real_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bmp_settings) * 0.1
    bmp_c_imag_mod = apply_bpm_movement_modulation(0, phase * 0.8, beat_intensity, 'cosine', bmp_settings) * 0.08
    
    # AUMENTATI I COEFFICIENTI QUI
    c_real_base = base_c_real + bmp_c_real_mod + effective_mid_freq * 0.4 # AUMENTATO DA 0.3
    c_imag_base = base_c_imag + bmp_c_imag_mod + effective_high_freq * 0.35 # AUMENTATO DA 0.25
    
    # Zoom con BPM sync (maggiore influenza delle frequenze)
    base_zoom = 1.0
    bmp_zoom_mod = apply_bpm_movement_modulation(0, phase * 1.5, beat_intensity, 'sine', bmp_settings) * 0.3
    # AUMENTATI I COEFFICIENTI QUI
    zoom = base_zoom + bmp_zoom_mod + effective_rms * 3.5 + effective_high_freq * 7.0 # AUMENTATO
    
    # Audio modulation con BPM (maggiore influenza delle frequenze)
    # AUMENTATI I COEFFICIENTI QUI
    base_audio_mod = (effective_rms * 3.5 + (effective_low_freq * 3.0 + effective_mid_freq * 3.5 + effective_high_freq * 1.5)) # AUMENTATO
    audio_mod = base_audio_mod * (1.0 + beat_intensity * 0.6)
    
    fractal = julia_set_numba(width, height, max_iter, c_real_base, c_imag_base, zoom, audio_mod)
    
    if color_settings['use_frequency_colors'] and (movement_scale_factor > 0 or any(f > 0 for f in [low_freq, mid_freq, high_freq])):
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)

    # Alpha blending reattivo ai BPM - AUMENTATA LA SENSIBILITA'
    alpha = min(0.95, 0.75 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.4 if bmp_settings['enabled'] else 0)) # AUMENTATO DA 0.3
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_burning_ship_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings):
    """Burning Ship con sincronizzazione BPM"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bmp_settings['movement_sync_type'], beat_times, bmp_settings)
    
    # Modifica: Applica movement_scale_factor all'RMS e alle frequenze
    effective_rms = rms * movement_scale_factor
    effective_low_freq = low_freq * movement_scale_factor
    effective_mid_freq = mid_freq * movement_scale_factor
    effective_high_freq = high_freq * movement_scale_factor

    # AUMENTATI I COEFFICIENTI QUI
    base_max_iter = 60 + effective_rms * 150 # AUMENTATO DA 100
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'sawtooth', bmp_settings))
    max_iter = max(40, min(150, max_iter)) # AUMENTATO LIMITE SUPERIORE
    
    base_zoom = 1.0
    bmp_zoom_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bmp_settings) * 0.2
    # AUMENTATI I COEFFICIENTI QUI
    zoom = base_zoom + bmp_zoom_mod + effective_rms * 3.5 + effective_mid_freq * 6.0 # AUMENTATO
    
    base_move_x, base_move_y = -1.8, -0.08
    bmp_move_x = apply_bpm_movement_modulation(0, phase * 1.2, beat_intensity, 'sine', bmp_settings) * 0.05
    bmp_move_y = apply_bpm_movement_modulation(0, phase * 0.9, beat_intensity, 'cosine', bmp_settings) * 0.03
    
    # Maggiore influenza delle frequenze sul movimento
    # AUMENTATI I COEFFICIENTI QUI
    move_x = base_move_x + bmp_move_x + effective_low_freq * 0.3 # AUMENTATO DA 0.2
    move_y = base_move_y + bmp_move_y + effective_mid_freq * 0.25 # AUMENTATO DA 0.15
    
    # AUMENTATI I COEFFICIENTI QUI
    audio_influence = (effective_rms * 2.5 + effective_high_freq * 2.0) * (1.0 + (beat_intensity * 0.4 if bmp_settings['enabled'] else 0)) # AUMENTATO
    
    fractal = burning_ship_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    
    if color_settings['use_frequency_colors'] and (movement_scale_factor > 0 or any(f > 0 for f in [low_freq, mid_freq, high_freq])):
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    # Alpha blending reattivo ai BPM - AUMENTATA LA SENSIBILITA'
    alpha = min(0.9, 0.7 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.35 if bmp_settings['enabled'] else 0)) # AUMENTATO DA 0.25
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_sierpinski_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings):
    """Sierpinski Carpet con sincronizzazione BPM"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bmp_settings['movement_sync_type'], beat_times, bmp_settings)
    
    # Modifica: Applica movement_scale_factor all'RMS e alle frequenze
    effective_rms = rms * movement_scale_factor
    effective_low_freq = low_freq * movement_scale_factor
    effective_mid_freq = mid_freq * movement_scale_factor
    effective_high_freq = high_freq * movement_scale_factor

    # AUMENTATI I COEFFICIENTI QUI
    base_iterations = 3 + int(effective_rms * 8) + int(effective_low_freq * 8) # AUMENTATO DA 5 e 5
    bmp_iter_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bmp_settings) * 2
    iterations = int(base_iterations + bmp_iter_mod)
    
    # AUMENTATI I COEFFICIENTI QUI
    base_audio_scale = (effective_rms * 4.0 + (effective_low_freq * 5.0 + effective_mid_freq * 2.5 + effective_high_freq * 1.5)) # AUMENTATO
    audio_scale = base_audio_scale * (1.0 + (beat_intensity * 0.5 if bmp_settings['enabled'] else 0))
    
    base_carpet_color_bgr = hex_to_bgr(color_settings['background_color'])
    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale, base_carpet_color_bgr)
    
    if color_settings['use_frequency_colors'] and (movement_scale_factor > 0 or any(f > 0 for f in [low_freq, mid_freq, high_freq])):
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    # Alpha blending reattivo ai BPM - AUMENTATA LA SENSIBILITA'
    alpha = min(0.85, 0.6 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.4 if bmp_settings['enabled'] else 0)) # AUMENTATO DA 0.3
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

# --- FUNZIONI DI MERGE VIDEO/AUDIO ---
def merge_video_audio(video_path, audio_path, output_path):
    """Combina video e audio usando ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',  # -y per sovrascrivere il file se esiste
            '-i', video_path,  # input video
            '-i', audio_path,  # input audio
            '-c:v', 'copy',    # copia il codec video senza ricodifica
            '-c:a', 'aac',     # usa codec audio AAC
            '-shortest',       # termina quando finisce il più corto
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True, "Merge completato con successo"
        else:
            return False, f"Errore ffmpeg: {result.stderr}"

    except FileNotFoundError:
        return False, "ffmpeg non trovato. Installa ffmpeg sul sistema."
    except Exception as e:
        return False, f"Errore durante il merge: {str(e)}"

# --- INTERFACCIA STREAMLIT MIGLIORATA ---

st.title("🌌 **SynestheticFlow**")
st.markdown("*<span style='font-size: 12px;'>Visualizer by loop507</span>*", unsafe_allow_html=True)

# --- STREAMLIT UI INTERFACE COMPLETION ---

# File upload section
st.sidebar.header("🎵 Carica Audio")
uploaded_file = st.sidebar.file_uploader(
    "Scegli un file audio", 
    type=['mp3', 'wav', 'flac', 'm4a', 'ogg']
)

# Video format selection
st.sidebar.header("📺 Formato Video")
selected_format = st.sidebar.selectbox(
    "Scegli formato:",
    list(VIDEO_FORMATS.keys())
)
width, height = VIDEO_FORMATS[selected_format]

# Fractal type selection
st.sidebar.header("🌀 Tipo Frattale")
fractal_type = st.sidebar.selectbox(
    "Scegli frattale:",
    ["Mandelbrot", "Julia", "Burning Ship", "Sierpinski"]
)

# Movement settings
st.sidebar.header("🎬 Movimento")
movement_scale = st.sidebar.slider(
    "Intensità movimento", 
    min_value=0.0, 
    max_value=3.0, 
    value=1.0, 
    step=0.1
)

# BPM Sync Settings
st.sidebar.header("🎵 Sincronizzazione BPM")
bmp_settings = {
    'enabled': st.sidebar.checkbox("Abilita sync BPM", value=True),
    'movement_sync_type': st.sidebar.selectbox(
        "Tipo sincronizzazione:",
        ["Beat principale (1/1)", "Mezzo beat (1/2)", "Doppio beat (2/1)", "Terzine (1/3)"]
    ),
    'beat_response_intensity': st.sidebar.slider(
        "Intensità risposta beat", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.8, 
        step=0.1
    ),
    'smooth_transitions': st.sidebar.checkbox("Transizioni smooth", value=True)
}

# Color settings
st.sidebar.header("🎨 Colori")
color_settings = {
    'use_frequency_colors': st.sidebar.checkbox("Usa colori frequenza", value=True),
    'background_color': st.sidebar.color_picker("Colore sfondo", "#000000"),
    'low_freq_color': st.sidebar.color_picker("Frequenze basse", "#FF0000"),
    'mid_freq_color': st.sidebar.color_picker("Frequenze medie", "#00FF00"),
    'high_freq_color': st.sidebar.color_picker("Frequenze acute", "#0000FF")
}

# Processing section
if uploaded_file is not None:
    st.success(f"File caricato: {uploaded_file.name}")
    
    # Show video settings info
    est_width, est_height, est_fps, est_size = get_optimal_settings(60, width, height)  # assuming 60s duration
    st.info(f"Impostazioni video: {width}x{height} @ {est_fps}fps (stima: ~{est_size}MB)")
    
    # Process button
    if st.button("🚀 Genera Video", type="primary"):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Prepare audio
                status_text.text("Preparazione audio...")
                audio_path = prepare_audio_file(uploaded_file, temp_dir)
                progress_bar.progress(10)
                
                # Step 2: Analyze audio
                status_text.text("Analisi audio...")
                y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
                duration = len(y) / sr
                progress_bar.progress(20)
                
                # Step 3: Setup video
                status_text.text("Inizializzazione video...")
                fps = 20
                total_frames = int(duration * fps)
                
                # Video writer setup
                temp_video_path = f"{temp_dir}/temp_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                
                progress_bar.progress(30)
                
                # Step 4: Generate frames
                status_text.text("Generazione frames...")
                
                for frame_idx in range(total_frames):
                    current_time = frame_idx / fps
                    
                    # Calculate audio data for current frame
                    start_sample = int(current_time * sr)
                    end_sample = int((current_time + 1/fps) * sr)
                    audio_chunk = y[start_sample:end_sample] if start_sample < len(y) else np.array([])
                    
                    # Process audio data
                    rms, freq_data = process_frame_data(audio_chunk)
                    
                    # Create base frame
                    frame_img = np.full((height, width, 3), hex_to_bgr(color_settings['background_color']), dtype=np.uint8)
                    
                    # Apply fractal based on selection
                    if fractal_type == "Mandelbrot":
                        frame_img = draw_mandelbrot_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, 
                            tempo, freq_data, color_settings, movement_scale, bmp_settings
                        )
                    elif fractal_type == "Julia":
                        frame_img = draw_julia_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, 
                            tempo, freq_data, color_settings, movement_scale, bmp_settings
                        )
                    elif fractal_type == "Burning Ship":
                        frame_img = draw_burning_ship_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, 
                            tempo, freq_data, color_settings, movement_scale, bmp_settings
                        )
                    elif fractal_type == "Sierpinski":
                        frame_img = draw_sierpinski_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, 
                            tempo, freq_data, color_settings, movement_scale, bmp_settings
                        )
                    
                    # Write frame
                    out.write(frame_img)
                    
                    # Update progress
                    progress = 30 + int((frame_idx / total_frames) * 60)
                    progress_bar.progress(progress)
                    
                    if frame_idx % 20 == 0:  # Update status every 20 frames
                        status_text.text(f"Frame {frame_idx}/{total_frames} ({current_time:.1f}s)")
                
                # Release video writer
                out.release()
                progress_bar.progress(90)
                
                # Step 5: Merge audio and video
                status_text.text("Merge audio/video...")
                final_output_path = f"{temp_dir}/synesthetic_output.mp4"
                success, message = merge_video_audio(temp_video_path, audio_path, final_output_path)
                
                if success:
                    progress_bar.progress(100)
                    status_text.text("✅ Video completato!")
                    
                    # Read the final video file
                    with open(final_output_path, 'rb') as f:
                        video_data = f.read()
                    
                    # Provide download
                    st.download_button(
                        label="📥 Scarica Video",
                        data=video_data,
                        file_name=f"synesthetic_{fractal_type.lower()}_{int(time.time())}.mp4",
                        mime="video/mp4"
                    )
                    
                else:
                    st.error(f"Errore nel merge: {message}")
                    
            except Exception as e:
                st.error(f"Errore durante la generazione: {str(e)}")
                
else:
    st.info("👆 Carica un file audio per iniziare")
