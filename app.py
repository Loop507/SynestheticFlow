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
        return 0, 0, 0

    # Normalizza i dati di frequenza per evitare valori eccessivi
    freq_data_norm = freq_data / (np.max(freq_data) + 1e-6) # Aggiungi epsilon per evitare divisione per zero

    total_bins = len(freq_data_norm)
    low_end = total_bins // 3
    mid_end = (total_bins * 2) // 3

    low_freq = np.mean(freq_data_norm[:low_end]) if low_end > 0 else 0
    mid_freq = np.mean(freq_data_norm[low_end:mid_end]) if mid_end > low_end else 0
    high_freq = np.mean(freq_data_norm[mid_end:]) if total_bins > mid_end else 0

    return low_freq, mid_freq, high_freq

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0

    # Applica una finestra per FFT per ridurre artefatti
    windowed_audio_chunk = audio_chunk * np.hanning(len(audio_chunk))
    freq_data = np.abs(np.fft.rfft(windowed_audio_chunk)) if len(windowed_audio_chunk) > 0 else np.array([])

    return rms, freq_data

def hex_to_bgr(hex_color):
    """Converte colore hex in formato BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR format

# --- FUNZIONI DI SUPPORTO PER SINCRONIZZAZIONE BPM ---

def calculate_bpm_phase(current_time, tempo, sync_type, beat_times, bpm_settings):
    """Calcola la fase corrente basata sul BPM per sincronizzazione precisa"""
    if not bpm_settings['enabled']:
        return 0.0, False, 1.0 # Restituisce fase 0, non sul beat, e intensità 1 (neutra)
    
    # Trova il beat più vicino
    # Assicurati che beat_times non sia vuoto
    if len(beat_times) == 0:
        return 0.0, False, 1.0

    beat_distances = np.abs(beat_times - current_time)
    nearest_beat_idx = np.argmin(beat_distances)
    nearest_beat_time = beat_times[nearest_beat_idx]
    
    # Calcola la distanza temporale dal beat più vicino
    time_from_beat = current_time - nearest_beat_time
    
    # Determina se siamo "su un beat" (entro una piccola finestra temporale)
    # Calcola una finestra basata sul tempo del beat per rendere sensibile al BPM
    beat_window = 60.0 / (tempo * 8) if tempo > 0 else 0.1 # Finestra di circa 1/8 del tempo di un beat, evita divisione per zero
    is_on_beat = abs(time_from_beat) <= beat_window
    
    # Calcola la fase in base al tipo di sincronizzazione
    beat_duration = 60.0 / tempo if tempo > 0 else 1.0 # Durata di un beat in secondi, evita divisione per zero
    
    sync_multipliers = {
        "Beat principale (1/1)": 1.0,
        "Mezzo beat (1/2)": 2.0,
        "Doppio beat (2/1)": 0.5,
        "Terzine (1/3)": 3.0
    }
    
    multiplier = sync_multipliers.get(sync_type, 1.0)
    phase_duration = beat_duration / multiplier
    
    # Calcola la fase (0 a 2π nel periodo del beat)
    # Assicurati che phase_duration non sia zero
    if phase_duration == 0:
        phase = 0.0
    else:
        time_in_cycle = (current_time % phase_duration)
        phase = (time_in_cycle / phase_duration) * 2 * np.pi
    
    # Calcola l'intensità del beat (più forte vicino ai beat)
    # Utilizza beat_window per la normalizzazione
    if beat_window == 0:
        beat_intensity = 0.1
    else:
        beat_intensity = max(0.1, 1.0 - (abs(time_from_beat) / beat_window)) if is_on_beat else 0.1
    
    return phase, is_on_beat, beat_intensity

def apply_bpm_movement_modulation(base_value, phase, beat_intensity, modulation_type, bpm_settings):
    """Applica modulazione basata sui BPM a un valore base"""
    if not bpm_settings['enabled']:
        return base_value
    
    intensity = bpm_settings['beat_response_intensity']
    
    modulation = 0
    if modulation_type == 'sine':
        modulation = np.sin(phase) * intensity * beat_intensity
    elif modulation_type == 'cosine':
        modulation = np.cos(phase) * intensity * beat_intensity
    elif modulation_type == 'pulse':
        # Modulazione a impulso per effetti più marcati (più forte sui beat)
        modulation = (np.sin(phase) ** 2) * intensity * beat_intensity # Usato sin(phase)**2 per un impulso morbido
    elif modulation_type == 'sawtooth':
        # Modulazione a dente di sega per movimenti lineari graduali e reset
        modulation = ((phase / (2 * np.pi)) % 1) * 2 * intensity * beat_intensity - intensity * beat_intensity # Da -intensity a +intensity
    else:
        modulation = 0
    
    if bpm_settings['smooth_transitions']:
        # Applica un fattore di smorzamento per transizioni più fluide
        smooth_factor = 0.5 # Puoi regolare questo valore
        modulation *= smooth_factor
    
    return base_value + modulation

# --- FUNZIONI FRATTALI AVANZATE ---

@jit(nopython=True)
def mandelbrot_set_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    """Genera il set di Mandelbrot con influenza audio (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Coordinate complesse con zoom e movimento
            c_real = (float(x) - width/2) / (zoom * width/4) + move_x
            c_imag = (float(y) - height/2) / (zoom * height/4) + move_y

            # Aggiunta influenza audio - modulazione sottile
            c_real += audio_influence * 0.005 * np.sin(float(x) * 0.001)
            c_imag += audio_influence * 0.005 * np.cos(float(y) * 0.001)

            # Inizializza esplicitamente z_real e z_imag come np.float64 per Numba
            z_real = np.float64(0.0)
            z_imag = np.float64(0.0)
            iteration = 0

            # Calcolo iterativo
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            # Colori basati sul numero di iterazioni
            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]  # Nero per punti dentro il set
            else:
                # Gradiente colorato più complesso
                # Mappa l'iterazione su un range 0-1 per il colore
                t = float(iteration) / max_iter
                r = int(9 * (1 - t) * t * t * t * 255)
                g = int(15 * (1 - t) * (1 - t) * t * t * 255)
                b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
                fractal[y, x] = [b, g, r] # BGR per OpenCV

    return fractal

@jit(nopython=True)
def julia_set_numba(width, height, max_iter, c_real_base, c_imag_base, zoom, audio_mod):
    """Genera il set di Julia con parametri dinamici (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    # Modifica i parametri C in base all'audio
    c_real = c_real_base + audio_mod * 0.05
    c_imag = c_imag_base + audio_mod * 0.07

    for y in range(height):
        for x in range(width):
            z_real = (float(x) - width/2) / (zoom * width/4)
            z_imag = (float(y) - height/2) / (zoom * width/4)

            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            # Colori più vivaci per Julia
            if iteration == max_iter:
                fractal[y, x] = [20, 20, 40]
            else:
                t = float(iteration) / max_iter
                # Variazioni di colore psichedeliche
                b = int(255 * (np.sin(t * 10 + 0) * 0.5 + 0.5))
                g = int(255 * (np.sin(t * 10 + 2 * np.pi / 3) * 0.5 + 0.5))
                r = int(255 * (np.sin(t * 10 + 4 * np.pi / 3) * 0.5 + 0.5))
                fractal[y, x] = [b, g, r]

    return fractal

@jit(nopython=True)
def burning_ship_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    """Genera il frattale Burning Ship (Numba ottimizzato)"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            c_real = (float(x) - width/2) / (zoom * width/4) + move_x
            c_imag = (float(y) - height/2) / (zoom * height/4) + move_y

            # Influenza audio
            c_real += audio_influence * 0.003 * np.sin(float(x) * 0.002 + float(y) * 0.001)
            c_imag += audio_influence * 0.003 * np.cos(float(x) * 0.001 + float(y) * 0.002)

            z_real, z_imag = 0.0, 0.0
            iteration = 0

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                # La differenza chiave: valori assoluti
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0 * np.abs(z_real) * np.abs(z_imag) + c_imag # abs() qui
                z_real = z_real_new
                iteration += 1

            # Colori fiammeggianti dinamici
            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]
            else:
                t = float(iteration) / max_iter
                b = int(255 * (np.sin(t * 5 + 0) * 0.5 + 0.5))
                g = int(255 * (np.sin(t * 5 + 1.5) * 0.5 + 0.5))
                r = int(255 * (np.sin(t * 5 + 3) * 0.5 + 0.5))
                fractal[y, x] = [b, g, r]

    return fractal

@jit(nopython=True)
def _remove_squares_numba(arr, level, x, y, size, fill_color):
    """Funzione ricorsiva per Sierpinski Carpet (Numba ottimizzato)"""
    if level == 0 or size < 3:
        return

    third = size // 3

    # Rimuovi il quadrato centrale
    for i in range(third):
        for j in range(third):
            # Controlla i limiti per evitare IndexError
            if x + third + i < arr.shape[0] and y + third + j < arr.shape[1]:
                arr[x + third + i, y + third + j] = fill_color # Usa fill_color (0 o altro)

    # Ricorsione sui quadrati rimanenti
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:  # Salta il quadrato centrale
                _remove_squares_numba(arr, level-1, x + i*third, y + j*third, third, fill_color)

def generate_sierpinski_carpet(width, height, iterations, audio_scale, base_color_bgr):
    """Genera il tappeto di Sierpinski con influenza audio"""
    size = min(width, height)
    # Inizia con un colore di base che verrà poi modulato
    carpet = np.full((size, size, 3), base_color_bgr, dtype=np.uint8)

    # Numero di iterazioni basato sull'audio
    # Limita le iterazioni per evitare calcoli troppo lunghi o immagini vuote
    iter_count = max(1, min(6, int(iterations + audio_scale * 2))) # Calibra audio_scale

    # Colore "vuoto" (nero) per i fori
    _remove_squares_numba(carpet, iter_count, 0, 0, size, (0,0,0)) # Passa il colore di riempimento

    # Ridimensiona il tappeto per adattarlo al frame
    # OpenCV resize è in BGR per default
    fractal = cv2.resize(carpet, (width, height), interpolation=cv2.INTER_AREA)

    return fractal

def apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings):
    """Applica colori basati sulle frequenze al frattale esistente"""
    if not color_settings['use_frequency_colors']:
        return fractal

    colored_fractal = fractal.copy()

    low_bgr = np.array(hex_to_bgr(color_settings['low_freq_color']))
    mid_bgr = np.array(hex_to_bgr(color_settings['mid_freq_color']))
    high_bgr = np.array(hex_to_bgr(color_settings['high_freq_color']))

    # Normalizza le influenze delle frequenze per il blending
    # Usa np.clip per assicurarsi che i valori rimangano tra 0 e 1
    low_intensity = np.clip(low_freq * 5.0, 0.0, 1.0) # Moltiplicatore calibrato
    mid_intensity = np.clip(mid_freq * 5.0, 0.0, 1.0)
    high_intensity = np.clip(high_freq * 5.0, 0.0, 1.0)

    # Crea un colore base mescolando i colori delle frequenze
    # Questa parte applica un colore generale che reagisce all'audio
    mixed_frequency_color = (
        low_bgr * low_intensity +
        mid_bgr * mid_intensity +
        high_bgr * high_intensity
    )
    # Evita divisione per zero e normalizza
    sum_intensity = low_intensity + mid_intensity + high_intensity
    if sum_intensity > 1e-6:
        mixed_frequency_color = np.clip(mixed_frequency_color / sum_intensity, 0, 255)
    else:
        mixed_frequency_color = np.array([0,0,0]) # Se non ci sono intensità, colore nero

    # Applica il colore misto ai pixel del frattale che non sono neri (o colore di sfondo)
    # In questo modo, i colori delle frequenze "infondono" il frattale
    # Solo i pixel del frattale (non il "vuoto" o lo sfondo) vengono colorati
    mask = np.any(colored_fractal != [0,0,0], axis=-1) # Maschera per i pixel non neri del frattale

    # Modula il colore del frattale con il colore misto delle frequenze
    # Un semplice blending o moltiplicazione può dare effetti interessanti
    # Qui usiamo una moltiplicazione per "tingere" il frattale
    for c in range(3): # Per ogni canale di colore B, G, R
        colored_fractal[mask, c] = np.clip(colored_fractal[mask, c] * (mixed_frequency_color[c] / 255.0) * 1.5, 0, 255) # Moltiplica e satura

    return colored_fractal

# --- FUNZIONI DI DISEGNO FRATTALE PER IL PROCESSING (ORA CON BPM SYNC) ---

def draw_mandelbrot_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bpm_settings):
    """Mandelbrot con sincronizzazione BPM avanzata"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    # Calcola fase e intensità BPM
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bpm_settings['movement_sync_type'], beat_times, bpm_settings)
    
    # Parametri dinamici con sincronizzazione BPM
    base_max_iter = 80 + rms * 150 * movement_scale_factor
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'pulse', bpm_settings))
    max_iter = max(50, min(200, max_iter))
    
    # Zoom sincronizzato sui BPM
    base_zoom = 1.5
    bpm_zoom_modulation = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bpm_settings) * 0.5
    audio_zoom_influence = rms * 5 * movement_scale_factor + low_freq * 10 * movement_scale_factor
    zoom = base_zoom + bpm_zoom_modulation + audio_zoom_influence * 0.3
    
    # Movimento sincronizzato
    base_move_x = -0.75
    base_move_y = 0.05
    
    bpm_move_x = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bpm_settings) * 0.1
    bpm_move_y = apply_bpm_movement_modulation(0, phase * 1.3, beat_intensity, 'cosine', bpm_settings) * 0.08
    
    move_x = base_move_x + bpm_move_x + mid_freq * 0.03 * movement_scale_factor
    move_y = base_move_y + bpm_move_y + high_freq * 0.025 * movement_scale_factor
    
    # Influenza audio modulata dai BPM
    base_audio_influence = (rms * 2.0 + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor
    audio_influence = base_audio_influence * (1.0 + beat_intensity * 0.5)
    
    fractal = mandelbrot_set_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    # Alpha blending reattivo ai BPM
    base_alpha = 0.65
    beat_alpha_boost = bpm_settings['beat_response_intensity'] * 0.25 if is_on_beat else 0
    alpha = min(0.95, base_alpha + beat_alpha_boost)
    
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_julia_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bpm_settings):
    """Julia Set con sincronizzazione BPM avanzata"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bpm_settings['movement_sync_type'], beat_times, bpm_settings)
    
    # Iterazioni con modulazione BPM
    base_max_iter = 70 + rms * 80 * movement_scale_factor
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'pulse', bpm_settings))
    max_iter = max(50, min(150, max_iter))
    
    # Parametri C di Julia sincronizzati
    base_c_real = -0.7
    base_c_imag = 0.27015
    
    bpm_c_real_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'sine', bpm_settings) * 0.1
    bpm_c_imag_mod = apply_bpm_movement_modulation(0, phase * 0.8, beat_intensity, 'cosine', bpm_settings) * 0.08
    
    c_real_base = base_c_real + bpm_c_real_mod
    c_imag_base = base_c_imag + bpm_c_imag_mod
    
    # Zoom con BPM sync
    base_zoom = 1.0
    bpm_zoom_mod = apply_bpm_movement_modulation(0, phase * 1.5, beat_intensity, 'sine', bpm_settings) * 0.3
    zoom = base_zoom + bpm_zoom_mod + rms * 1.5 * movement_scale_factor + high_freq * 2.0 * movement_scale_factor
    
    # Audio modulation con BPM
    base_audio_mod = (rms * 1.5 + (low_freq * 0.5 + mid_freq * 0.8 + high_freq * 0.2)) * movement_scale_factor
    audio_mod = base_audio_mod * (1.0 + beat_intensity * 0.6)
    
    fractal = julia_set_numba(width, height, max_iter, c_real_base, c_imag_base, zoom, audio_mod)
    
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    alpha = min(0.95, 0.75 + beat_intensity * bpm_settings['beat_response_intensity'] * 0.2)
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_burning_ship_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bpm_settings):
    """Burning Ship con sincronizzazione BPM"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bpm_settings['movement_sync_type'], beat_times, bpm_settings)
    
    base_max_iter = 60 + rms * 60 * movement_scale_factor
    max_iter = int(apply_bpm_movement_modulation(base_max_iter, phase, beat_intensity, 'sawtooth', bpm_settings))
    max_iter = max(40, min(120, max_iter))
    
    base_zoom = 1.0
    bpm_zoom_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bpm_settings) * 0.2
    zoom = base_zoom + bpm_zoom_mod + rms * 1.5 * movement_scale_factor + mid_freq * 2.0 * movement_scale_factor
    
    base_move_x, base_move_y = -1.8, -0.08
    bpm_move_x = apply_bpm_movement_modulation(0, phase * 1.2, beat_intensity, 'sine', bpm_settings) * 0.05
    bpm_move_y = apply_bpm_movement_modulation(0, phase * 0.9, beat_intensity, 'cosine', bpm_settings) * 0.03
    
    move_x = base_move_x + bpm_move_x
    move_y = base_move_y + bpm_move_y
    
    audio_influence = (rms * 1.0 + high_freq * 0.5) * movement_scale_factor * (1.0 + beat_intensity * 0.4)
    
    fractal = burning_ship_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    alpha = min(0.9, 0.7 + beat_intensity * bpm_settings['beat_response_intensity'] * 0.15)
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_sierpinski_fractal_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bpm_settings):
    """Sierpinski Carpet con sincronizzazione BPM"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bpm_settings['movement_sync_type'], beat_times, bpm_settings)
    
    base_iterations = 3 + int(rms * 3 * movement_scale_factor) + int(low_freq * 2 * movement_scale_factor)
    bpm_iter_mod = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bpm_settings) * 2
    iterations = int(base_iterations + bpm_iter_mod)
    
    base_audio_scale = (rms + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor
    audio_scale = base_audio_scale * (1.0 + beat_intensity * 0.5)
    
    base_carpet_color_bgr = hex_to_bgr(color_settings['background_color'])
    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale, base_carpet_color_bgr)
    
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    
    alpha = min(0.85, 0.6 + beat_intensity * bpm_settings['beat_response_intensity'] * 0.2)
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

# Selezione formato video
st.subheader("📐 Formato Video")
selected_format = st.selectbox(
    "Scegli il rapporto di aspetto:",
    list(VIDEO_FORMATS.keys()),
    index=0
)

width, height = VIDEO_FORMATS[selected_format]
st.info(f"📺 Formato selezionato: **{selected_format}** - Risoluzione: {width}x{height}px")

# --- CONTROLLI FRATTALI ---
st.subheader("🌀 Tipo di Effetto") # Modificato qui

fractal_type = st.selectbox(
    "Seleziona l'effetto da generare:", # Modificato qui
    [
        "🌀 Mandelbrot Set - Classico e ipnotico",
        "🔥 Julia Set - Dinamico e fluido",
        "🚢 Burning Ship - Forme organiche",
        "📐 Sierpinski Carpet - Geometrico"
    ],
    index=0
)

# --- CONTROLLI MOVIMENTO EFFETTI ---
st.subheader("⚙️ Intensità Movimento Effetti")
movement_intensity = st.selectbox(
    "Seleziona l'intensità del movimento degli effetti:",
    ["Soft", "Medium", "Hard"],
    index=1 # Default to Medium
)

# Define scaling factors based on movement intensity
movement_scale_factors = {
    "Soft": 0.5,
    "Medium": 1.0,
    "Hard": 1.5
}
current_movement_scale_factor = movement_scale_factors[movement_intensity]

# --- CONTROLLI COLORI ---
st.subheader("🎨 Controlli Colori")

col1, col2 = st.columns(2)

with col1:
    use_frequency_colors = st.checkbox("🌈 Effetti generativi", value=True)
    background_color = st.color_picker("🖤 Colore sfondo", value="#000000")

with col2:
    if use_frequency_colors:
        st.markdown("**Colori Frequenze:**")
        low_freq_color = st.color_picker("🔴 Frequenze Basse", value="#FF0066") # Rosa vivace
        mid_freq_color = st.color_picker("🟢 Frequenze Medie", value="#00FF88") # Verde acqua
        high_freq_color = st.color_picker("🔵 Frequenze Acute", value="#0066FF") # Blu elettrico
    else:
        # Se non usiamo colori per frequenze, manteniamo un default logico per fallback
        low_freq_color = "#FF0066"
        mid_freq_color = "#00FF88"
        high_freq_color = "#0066FF"
        st.info("Abilita 'Effetti generativi' per personalizzare")

color_settings = {
    'use_frequency_colors': use_frequency_colors,
    'background_color': background_color,
    'low_freq_color': low_freq_color,
    'mid_freq_color': mid_freq_color,
    'high_freq_color': high_freq_color
}

# --- CONTROLLI SINCRONIZZAZIONE BPM ---
st.subheader("🎵 Sincronizzazione BPM")

col_bpm1, col_bpm2 = st.columns(2) # Modificato il nome della variabile da col_bmp2 a col_bpm2 per coerenza

with col_bpm1:
    bpm_sync_enabled = st.checkbox("🎯 Sincronizza movimento sui BPM", value=True)
    beat_response_intensity = st.slider(
        "📈 Intensità risposta ai beat", 
        min_value=0.1, max_value=3.0, value=1.0, step=0.1
    )

with col_bpm2:
    if bpm_sync_enabled:
        movement_sync_type = st.selectbox(
            "🔄 Tipo di sincronizzazione movimento:",
            [
                "Beat principale (1/1)", 
                "Mezzo beat (1/2)", 
                "Doppio beat (2/1)",
                "Terzine (1/3)"
            ],
            index=0
        )
        
        smooth_transitions = st.checkbox("🌊 Transizioni fluide", value=True)
    else:
        movement_sync_type = "Beat principale (1/1)"
        smooth_transitions = True

# Crea il dizionario delle impostazioni BPM
bpm_settings = {
    'enabled': bpm_sync_enabled,
    'beat_response_intensity': beat_response_intensity,
    'movement_sync_type': movement_sync_type,
    'smooth_transitions': smooth_transitions
}

uploaded_audio = st.file_uploader("🎵 Carica file audio", type=["wav", "mp3", "ogg"])

if uploaded_audio:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = prepare_audio_file(uploaded_audio, temp_dir)
            y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
            duration = librosa.get_duration(filename=audio_path)
            width, height, fps, est_size = get_optimal_settings(duration, width, height)

            st.info(f"🎼 BPM: {float(tempo):.0f} | ⏱️ {float(duration):.1f}s | 🎮 {fps} FPS | 📆 ~{est_size} MB")

            if st.button("🚀 CREA VIDEO FRATTALE"):
                video_temp = os.path.join(temp_dir, "video_temp.mp4")
                video_final = os.path.join(temp_dir, "video_with_audio.mp4")

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_temp, fourcc, fps, (width, height))
                video_placeholder = st.empty()
                progress_bar = st.progress(0)

                frame_count = int(duration * fps)
                frame_duration = 1.0 / fps
                bg_color_bgr = hex_to_bgr(color_settings['background_color'])

                # Mostra informazioni di sincronizzazione BPM
                if bpm_settings['enabled']:
                    st.info(f"🎵 Sincronizzazione BPM attiva - Tipo: {bpm_settings['movement_sync_type']} - Intensità: {bpm_settings['beat_response_intensity']:.1f}")
                
                st.info("🌌 Generazione frattali procedurali con sincronizzazione BPM...")

                for frame_idx in range(frame_count):
                    current_time = frame_idx * frame_duration

                    start_sample = int(current_time * sr)
                    end_sample = start_sample + int(frame_duration * sr)
                    audio_chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                    rms, freq_data = process_frame_data(audio_chunk)

                    frame_img = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)

                    # Usa le funzioni di frattali con sincronizzazione BPM
                    if "Mandelbrot" in fractal_type:
                        frame_img = draw_mandelbrot_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, tempo, 
                            freq_data, color_settings, current_movement_scale_factor, bpm_settings
                        )
                    elif "Julia" in fractal_type:
                        frame_img = draw_julia_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, tempo,
                            freq_data, color_settings, current_movement_scale_factor, bpm_settings
                        )
                    elif "Burning Ship" in fractal_type:
                        frame_img = draw_burning_ship_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, tempo,
                            freq_data, color_settings, current_movement_scale_factor, bpm_settings
                        )
                    elif "Sierpinski" in fractal_type:
                        frame_img = draw_sierpinski_fractal_bpm_sync(
                            frame_img, width, height, rms, current_time, beat_times, tempo,
                            freq_data, color_settings, current_movement_scale_factor, bpm_settings
                        )

                    video_writer.write(frame_img)
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    progress = (frame_idx + 1) / frame_count
                    progress_bar.progress(progress)

                    time.sleep(frame_duration * 0.05)  # Anteprima più veloce

                video_writer.release()
                st.success("✅ Frattali con sincronizzazione BPM generati con successo!")
                
                st.info("🎵 Sincronizzazione audio...")
                success, message = merge_video_audio(video_temp, audio_path, video_final)

                if success:
                    st.success("🎉 Video frattale con audio completato!")
                    with open(video_final, "rb") as f:
                        video_bytes = f.read()

                    format_name = selected_format.split(" ")[0].replace(":", "x")
                    fractal_name = fractal_type.split(" ")[1].lower() # Ora non c'è il mix
                    filename = f"fractal_{fractal_name}_{format_name}_{width}x{height}.mp4"

                    st.download_button(
                        "⬇️ Scarica Video Frattale Completo",
                        video_bytes,
                        file_name=filename,
                        mime="video/mp4"
                    )
                else:
                    st.warning(f"⚠️ {message}")
                    with open(video_temp, "rb") as f:
                        video_bytes = f.read()

                    format_name = selected_format.split(" ")[0].replace(":", "x")
                    fractal_name = fractal_type.split(" ")[1].lower()
                    filename = f"fractal_{fractal_name}_{format_name}_{width}x{height}_no_audio.mp4"

                    st.download_button(
                        "⬇️ Scarica Video Frattale (solo visivo)",
                        video_bytes,
                        file_name=filename,
                        mime="video/mp4"
                    )

        except Exception as e:
            st.error(f"❌ Errore generazione video: {e}")

# Informazioni tecniche nell'expander
with st.expander("🌌 Informazioni Frattali"):
    st.markdown("""
    **SynestheticFlow** genera frattali matematici complessi sincronizzati con l'audio e i BPM:

    **🌀 Tipi di Frattali Disponibili:**

    - **Mandelbrot Set**: Il frattale più famoso, genera infinite spirali e forme organiche.
    - **Julia Set**: Forme fluide e dinamiche che cambiano costantemente.
    - **Burning Ship**: Crea strutture che ricordano navi e paesaggi bruciati.
    - **Sierpinski Carpet**: Pattern geometrici auto-simili.

    **🎵 Sincronizzazione BPM Avanzata:**
    - **Beat Detection Preciso**: Calcolo della fase esatta rispetto ai BPM rilevati.
    - **Tipi di Sincronizzazione**:
        - *Beat principale (1/1)*: Movimento sincronizzato su ogni beat
        - *Mezzo beat (1/2)*: Movimento doppio rispetto al beat
        - *Doppio beat (2/1)*: Movimento più lento, ogni due beat
        - *Terzine (1/3)*: Movimento triplo per ritmi complessi
    - **Intensità Risposta**: Controllo della forza della reazione ai beat (0.1x - 3.0x)
    - **Transizioni Fluide**: Filtro passa-basso per movimenti più organici

    **🎵 Reattività Audio Classica:**
    - **RMS (Volume)**: Controlla zoom, intensità e velocità di morphing.
    - **Frequenze Basse**: Influenzano movimento orizzontale e parametri base.
    - **Frequenze Medie**: Controllano movimento verticale e dettagli.
    - **Frequenze Acute**: Modulano zoom e distorsioni.
    - **Beat Detection**: Intensifica colori e blending durante i colpi ritmici.

    **⚡ Modulazioni BPM per Frattale:**
    - **Zoom Dinamico**: Pulsazioni sincronizzate sui beat per effetti di "respirazione"
    - **Movimento Parametrico**: Traslazioni X/Y che seguono esattamente il tempo musicale
    - **Iterazioni Variabili**: Il dettaglio del frattale cambia a tempo di musica
    - **Alpha Blending Ritmico**: L'opacità dei frattali "pompa" sui beat

    **⚙️ Intensità Movimento Effetti:**
    - **Soft**: Movimenti più lenti e sottili (0.5x).
    - **Medium**: Movimenti bilanciati e reattivi (1.0x).
    - **Hard**: Movimenti rapidi e marcati per un effetto più psichedelico (1.5x).

    **⚡ Ottimizzazioni:**
    - Algoritmi compilati con Numba per performance superiori.
    - Calcoli paralleli per rendering in tempo quasi reale.
    - Gestione memoria efficiente per video lunghi.
    - Calcolo di fase BPM ottimizzato per precisione ritmica.

    **🎨 Sistema Colori Avanzato:**
    - Modulazione colore dinamica basata sull'intensità delle bande di frequenza.
    - Blending intelligente tra sfondo e frattale.
    - Colori predefiniti vivaci per un impatto visivo immediato.
    - Intensificazione cromatica sui beat per sincronizzazione visiva.

    **🔧 Algoritmo di Sincronizzazione BPM:**
    1. **Analisi Temporale**: Calcolo della distanza dal beat più vicino
    2. **Calcolo di Fase**: Determinazione della posizione nel ciclo ritmico (0-2π)
    3. **Modulazione Matematica**: Applicazione di funzioni sinusoidali, coseno, impulso o dente di sega
    4. **Blending Dinamico**: Combinazione tra movimento base e modulazione BPM
    5. **Filtri di Smorzamento**: Opzione per transizioni più fluide

    **Requisiti**: `ffmpeg` installato sul sistema per la fusione audio/video.
    """)
