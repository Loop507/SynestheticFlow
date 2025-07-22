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
    """Analizza le bande di frequenza (basse, medie, acute) con maggiore sensibilità"""
    if len(freq_data) == 0:
        return 0, 0, 0

    # Normalizza i dati di frequenza per evitare valori eccessivi
    freq_data_norm = freq_data / (np.max(freq_data) + 1e-6)

    total_bins = len(freq_data_norm)
    low_end = total_bins // 4  # Più preciso per le basse
    mid_end = (total_bins * 3) // 4  # Più spazio per le medie

    # Calcola con maggiore amplificazione per reattività
    low_freq = np.mean(freq_data_norm[:low_end]) ** 0.7 if low_end > 0 else 0
    mid_freq = np.mean(freq_data_norm[low_end:mid_end]) ** 0.6 if mid_end > low_end else 0
    high_freq = np.mean(freq_data_norm[mid_end:]) ** 0.5 if total_bins > mid_end else 0

    # Amplifica ulteriormente per maggiore reattività
    return min(1.0, low_freq * 2.5), min(1.0, mid_freq * 2.2), min(1.0, high_freq * 2.8)

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

def create_smooth_movement_parameters(frame_idx, tempo, movement_scale):
    """Crea parametri di movimento fluidi e costanti basati su onde sinusoidali multiple"""
    base_time = frame_idx * 0.02 * movement_scale  # Tempo base più fluido
    tempo_factor = max(0.5, tempo / 120.0)  # Normalizza BPM
    
    # Onde sinusoidali multiple per movimento complesso ma fluido
    wave1 = np.sin(base_time * tempo_factor)
    wave2 = np.sin(base_time * tempo_factor * 1.618) * 0.6
    wave3 = np.cos(base_time * tempo_factor * 0.786) * 0.4
    
    # Parametri di movimento combinati
    x_movement = (wave1 + wave2 * 0.5) * 0.15
    y_movement = (wave2 + wave3 * 0.7) * 0.12
    rotation = (wave1 * wave3) * 0.1
    scale_oscillation = 1.0 + (wave2 + wave3) * 0.2
    
    return x_movement, y_movement, rotation, scale_oscillation

def create_dynamic_color_palette(low_freq, mid_freq, high_freq, base_colors, frame_idx, beat_intensity):
    """Crea una palette di colori dinamica basata sulle frequenze audio"""
    
    # Colori base da hex a BGR
    low_bgr = np.array(hex_to_bgr(base_colors['low_freq_color']), dtype=np.float32)
    mid_bgr = np.array(hex_to_bgr(base_colors['mid_freq_color']), dtype=np.float32)
    high_bgr = np.array(hex_to_bgr(base_colors['high_freq_color']), dtype=np.float32)
    
    # Intensità amplificate delle frequenze
    low_intensity = np.clip(low_freq * 3.5 + beat_intensity * 0.5, 0.0, 1.0)
    mid_intensity = np.clip(mid_freq * 3.2 + beat_intensity * 0.3, 0.0, 1.0)
    high_intensity = np.clip(high_freq * 4.0 + beat_intensity * 0.7, 0.0, 1.0)
    
    # Modulazione temporale per colori che cambiano nel tempo
    time_mod = np.sin(frame_idx * 0.01) * 0.5 + 0.5
    
    # Combina colori con modulazione temporale
    dynamic_low = low_bgr * (low_intensity * (0.7 + 0.3 * time_mod))
    dynamic_mid = mid_bgr * (mid_intensity * (0.8 + 0.2 * np.sin(frame_idx * 0.015)))
    dynamic_high = high_bgr * (high_intensity * (0.6 + 0.4 * np.cos(frame_idx * 0.008)))
    
    # Colore finale mixato
    total_intensity = low_intensity + mid_intensity + high_intensity + 0.1
    mixed_color = (dynamic_low + dynamic_mid + dynamic_high) / total_intensity
    
    # Saturazione dinamica basata sull'energia totale
    energy = (low_freq + mid_freq + high_freq) / 3.0
    saturation_boost = 1.0 + energy * 1.5 + beat_intensity * 0.8
    
    mixed_color = np.clip(mixed_color * saturation_boost, 0, 255).astype(np.uint8)
    
    return mixed_color, low_intensity, mid_intensity, high_intensity

def apply_advanced_color_modulation(fractal, color_data, frame_idx):
    """Applica modulazione colore avanzata al frattale"""
    mixed_color, low_int, mid_int, high_int = color_data
    
    # Crea maschere per diverse zone dell'immagine
    height, width = fractal.shape[:2]
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Zone concentriche per effetti colore diversificati
    center_x, center_y = width // 2, height // 2
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distance = distance_from_center / max_distance
    
    # Maschera per pixel non neri (parti attive del frattale)
    active_mask = np.any(fractal != [0, 0, 0], axis=-1)
    
    # Modulazione colore per zone diverse
    for c in range(3):  # B, G, R
        # Zona centrale: colori più intensi
        central_zone = normalized_distance < 0.3
        mid_zone = (normalized_distance >= 0.3) & (normalized_distance < 0.7)
        outer_zone = normalized_distance >= 0.7
        
        # Applica modulazione per zona
        central_mod = mixed_color[c] * (1.2 + low_int * 0.8) / 255.0
        mid_mod = mixed_color[c] * (1.0 + mid_int * 0.6) / 255.0
        outer_mod = mixed_color[c] * (0.9 + high_int * 1.0) / 255.0
        
        # Combina le modulazioni
        color_mask = np.where(central_zone, central_mod, 
                             np.where(mid_zone, mid_mod, outer_mod))
        
        # Applica ai pixel attivi del frattale
        fractal[active_mask, c] = np.clip(
            fractal[active_mask, c] * color_mask[active_mask] * 1.4, 0, 255
        )
    
    # Aggiungi effetto "pulse" sui beat
    pulse_effect = 0.02 * np.sin(frame_idx * 0.1)
    fractal[active_mask] = np.clip(fractal[active_mask] * (1.0 + pulse_effect), 0, 255)
    
    return fractal

# --- FUNZIONI FRATTALI AVANZATE CON MIGLIORAMENTI ---

@jit(nopython=True)
def mandelbrot_set_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence):
    """Mandelbrot con rotazione e movimento fluido migliorati"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)

    for y in range(height):
        for x in range(width):
            # Coordinate con rotazione
            centered_x = (x - width/2) / (zoom * width/4)
            centered_y = (y - height/2) / (zoom * height/4)
            
            # Applica rotazione
            rotated_x = centered_x * cos_rot - centered_y * sin_rot
            rotated_y = centered_x * sin_rot + centered_y * cos_rot
            
            # Posizione finale con movimento
            # Casting esplicito a float per evitare problemi di inferenza di tipo con Numba
            c_real = rotated_x + move_x + audio_influence * 0.008 * float(np.sin(x * 0.002))
            c_imag = rotated_y + move_y + audio_influence * 0.008 * float(np.cos(y * 0.002))

            z_real, z_imag = 0.0, 0.0
            iteration = 0

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]
            else:
                # Gradiente colorato più ricco
                t = float(iteration) / max_iter
                smooth_t = t + 1 - np.log2(np.log2(z_real*z_real + z_imag*z_imag))
                
                r = int(np.sin(smooth_t * 12 + 0) * 127 + 128)
                g = int(np.sin(smooth_t * 12 + 2) * 127 + 128)
                b = int(np.sin(smooth_t * 12 + 4) * 127 + 128)
                fractal[y, x] = [b, g, r]

    return fractal

@jit(nopython=True)
def julia_set_enhanced(width, height, max_iter, c_real_base, c_imag_base, zoom, rotation, audio_mod):
    """Julia Set con rotazione e modulazione audio migliorata"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)

    # Casting esplicito a float per evitare problemi di inferenza di tipo con Numba
    c_real = c_real_base + audio_mod * 0.08 * float(np.sin(audio_mod * 10))
    c_imag = c_imag_base + audio_mod * 0.09 * float(np.cos(audio_mod * 12))
    
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)

    for y in range(height):
        for x in range(width):
            centered_x = (x - width/2) / (zoom * width/4)
            centered_y = (y - height/2) / (zoom * width/4)
            
            # Rotazione
            z_real = centered_x * cos_rot - centered_y * sin_rot
            z_imag = centered_x * sin_rot + centered_y * cos_rot

            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1

            if iteration == max_iter:
                fractal[y, x] = [25, 25, 50]
            else:
                t = float(iteration) / max_iter
                # Colori più dinamici e psichedelici
                b = int(255 * (np.sin(t * 15 + audio_mod * 5) * 0.5 + 0.5))
                g = int(255 * (np.sin(t * 15 + audio_mod * 7 + 2.1) * 0.5 + 0.5))
                r = int(255 * (np.sin(t * 15 + audio_mod * 6 + 4.2) * 0.5 + 0.5))
                fractal[y, x] = [b, g, r]

    return fractal

@jit(nopython=True)
def burning_ship_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence):
    """Burning Ship con effetti di rotazione e audio migliorati"""
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)

    for y in range(height):
        for x in range(width):
            centered_x = (x - width/2) / (zoom * width/4)
            centered_y = (y - height/2) / (zoom * height/4)
            
            rotated_x = centered_x * cos_rot - centered_y * sin_rot
            rotated_y = centered_x * sin_rot + centered_y * cos_rot
            
            # Casting esplicito a float per evitare problemi di inferenza di tipo con Numba
            c_real = rotated_x + move_x + audio_influence * 0.006 * float(np.sin(x * 0.003 + y * 0.002))
            c_imag = rotated_y + move_y + audio_influence * 0.006 * float(np.cos(x * 0.002 + y * 0.003))

            z_real, z_imag = 0.0, 0.0
            iteration = 0

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0 * np.abs(z_real) * np.abs(z_imag) + c_imag
                z_real = z_real_new
                iteration += 1

            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]
            else:
                t = float(iteration) / max_iter
                # Colori "fiammeggianti" più realistici
                b = int(255 * np.clip(np.sin(t * 8 + 0) * 0.3 + 0.1, 0, 1))
                g = int(255 * np.clip(np.sin(t * 8 + 1.5) * 0.6 + 0.3, 0, 1))
                r = int(255 * np.clip(np.sin(t * 8 + 3) * 0.8 + 0.7, 0, 1))
                fractal[y, x] = [b, g, r]

    return fractal

# --- FUNZIONI DI DISEGNO MIGLIORATE ---

def draw_mandelbrot_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Mandelbrot con movimento fluido e colori reattivi migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    # Parametri di movimento fluidi
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor)
    
    # Parametri audio-reattivi migliorati
    max_iter = max(60, min(250, int(100 + rms * 200 * movement_scale_factor)))
    zoom = (1.8 + rms * 4 * movement_scale_factor + low_freq * 8 * movement_scale_factor) * scale_osc
    move_x = x_move - 0.75 + mid_freq * 0.15 * movement_scale_factor
    move_y = y_move + 0.05 + high_freq * 0.12 * movement_scale_factor
    audio_influence = (rms * 2.5 + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor
    
    fractal = mandelbrot_set_enhanced(width,
        height, max_iter, zoom, move_x, move_y, rotation, audio_influence
    )

    if color_settings['use_frequency_colors']:
        beat_intensity = 1.5 if beat else 1.0
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity)
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx)

    alpha = 0.85 if beat else 0.75
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def draw_julia_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Julia Set con movimento e colori migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor)
    
    max_iter = max(60, min(180, int(90 + rms * 120 * movement_scale_factor)))
    c_real_base = -0.7 + x_move * 0.5
    c_imag_base = 0.27015 + y_move * 0.4
    zoom = (1.2 + rms * 2 * movement_scale_factor + high_freq * 3.0 * movement_scale_factor) * scale_osc
    audio_mod = (rms * 2.0 + (low_freq * 0.6 + mid_freq + high_freq * 0.4)) * movement_scale_factor

    fractal = julia_set_enhanced(width, height, max_iter, c_real_base, c_imag_base, zoom, rotation, audio_mod)

    if color_settings['use_frequency_colors']:
        beat_intensity = 1.8 if beat else 1.0
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity)
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx)

    alpha = 0.9 if beat else 0.8
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def draw_burning_ship_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Burning Ship con movimento fluido migliorato"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor)
    
    max_iter = max(50, min(150, int(80 + rms * 90 * movement_scale_factor)))
    zoom = (1.2 + rms * 2.5 * movement_scale_factor + mid_freq * 3.5 * movement_scale_factor) * scale_osc
    move_x = -1.8 + x_move * 0.3
    move_y = -0.08 + y_move * 0.2
    audio_influence = (rms * 1.5 + high_freq * 0.8) * movement_scale_factor

    fractal = burning_ship_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence)

    if color_settings['use_frequency_colors']:
        beat_intensity = 1.6 if beat else 1.0
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity)
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx)

    alpha = 0.88 if beat else 0.78
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

# Mantieni le funzioni originali per Sierpinski e merge video
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
                arr[x + third + i, y + third + j] = fill_color

    # Ricorsione sui quadrati rimanenti
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:  # Salta il quadrato centrale
                _remove_squares_numba(arr, level-1, x + i*third, y + j*third, third, fill_color)

def generate_sierpinski_carpet(width, height, iterations, audio_scale, base_color_bgr):
    """Genera il tappeto di Sierpinski con influenza audio"""
    size = min(width, height)
    carpet = np.full((size, size, 3), base_color_bgr, dtype=np.uint8)

    iter_count = max(1, min(6, int(iterations + audio_scale * 2)))
    _remove_squares_numba(carpet, iter_count, 0, 0, size, (0,0,0))
    
    fractal = cv2.resize(carpet, (width, height), interpolation=cv2.INTER_AREA)
    return fractal

def draw_sierpinski_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Sierpinski con colori migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)

    iterations = 3 + int(rms * 4 * movement_scale_factor) + int(low_freq * 3 * movement_scale_factor)
    audio_scale = (rms + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor

    base_carpet_color_bgr = hex_to_bgr(color_settings['background_color'])
    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale, base_carpet_color_bgr)

    if color_settings['use_frequency_colors']:
        beat_intensity = 1.4 if beat else 1.0
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity)
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx)

    alpha = 0.8 if beat else 0.7
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def merge_video_audio(video_path, audio_path, output_path):
    """Combina video e audio usando ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
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

st.title(" **SynestheticFlow Enhanced**")
st.markdown("*<span style='font-size: 12px;'>Visualizer Potenziato by loop507</span>*", unsafe_allow_html=True)

# Selezione formato video
st.subheader(" Formato Video")
selected_format = st.selectbox(
    "Scegli il rapporto di aspetto:",
    list(VIDEO_FORMATS.keys()),
    index=0
)

width, height = VIDEO_FORMATS[selected_format]
st.info(f" Formato selezionato: **{selected_format}** - Risoluzione: {width}x{height}px")

# --- CONTROLLI FRATTALI ---
st.subheader(" Tipo di Effetto")

fractal_type = st.selectbox(
    "Seleziona l'effetto da generare:",
    [
        " Mandelbrot Set - Movimento fluido e rotazione",
        " Julia Set - Dinamico con rotazione avanzata",
        " Burning Ship - Forme organiche rotanti",
        " Sierpinski Carpet - Geometrico colorato"
    ],
    index=0
)

# --- CONTROLLI MOVIMENTO EFFETTI ---
st.subheader(" Intensità Movimento Effetti")
movement_intensity = st.selectbox(
    "Seleziona l'intensità del movimento degli effetti:",
    ["Soft", "Medium", "Hard", "Extreme"],
    index=1
)

movement_scale_factors = {
    "Soft": 0.4,
    "Medium": 1.0,
    "Hard": 1.8,
    "Extreme": 2.5
}
current_movement_scale_factor = movement_scale_factors[movement_intensity]

# --- CONTROLLI COLORI AVANZATI ---
st.subheader(" Controlli Colori Avanzati")

col1, col2 = st.columns(2)

with col1:
    use_frequency_colors = st.checkbox(" Colori Reattivi Audio", value=True)
    background_color = st.color_picker(" Colore sfondo", value="#000000")
    
    if use_frequency_colors:
        st.markdown("** Intensità Reazione Audio:**")
        audio_color_intensity = st.slider("Intensità colori", 0.5, 3.0, 1.5, 0.1)

with col2:
    if use_frequency_colors:
        st.markdown("**Palette Frequenze Audio:**")
        low_freq_color = st.color_picker(" Basse (Sub-Bass/Bass)", value="#FF1744")
        mid_freq_color = st.color_picker(" Medie (Vocal/Lead)", value="#00E676")
        high_freq_color = st.color_picker(" Acute (Hi-Hat/Treble)", value="#2196F3")
    else:
        low_freq_color = "#FF1744"
        mid_freq_color = "#00E676"
        high_freq_color = "#2196F3"
        st.info("Abilita 'Colori Reattivi Audio' per personalizzare")

color_settings = {
    'use_frequency_colors': use_frequency_colors,
    'background_color': background_color,
    'low_freq_color': low_freq_color,
    'mid_freq_color': mid_freq_color,
    'high_freq_color': high_freq_color,
    'audio_color_intensity': audio_color_intensity if use_frequency_colors else 1.0
}

# --- UPLOAD E CONTROLLI ---
st.subheader(" Upload Audio")
uploaded_file = st.file_uploader(
    "Carica il file audio:",
    type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
    help="Formati supportati: WAV, MP3, OGG, FLAC, M4A"
)

if uploaded_file:
    st.success(f" File caricato: **{uploaded_file.name}**")
    
    # Controlli video
    st.subheader(" Controlli Video")
    col1, col2 = st.columns(2)
    
    with col1:
        video_duration = st.slider(" Durata video (sec)", 10, 300, 60, 5)
    
    with col2:
        video_quality = st.selectbox(
            " Qualità video:",
            ["Alta (20 FPS)", "Media (15 FPS)", "Veloce (12 FPS)"],
            index=0
        )
    
    fps_settings = {
        "Alta (20 FPS)": 20,
        "Media (15 FPS)": 15,
        "Veloce (12 FPS)": 12
    }
    fps = fps_settings[video_quality]
    
    # Stima dimensioni
    estimated_frames = video_duration * fps
    estimated_size_mb = (width * height * estimated_frames * 3) / (1024 * 1024)
    
    st.info(f" **Stima:** {estimated_frames} frame, ~{estimated_size_mb:.1f} MB")
    
    # Pulsante generazione
    if st.button(" **Genera Video Visualizer**", type="primary"):
        
        # Setup progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                status_text.text(" Analisi audio in corso...")
                
                # Prepara file audio
                audio_path = prepare_audio_file(uploaded_file, temp_dir)
                y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
                
                st.success(f"🎼 Audio analizzato - BPM: {float(tempo):.1f}")
            
                # Calcola parametri video
                total_frames = video_duration * fps
                hop_length = len(y) // total_frames
                
                # Setup video writer
                video_path = f"{temp_dir}/fractal_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                if not video_writer.isOpened():
                    st.error(" Errore nell'inizializzazione del video writer")
                    st.stop()
                
                status_text.text(" Generazione frame in corso...")
                
                # Genera frame
                for frame_idx in range(total_frames):
                    # Calcola posizione audio
                    start_sample = frame_idx * hop_length
                    end_sample = min(start_sample + hop_length, len(y))
                    audio_chunk = y[start_sample:end_sample]
                    
                    # Analisi audio frame
                    rms, freq_data = process_frame_data(audio_chunk)
                    
                    # Rileva beat
                    current_time = frame_idx / fps
                    beat = any(abs(current_time - bt) < 0.1 for bt in beat_times)
                    
                    # Crea frame base
                    frame_img = np.full((height, width, 3), hex_to_bgr(background_color), dtype=np.uint8)
                    
                    # Disegna frattale selezionato
                    if "Mandelbrot" in fractal_type:
                        frame_img = draw_mandelbrot_fractal_enhanced(
                            frame_img, width, height, rms, frame_idx, beat, 
                            freq_data, color_settings, current_movement_scale_factor, tempo
                        )
                    elif "Julia" in fractal_type:
                        frame_img = draw_julia_fractal_enhanced(
                            frame_img, width, height, rms, frame_idx, beat, 
                            freq_data, color_settings, current_movement_scale_factor, tempo
                        )
                    elif "Burning Ship" in fractal_type:
                        frame_img = draw_burning_ship_fractal_enhanced(
                            frame_img, width, height, rms, frame_idx, beat, 
                            freq_data, color_settings, current_movement_scale_factor, tempo
                        )
                    elif "Sierpinski" in fractal_type:
                        frame_img = draw_sierpinski_fractal_enhanced(
                            frame_img, width, height, rms, frame_idx, beat, 
                            freq_data, color_settings, current_movement_scale_factor, tempo
                        )
                    
                    # Scrivi frame
                    video_writer.write(frame_img)
                    
                    # Aggiorna progress
                    progress = (frame_idx + 1) / total_frames
                    progress_bar.progress(progress)
                    
                    if frame_idx % 10 == 0:
                        status_text.text(f" Frame {frame_idx + 1}/{total_frames} ({progress*100:.1f}%)")
                
                video_writer.release()
                
                # Merge audio e video
                status_text.text(" Merge audio e video...")
                output_path = f"{temp_dir}/final_video.mp4"
                
                success, message = merge_video_audio(video_path, audio_path, output_path)
                
                if success:
                    status_text.text(" Video completato!")
                    progress_bar.progress(1.0)
                    
                    # Download
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    st.success(" **Video generato con successo!**")
                    st.download_button(
                        label=" **Scarica Video**",
                        data=video_bytes,
                        file_name=f"synesthetic_flow_{fractal_type.split()[1].lower()}_{movement_intensity.lower()}.mp4",
                        mime="video/mp4"
                    )
                    
                    # Mostra video preview
                    st.video(video_bytes)
                    
                else:
                    st.error(f" Errore nel merge: {message}")
                    
                    # Offri download solo video
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    st.warning(" Scarica solo la parte video (senza audio):")
                    st.download_button(
                        label=" Scarica Video (senza audio)",
                        data=video_bytes,
                        file_name=f"synesthetic_flow_video_only.mp4",
                        mime="video/mp4"
                    )
                
        except Exception as e:
            st.error(f" Errore durante la generazione: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info(" Carica un file audio per iniziare")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "*<span style='font-size: 10px; color: #666;'>"
    "SynestheticFlow Enhanced v2.0 - Visualizer audio-reattivo con frattali avanzati"
    "</span>*", 
    unsafe_allow_html=True
)
