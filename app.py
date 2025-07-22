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
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr) # [cite: 1, 2]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) # [cite: 2]
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
    freq_data_norm = freq_data / (np.max(freq_data) + 1e-6) # [cite: 3]

    total_bins = len(freq_data_norm) # [cite: 3]
    low_end = total_bins // 4  # Più preciso per le basse # [cite: 3]
    mid_end = (total_bins * 3) // 4  # Più spazio per le medie # [cite: 3]

    # Calcola con maggiore amplificazione per reattività
    low_freq = np.mean(freq_data_norm[:low_end]) ** 0.7 if low_end > 0 else 0  # Radice per amplificare # [cite: 3]
    mid_freq = np.mean(freq_data_norm[low_end:mid_end]) ** 0.6 if mid_end > low_end else 0 # [cite: 3]
    high_freq = np.mean(freq_data_norm[mid_end:]) ** 0.5 if total_bins > mid_end else 0 # [cite: 3, 4]

    # Amplifica ulteriormente per maggiore reattività
    return min(1.0, low_freq * 2.5), min(1.0, mid_freq * 2.2), min(1.0, high_freq * 2.8) # [cite: 4]

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0 # [cite: 4]

    # Applica una finestra per FFT per ridurre artefatti
    windowed_audio_chunk = audio_chunk * np.hanning(len(audio_chunk)) # [cite: 4]
    freq_data = np.abs(np.fft.rfft(windowed_audio_chunk)) if len(windowed_audio_chunk) > 0 else np.array([]) # [cite: 4]

    return rms, freq_data # [cite: 4]

def hex_to_bgr(hex_color):
    """Converte colore hex in formato BGR per OpenCV"""
    hex_color = hex_color.lstrip('#') # [cite: 4]
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) # [cite: 5]
    return (rgb[2], rgb[1], rgb[0])  # BGR format # [cite: 5]

def create_smooth_movement_parameters(frame_idx, tempo, movement_scale):
    """Crea parametri di movimento fluidi e costanti basati su onde sinusoidali multiple"""
    base_time = frame_idx * 0.02 * movement_scale  # Tempo base più fluido # [cite: 5]
    tempo_factor = max(0.5, tempo / 120.0)  # Normalizza BPM # [cite: 5]
    
    # Onde sinusoidali multiple per movimento complesso ma fluido
    wave1 = np.sin(base_time * tempo_factor) # [cite: 5]
    wave2 = np.sin(base_time * tempo_factor * 1.618) * 0.6  # Rapporto aureo # [cite: 6]
    wave3 = np.cos(base_time * tempo_factor * 0.786) * 0.4 # [cite: 6]
    
    # Parametri di movimento combinati
    x_movement = (wave1 + wave2 * 0.5) * 0.15 # [cite: 6]
    y_movement = (wave2 + wave3 * 0.7) * 0.12 # [cite: 6]
    rotation = (wave1 * wave3) * 0.1 # [cite: 6]
    scale_oscillation = 1.0 + (wave2 + wave3) * 0.2 # [cite: 6]
    
    return x_movement, y_movement, rotation, scale_oscillation # [cite: 6]

def create_dynamic_color_palette(low_freq, mid_freq, high_freq, base_colors, frame_idx, beat_intensity):
    """Crea una palette di colori dinamica basata sulle frequenze audio""" # [cite: 7]
    
    # Colori base da hex a BGR
    low_bgr = np.array(hex_to_bgr(base_colors['low_freq_color']), dtype=np.float32) # [cite: 7]
    mid_bgr = np.array(hex_to_bgr(base_colors['mid_freq_color']), dtype=np.float32) # [cite: 7]
    high_bgr = np.array(hex_to_bgr(base_colors['high_freq_color']), dtype=np.float32) # [cite: 7]
    
    # Intensità amplificate delle frequenze
    low_intensity = np.clip(low_freq * 3.5 + beat_intensity * 0.5, 0.0, 1.0) # [cite: 7]
    mid_intensity = np.clip(mid_freq * 3.2 + beat_intensity * 0.3, 0.0, 1.0) # [cite: 7]
    high_intensity = np.clip(high_freq * 4.0 + beat_intensity * 0.7, 0.0, 1.0) # [cite: 7, 8]
    
    # Modulazione temporale per colori che cambiano nel tempo
    time_mod = np.sin(frame_idx * 0.01) * 0.5 + 0.5 # [cite: 8]
    
    # Combina colori con modulazione temporale
    dynamic_low = low_bgr * (low_intensity * (0.7 + 0.3 * time_mod)) # [cite: 8]
    dynamic_mid = mid_bgr * (mid_intensity * (0.8 + 0.2 * np.sin(frame_idx * 0.015))) # [cite: 8]
    dynamic_high = high_bgr * (high_intensity * (0.6 + 0.4 * np.cos(frame_idx * 0.008))) # [cite: 8]
    
    # Colore finale mixato
    total_intensity = low_intensity + mid_intensity + high_intensity + 0.1 # [cite: 9]
    mixed_color = (dynamic_low + dynamic_mid + dynamic_high) / total_intensity # [cite: 9]
    
    # Saturazione dinamica basata sull'energia totale
    energy = (low_freq + mid_freq + high_freq) / 3.0 # [cite: 9]
    saturation_boost = 1.0 + energy * 1.5 + beat_intensity * 0.8 # [cite: 9]
    
    mixed_color = np.clip(mixed_color * saturation_boost, 0, 255).astype(np.uint8) # [cite: 9]
    
    return mixed_color, low_intensity, mid_intensity, high_intensity # [cite: 9]

def apply_advanced_color_modulation(fractal, color_data, frame_idx):
    """Applica modulazione colore avanzata al frattale""" # [cite: 10]
    mixed_color, low_int, mid_int, high_int = color_data # [cite: 10]
    
    # Crea maschere per diverse zone dell'immagine
    height, width = fractal.shape[:2] # [cite: 10]
    y_coords, x_coords = np.ogrid[:height, :width] # [cite: 10]
    
    # Zone concentriche per effetti colore diversificati
    center_x, center_y = width // 2, height // 2 # [cite: 10]
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2) # [cite: 10]
    max_distance = np.sqrt(center_x**2 + center_y**2) # [cite: 10]
    normalized_distance = distance_from_center / max_distance # [cite: 10]
    
    # Maschera per pixel non neri (parti attive del frattale) # [cite: 10, 11]
    active_mask = np.any(fractal != [0, 0, 0], axis=-1) # [cite: 11]
    
    # Modulazione colore per zone diverse
    for c in range(3):  # B, G, R # [cite: 11]
        # Zona centrale: colori più intensi
        central_zone = normalized_distance < 0.3 # [cite: 11]
        mid_zone = (normalized_distance >= 0.3) & (normalized_distance < 0.7) # [cite: 11]
        outer_zone = normalized_distance >= 0.7 # [cite: 11]
        
        # Applica modulazione per zona # [cite: 11, 12]
        central_mod = mixed_color[c] * (1.2 + low_int * 0.8) / 255.0 # [cite: 12]
        mid_mod = mixed_color[c] * (1.0 + mid_int * 0.6) / 255.0 # [cite: 12]
        outer_mod = mixed_color[c] * (0.9 + high_int * 1.0) / 255.0 # [cite: 12]
        
        # Combina le modulazioni
        color_mask = np.where(central_zone, central_mod, # [cite: 12, 13]
                             np.where(mid_zone, mid_mod, outer_mod)) # [cite: 13]
        
        # Applica ai pixel attivi del frattale
        fractal[active_mask, c] = np.clip( # [cite: 13]
            fractal[active_mask, c] * color_mask[active_mask] * 1.4, 0, 255 # [cite: 13]
        )
    
    # Aggiungi effetto "pulse" sui beat # [cite: 13, 14]
    pulse_effect = 0.02 * np.sin(frame_idx * 0.1) # [cite: 14]
    fractal[active_mask] = np.clip(fractal[active_mask] * (1.0 + pulse_effect), 0, 255) # [cite: 14]
    
    return fractal # [cite: 14]

# --- FUNZIONI FRATTALI AVANZATE CON MIGLIORAMENTI ---

@jit(nopython=True)
def mandelbrot_set_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence):
    """Mandelbrot con rotazione e movimento fluido migliorati""" # [cite: 14]
    fractal = np.zeros((height, width, 3), dtype=np.uint8) # [cite: 14]
    
    cos_rot = np.cos(rotation) # [cite: 14]
    sin_rot = np.sin(rotation) # [cite: 14]

    for y in range(height): # [cite: 14]
        for x in range(width): # [cite: 15]
            # Coordinate con rotazione
            centered_x = (x - width/2) / (zoom * width/4) # [cite: 15]
            centered_y = (y - height/2) / (zoom * height/4) # [cite: 15]
            
            # Applica rotazione
            rotated_x = centered_x * cos_rot - centered_y * sin_rot # [cite: 15, 16]
            rotated_y = centered_x * sin_rot + centered_y * cos_rot # [cite: 16]
            
            # Posizione finale con movimento
            c_real = rotated_x + move_x + audio_influence * 0.008 * np.sin(x * 0.002) # [cite: 16]
            c_imag = rotated_y + move_y + audio_influence * 0.008 * np.cos(y * 0.002) # [cite: 16, 17]

            z_real, z_imag = 0.0, 0.0 # [cite: 17]
            iteration = 0 # [cite: 17]

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0: # [cite: 17]
                z_real_new = z_real*z_real - z_imag*z_imag + c_real # [cite: 17]
                z_imag = 2.0*z_real*z_imag + c_imag # [cite: 17]
                z_real = z_real_new # [cite: 17, 18]
                iteration += 1 # [cite: 18]

            if iteration == max_iter: # [cite: 18]
                fractal[y, x] = [0, 0, 0] # [cite: 18]
            else:
                # Gradiente colorato più ricco # [cite: 18, 19]
                t = float(iteration) / max_iter # [cite: 19]
                smooth_t = t + 1 - np.log2(np.log2(z_real*z_real + z_imag*z_imag)) # [cite: 19]
                
                r = int(np.sin(smooth_t * 12 + 0) * 127 + 128) # [cite: 19]
                g = int(np.sin(smooth_t * 12 + 2) * 127 + 128) # [cite: 20]
                b = int(np.sin(smooth_t * 12 + 4) * 127 + 128) # [cite: 20]
                fractal[y, x] = [b, g, r] # [cite: 20]

    return fractal # [cite: 20]

@jit(nopython=True)
def julia_set_enhanced(width, height, max_iter, c_real_base, c_imag_base, zoom, rotation, audio_mod):
    """Julia Set con rotazione e modulazione audio migliorata""" # [cite: 20]
    fractal = np.zeros((height, width, 3), dtype=np.uint8) # [cite: 20]

    c_real = c_real_base + audio_mod * 0.08 * np.sin(audio_mod * 10) # [cite: 20, 21]
    c_imag = c_imag_base + audio_mod * 0.09 * np.cos(audio_mod * 12) # [cite: 21]
    
    cos_rot = np.cos(rotation) # [cite: 21]
    sin_rot = np.sin(rotation) # [cite: 21]

    for y in range(height): # [cite: 21]
        for x in range(width): # [cite: 21]
            centered_x = (x - width/2) / (zoom * width/4) # [cite: 21]
            centered_y = (y - height/2) / (zoom * width/4) # [cite: 21]
            
            # Rotazione # [cite: 22]
            z_real = centered_x * cos_rot - centered_y * sin_rot # [cite: 22]
            z_imag = centered_x * sin_rot + centered_y * cos_rot # [cite: 22]

            iteration = 0 # [cite: 22]
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0: # [cite: 22]
                z_real_new = z_real*z_real - z_imag*z_imag + c_real # [cite: 22, 23]
                z_imag = 2.0*z_real*z_imag + c_imag # [cite: 23]
                z_real = z_real_new # [cite: 23]
                iteration += 1 # [cite: 23]

            if iteration == max_iter: # [cite: 23, 24]
                fractal[y, x] = [25, 25, 50] # [cite: 24]
            else:
                t = float(iteration) / max_iter # [cite: 24]
                # Colori più dinamici e psichedelici
                b = int(255 * (np.sin(t * 15 + audio_mod * 5) * 0.5 + 0.5)) # [cite: 24]
                g = int(255 * (np.sin(t * 15 + audio_mod * 7 + 2.1) * 0.5 + 0.5)) # [cite: 25]
                r = int(255 * (np.sin(t * 15 + audio_mod * 6 + 4.2) * 0.5 + 0.5)) # [cite: 25]
                fractal[y, x] = [b, g, r] # [cite: 25]

    return fractal # [cite: 25]

@jit(nopython=True)
def burning_ship_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence):
    """Burning Ship con effetti di rotazione e audio migliorati""" # [cite: 25, 26]
    fractal = np.zeros((height, width, 3), dtype=np.uint8) # [cite: 26]
    
    cos_rot = np.cos(rotation) # [cite: 26]
    sin_rot = np.sin(rotation) # [cite: 26]

    for y in range(height): # [cite: 26]
        for x in range(width): # [cite: 26]
            centered_x = (x - width/2) / (zoom * width/4) # [cite: 26]
            centered_y = (y - height/2) / (zoom * height/4) # [cite: 26]
            
            rotated_x = centered_x * cos_rot - centered_y * sin_rot # [cite: 27]
            rotated_y = centered_x * sin_rot + centered_y * cos_rot # [cite: 27]
            
            c_real = rotated_x + move_x + audio_influence * 0.006 * np.sin(x * 0.003 + y * 0.002) # [cite: 27]
            c_imag = rotated_y + move_y + audio_influence * 0.006 * np.cos(x * 0.002 + y * 0.003) # [cite: 27, 28]

            z_real, z_imag = 0.0, 0.0 # [cite: 28]
            iteration = 0 # [cite: 28]

            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4.0: # [cite: 28]
                z_real_new = z_real*z_real - z_imag*z_imag + c_real # [cite: 28]
                z_imag = 2.0 * np.abs(z_real) * np.abs(z_imag) + c_imag # [cite: 28, 29]
                z_real = z_real_new # [cite: 29]
                iteration += 1 # [cite: 29]

            if iteration == max_iter: # [cite: 29]
                fractal[y, x] = [0, 0, 0] # [cite: 29]
            else:
                t = float(iteration) / max_iter # [cite: 30]
                # Colori "fiammeggianti" più realistici
                b = int(255 * np.clip(np.sin(t * 8 + 0) * 0.3 + 0.1, 0, 1)) # [cite: 30]
                g = int(255 * np.clip(np.sin(t * 8 + 1.5) * 0.6 + 0.3, 0, 1)) # [cite: 30]
                r = int(255 * np.clip(np.sin(t * 8 + 3) * 0.8 + 0.7, 0, 1)) # [cite: 31]
                fractal[y, x] = [b, g, r] # [cite: 31]

    return fractal # [cite: 31]

# --- FUNZIONI DI DISEGNO MIGLIORATE ---

def draw_mandelbrot_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Mandelbrot con movimento fluido e colori reattivi migliorati""" # [cite: 31]
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data) # [cite: 31]
    
    # Parametri di movimento fluidi
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor) # [cite: 31, 32]
    
    # Parametri audio-reattivi migliorati # [cite: 32]
    max_iter = max(60, min(250, int(100 + rms * 200 * movement_scale_factor))) # [cite: 32]
    zoom = (1.8 + rms * 4 * movement_scale_factor + low_freq * 8 * movement_scale_factor) * scale_osc # [cite: 32]
    move_x = x_move - 0.75 + mid_freq * 0.15 * movement_scale_factor # [cite: 32]
    move_y = y_move + 0.05 + high_freq * 0.12 * movement_scale_factor # [cite: 32]
    audio_influence = (rms * 2.5 + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor # [cite: 32]
    
    fractal = mandelbrot_set_enhanced(width, # [cite: 32]
        height, max_iter, zoom, move_x, move_y, rotation, audio_influence # [cite: 33]
    )

    if color_settings['use_frequency_colors']: # [cite: 33]
        beat_intensity = 1.5 if beat else 1.0 # [cite: 33]
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity) # [cite: 33]
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx) # [cite: 33]

    alpha = 0.85 if beat else 0.75 # [cite: 33]
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img) # [cite: 33]
    return frame_img # [cite: 33]

def draw_julia_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Julia Set con movimento e colori migliorati""" # [cite: 33]
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data) # [cite: 34]
    
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor) # [cite: 34]
    
    max_iter = max(60, min(180, int(90 + rms * 120 * movement_scale_factor))) # [cite: 34]
    c_real_base = -0.7 + x_move * 0.5 # [cite: 34]
    c_imag_base = 0.27015 + y_move * 0.4 # [cite: 34]
    zoom = (1.2 + rms * 2 * movement_scale_factor + high_freq * 3.0 * movement_scale_factor) * scale_osc # [cite: 34]
    audio_mod = (rms * 2.0 + (low_freq * 0.6 + mid_freq + high_freq * 0.4)) * movement_scale_factor # [cite: 34]

    fractal = julia_set_enhanced(width, height, max_iter, c_real_base, c_imag_base, zoom, rotation, audio_mod) # [cite: 35]

    if color_settings['use_frequency_colors']: # [cite: 35]
        beat_intensity = 1.8 if beat else 1.0 # [cite: 35]
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity) # [cite: 35]
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx) # [cite: 35]

    alpha = 0.9 if beat else 0.8 # [cite: 35]
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img) # [cite: 35]
    return frame_img # [cite: 35]

def draw_burning_ship_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Burning Ship con movimento fluido migliorato""" # [cite: 35, 36]
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data) # [cite: 36]
    
    x_move, y_move, rotation, scale_osc = create_smooth_movement_parameters(frame_idx, tempo, movement_scale_factor) # [cite: 36]
    
    max_iter = max(50, min(150, int(80 + rms * 90 * movement_scale_factor))) # [cite: 36]
    zoom = (1.2 + rms * 2.5 * movement_scale_factor + mid_freq * 3.5 * movement_scale_factor) * scale_osc # [cite: 36]
    move_x = -1.8 + x_move * 0.3 # [cite: 36]
    move_y = -0.08 + y_move * 0.2 # [cite: 36]
    audio_influence = (rms * 1.5 + high_freq * 0.8) * movement_scale_factor # [cite: 36]

    fractal = burning_ship_enhanced(width, height, max_iter, zoom, move_x, move_y, rotation, audio_influence) # [cite: 36, 37]

    if color_settings['use_frequency_colors']: # [cite: 37]
        beat_intensity = 1.6 if beat else 1.0 # [cite: 37]
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity) # [cite: 37]
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx) # [cite: 37]

    alpha = 0.88 if beat else 0.78 # [cite: 37]
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img) # [cite: 37]
    return frame_img # [cite: 37]

# Mantieni le funzioni originali per Sierpinski e merge video
@jit(nopython=True)
def _remove_squares_numba(arr, level, x, y, size, fill_color):
    """Funzione ricorsiva per Sierpinski Carpet (Numba ottimizzato)""" # [cite: 37, 38]
    if level == 0 or size < 3: # [cite: 38]
        return # [cite: 38]

    third = size // 3 # [cite: 38]

    # Rimuovi il quadrato centrale # [cite: 38]
    for i in range(third): # [cite: 38]
        for j in range(third): # [cite: 38]
            # Controlla i limiti per evitare IndexError
            if x + third + i < arr.shape[0] and y + third + j < arr.shape[1]: # [cite: 38, 39]
                arr[x + third + i, y + third + j] = fill_color # 

    # Ricorsione sui quadrati rimanenti # 
    for i in range(3): # 
        for j in range(3): # 
            if i != 1 or j != 1:  # Salta il quadrato centrale # 
                _remove_squares_numba(arr, level-1, x + i*third, y + j*third, third, fill_color) # [cite: 39, 40]

def generate_sierpinski_carpet(width, height, iterations, audio_scale, base_color_bgr):
    """Genera il tappeto di Sierpinski con influenza audio""" # [cite: 40]
    size = min(width, height) # [cite: 40]
    carpet = np.full((size, size, 3), base_color_bgr, dtype=np.uint8) # [cite: 40]

    iter_count = max(1, min(6, int(iterations + audio_scale * 2))) # [cite: 40]
    _remove_squares_numba(carpet, iter_count, 0, 0, size, (0,0,0)) # [cite: 40]
    
    fractal = cv2.resize(carpet, (width, height), interpolation=cv2.INTER_AREA) # [cite: 40]
    return fractal # [cite: 40]

def draw_sierpinski_fractal_enhanced(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor, tempo):
    """Sierpinski con colori migliorati""" # [cite: 40]
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data) # [cite: 41]

    iterations = 3 + int(rms * 4 * movement_scale_factor) + int(low_freq * 3 * movement_scale_factor) # [cite: 41]
    audio_scale = (rms + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor # [cite: 41]

    base_carpet_color_bgr = hex_to_bgr(color_settings['background_color']) # [cite: 41]
    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale, base_carpet_color_bgr) # [cite: 41]

    if color_settings['use_frequency_colors']: # [cite: 41]
        beat_intensity = 1.4 if beat else 1.0 # [cite: 41]
        color_data = create_dynamic_color_palette(low_freq, mid_freq, high_freq, color_settings, frame_idx, beat_intensity) # [cite: 41]
        fractal = apply_advanced_color_modulation(fractal, color_data, frame_idx) # [cite: 41]

    alpha = 0.8 if beat else 0.7 # [cite: 42]
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img) # [cite: 42]
    return frame_img # [cite: 42]

def merge_video_audio(video_path, audio_path, output_path):
    """Combina video e audio usando ffmpeg""" # [cite: 42]
    try: # [cite: 42]
        cmd = [ # [cite: 42]
            'ffmpeg', '-y', # [cite: 42]
            '-i', video_path, # [cite: 42]
            '-i', audio_path, # [cite: 42]
            '-c:v', 'copy', # [cite: 42]
            '-c:a', 'aac', # [cite: 43]
            '-shortest', # [cite: 43]
            output_path # [cite: 43]
        ]

        result = subprocess.run(cmd, capture_output=True, text=True) # [cite: 43]

        if result.returncode == 0: # [cite: 43]
            return True, "Merge completato con successo" # [cite: 44]
        else: # [cite: 44]
            return False, f"Errore ffmpeg: {result.stderr}" # [cite: 44]

    except FileNotFoundError: # [cite: 44]
        return False, "ffmpeg non trovato. Installa ffmpeg sul sistema." # [cite: 45]
    except Exception as e: # [cite: 45]
        return False, f"Errore durante il merge: {str(e)}" # [cite: 45]

# --- INTERFACCIA STREAMLIT MIGLIORATA ---

st.title(" **SynestheticFlow Enhanced**") # [cite: 45]
st.markdown("*<span style='font-size: 12px;'>Visualizer Potenziato by loop507</span>*", unsafe_allow_html=True) # [cite: 45]

# Selezione formato video
st.subheader(" Formato Video") # [cite: 45]
selected_format = st.selectbox( # [cite: 45]
    "Scegli il rapporto di aspetto:", # [cite: 45]
    list(VIDEO_FORMATS.keys()), # [cite: 45]
    index=0 # [cite: 45]
)

width, height = VIDEO_FORMATS[selected_format] # [cite: 45]
st.info(f" Formato selezionato: **{selected_format}** - Risoluzione: {width}x{height}px") # [cite: 45]

# --- CONTROLLI FRATTALI ---
st.subheader(" Tipo di Effetto") # [cite: 46]

fractal_type = st.selectbox( # [cite: 46]
    "Seleziona l'effetto da generare:", # [cite: 46]
    [
        " Mandelbrot Set - Movimento fluido e rotazione", # [cite: 46]
        " Julia Set - Dinamico con rotazione avanzata", # [cite: 46]
        " Burning Ship - Forme organiche rotanti", # [cite: 46]
        " Sierpinski Carpet - Geometrico colorato" # [cite: 46]
    ],
    index=0 # [cite: 46]
)

# --- CONTROLLI MOVIMENTO EFFETTI ---
st.subheader(" Intensità Movimento Effetti") # [cite: 46]
movement_intensity = st.selectbox( # [cite: 46]
    "Seleziona l'intensità del movimento degli effetti:", # [cite: 46]
    ["Soft", "Medium", "Hard", "Extreme"], # [cite: 46]
    index=1 # [cite: 46]
)

movement_scale_factors = { # [cite: 47]
    "Soft": 0.4, # [cite: 47]
    "Medium": 1.0, # [cite: 47]
    "Hard": 1.8, # [cite: 47]
    "Extreme": 2.5 # [cite: 47]
}
current_movement_scale_factor = movement_scale_factors[movement_intensity] # [cite: 47]

# --- CONTROLLI COLORI AVANZATI ---
st.subheader(" Controlli Colori Avanzati") # [cite: 47]

col1, col2 = st.columns(2) # [cite: 47]

with col1: # [cite: 47]
    use_frequency_colors = st.checkbox(" Colori Reattivi Audio", value=True) # [cite: 47]
    background_color = st.color_picker(" Colore sfondo", value="#000000") # [cite: 47]
    
    if use_frequency_colors: # [cite: 47]
        st.markdown("** Intensità Reazione Audio:**") # [cite: 47]
        audio_color_intensity = st.slider("Intensità colori", 0.5, 3.0, 1.5, 0.1) # [cite: 47]

with col2: # [cite: 48]
    if use_frequency_colors: # [cite: 48]
        st.markdown("**Palette Frequenze Audio:**") # [cite: 48]
        low_freq_color = st.color_picker(" Basse (Sub-Bass/Bass)", value="#FF1744") # [cite: 48]
        mid_freq_color = st.color_picker(" Medie (Vocal/Lead)", value="#00E676") # [cite: 48]
        high_freq_color = st.color_picker(" Acute (Hi-Hat/Treble)", value="#2196F3") # [cite: 48]
    else: # [cite: 48]
        low_freq_color = "#FF1744" # [cite: 48]
        mid_freq_color = "#00E676" # [cite: 48]
        high_freq_color = "#2196F3" # [cite: 48]
        st.info("Abilita 'Colori Reattivi Audio' per personalizzare") # [cite: 48]

color_settings = { # [cite: 49]
    'use_frequency_colors': use_frequency_colors, # [cite: 49]
    'background_color': background_color, # [cite: 49]
    'low_freq_color': low_freq_color, # [cite: 49]
    'mid_freq_color': mid_freq_color, # [cite: 49]
    'high_freq_color': high_freq_color, # [cite: 49]
    'audio_color_intensity': audio_color_intensity if use_frequency_colors else 1.0 # [cite: 49]
}

# --- UPLOAD E CONTROLLI ---
st.subheader(" Upload Audio") # [cite: 49]
uploaded_file = st.file_uploader( # [cite: 49]
    "Carica il file audio:", # [cite: 49]
    type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], # [cite: 49]
    help="Formati supportati: WAV, MP3, OGG, FLAC, M4A" # [cite: 49]
)

if uploaded_file: # [cite: 49]
    st.success(f" File caricato: **{uploaded_file.name}**") # [cite: 50]
    
    # Controlli video
    st.subheader(" Controlli Video") # [cite: 50]
    col1, col2 = st.columns(2) # [cite: 50]
    
    with col1: # [cite: 50]
        video_duration = st.slider(" Durata video (sec)", 10, 300, 60, 5) # [cite: 50]
    
    with col2: # [cite: 50]
        video_quality = st.selectbox( # [cite: 50]
            " Qualità video:", # [cite: 51]
            ["Alta (20 FPS)", "Media (15 FPS)", "Veloce (12 FPS)"], # [cite: 51]
            index=0 # [cite: 51]
        )
    
    fps_settings = { # [cite: 51]
        "Alta (20 FPS)": 20, # [cite: 51]
        "Media (15 FPS)": 15, # [cite: 51]
        "Veloce (12 FPS)": 12 # [cite: 51]
    }
    fps = fps_settings[video_quality] # [cite: 51]
    
    # Stima dimensioni
    estimated_frames = video_duration * fps # [cite: 51]
    estimated_size_mb = (width * height * estimated_frames * 3) / (1024 * 1024) # [cite: 51]
    
    st.info(f" **Stima:** {estimated_frames} frame, ~{estimated_size_mb:.1f} MB") # [cite: 51]
    
    # Pulsante generazione
    if st.button(" **Genera Video Visualizer**", type="primary"): # [cite: 52]
        
        # Setup progress # [cite: 52]
        progress_bar = st.progress(0) # [cite: 52]
        status_text = st.empty() # [cite: 52]
        
        try: # [cite: 52]
            with tempfile.TemporaryDirectory() as temp_dir: # [cite: 53]
                status_text.text(" Analisi audio in corso...") # [cite: 53]
                
                # Prepara file audio # [cite: 53]
                audio_path = prepare_audio_file(uploaded_file, temp_dir) # [cite: 53]
                y, beat_times, tempo, sr = analyze_audio_minimal(audio_path) # [cite: 53]
                
                # Modifica qui: Assicurati che 'tempo' sia un float per la formattazione
                st.success(f"🎼 Audio analizzato - BPM: {float(tempo):.1f}") # Modificato
            
                # Calcola parametri video # [cite: 54]
                total_frames = video_duration * fps # [cite: 54]
                hop_length = len(y) // total_frames # [cite: 54]
                
                # Setup video writer # [cite: 55]
                video_path = f"{temp_dir}/fractal_video.mp4" # [cite: 55]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # [cite: 55]
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) # [cite: 55]
                
                if not video_writer.isOpened(): # [cite: 55, 56]
                    st.error(" Errore nell'inizializzazione del video writer") # [cite: 56]
                    st.stop() # [cite: 56]
                
                status_text.text(" Generazione frame in corso...") # [cite: 56]
                
                # Genera frame # [cite: 57]
                for frame_idx in range(total_frames): # [cite: 57]
                    # Calcola posizione audio # [cite: 57]
                    start_sample = frame_idx * hop_length # [cite: 57]
                    end_sample = min(start_sample + hop_length, len(y)) # [cite: 58]
                    audio_chunk = y[start_sample:end_sample] # [cite: 58]
                    
                    # Analisi audio frame # [cite: 58]
                    rms, freq_data = process_frame_data(audio_chunk) # [cite: 59]
                    
                    # Rileva beat # [cite: 59]
                    current_time = frame_idx / fps # [cite: 59]
                    beat = any(abs(current_time - bt) < 0.1 for bt in beat_times) # [cite: 59, 60]
                    
                    # Crea frame base # [cite: 60]
                    frame_img = np.full((height, width, 3), hex_to_bgr(background_color), dtype=np.uint8) # [cite: 60]
                    
                    # Disegna frattale selezionato # [cite: 60, 61]
                    if "Mandelbrot" in fractal_type: # [cite: 61]
                        frame_img = draw_mandelbrot_fractal_enhanced( # [cite: 61]
                            frame_img, width, height, rms, frame_idx, beat, # [cite: 61]
                            freq_data, color_settings, current_movement_scale_factor, tempo # [cite: 62]
                        )
                    elif "Julia" in fractal_type: # [cite: 62]
                        frame_img = draw_julia_fractal_enhanced( # [cite: 62, 63]
                            frame_img, width, height, rms, frame_idx, beat, # [cite: 63]
                            freq_data, color_settings, current_movement_scale_factor, tempo # [cite: 63]
                        )
                    elif "Burning Ship" in fractal_type: # [cite: 63, 64]
                        frame_img = draw_burning_ship_fractal_enhanced( # [cite: 64]
                            frame_img, width, height, rms, frame_idx, beat, # [cite: 64]
                            freq_data, color_settings, current_movement_scale_factor, tempo # [cite: 64]
                        )
                    elif "Sierpinski" in fractal_type: # [cite: 65]
                        frame_img = draw_sierpinski_fractal_enhanced( # [cite: 65]
                            frame_img, width, height, rms, frame_idx, beat, # [cite: 65]
                            freq_data, color_settings, current_movement_scale_factor, tempo # [cite: 66]
                        )
                    
                    # Scrivi frame # [cite: 66, 67]
                    video_writer.write(frame_img) # [cite: 67]
                    
                    # Aggiorna progress # [cite: 67]
                    progress = (frame_idx + 1) / total_frames # [cite: 67, 68]
                    progress_bar.progress(progress) # [cite: 68]
                    
                    if frame_idx % 10 == 0: # [cite: 68]
                        status_text.text(f" Frame {frame_idx + 1}/{total_frames} ({progress*100:.1f}%)") # [cite: 68]
                
                video_writer.release() # [cite: 69]
                
                # Merge audio e video # [cite: 69]
                status_text.text(" Merge audio e video...") # [cite: 69]
                output_path = f"{temp_dir}/final_video.mp4" # [cite: 69]
                
                success, message = merge_video_audio(video_path, audio_path, output_path) # [cite: 70]
                
                if success: # [cite: 70]
                    status_text.text(" Video completato!") # [cite: 70]
                    progress_bar.progress(1.0) # [cite: 70, 71]
                    
                    # Download # [cite: 71]
                    with open(output_path, 'rb') as video_file: # [cite: 71]
                        video_bytes = video_file.read() # [cite: 72]
                    
                    st.success(" **Video generato con successo!**") # [cite: 72]
                    st.download_button( # [cite: 72]
                        label=" **Scarica Video**", # [cite: 73]
                        data=video_bytes, # [cite: 73]
                        file_name=f"synesthetic_flow_{fractal_type.split()[1].lower()}_{movement_intensity.lower()}.mp4", # [cite: 73]
                        mime="video/mp4" # [cite: 73]
                    )
                    
                    # Mostra video preview # [cite: 74]
                    st.video(video_bytes) # [cite: 74]
                    
                else: # [cite: 74]
                    st.error(f" Errore nel merge: {message}") # [cite: 74]
                    
                    # Offri download solo video # [cite: 75]
                    with open(video_path, 'rb') as video_file: # [cite: 75]
                        video_bytes = video_file.read() # [cite: 76]
                    
                    st.warning(" Scarica solo la parte video (senza audio):") # [cite: 76]
                    st.download_button( # [cite: 76]
                        label=" Scarica Video (senza audio)", # [cite: 77]
                        data=video_bytes, # [cite: 77]
                        file_name=f"synesthetic_flow_video_only.mp4", # [cite: 77]
                        mime="video/mp4" # [cite: 77]
                    )
                
        except Exception as e: # [cite: 77]
            st.error(f" Errore durante la generazione: {str(e)}") # [cite: 78]
            import traceback # [cite: 78]
            st.code(traceback.format_exc()) # [cite: 78]

else: # [cite: 78]
    st.info(" Carica un file audio per iniziare") # [cite: 78]

# --- FOOTER ---
st.markdown("---") # [cite: 79]
st.markdown( # [cite: 79]
    "*<span style='font-size: 10px; color: #666;'>" # [cite: 79]
    "SynestheticFlow Enhanced v2.0 - Visualizer audio-reattivo con frattali avanzati" # [cite: 79]
    "</span>*", 
    unsafe_allow_html=True # [cite: 79]
)
