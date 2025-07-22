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

            # Aggiunta influenza audio - modulazione sottile
            c_real += audio_influence * 0.005 * np.sin(x * 0.001)
            c_imag += audio_influence * 0.005 * np.cos(y * 0.001)

            z_real, z_imag = 0.0, 0.0
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
            c_real = (x - width/2) / (zoom * width/4) + move_x
            c_imag = (y - height/2) / (zoom * height/4) + move_y

            # Influenza audio
            c_real += audio_influence * 0.003 * np.sin(x * 0.002 + y * 0.001)
            c_imag += audio_influence * 0.003 * np.cos(x * 0.001 + y * 0.002)

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

def generate_sierpinski_carpet(width, height, iterations, audio_scale, base_color_bgr, rotation_angle=0):
    """Genera il tappeto di Sierpinski con influenza audio e rotazione"""
    size = min(width, height)
    carpet = np.full((size, size, 3), base_color_bgr, dtype=np.uint8)

    iter_count = max(1, min(6, int(iterations + audio_scale * 2)))

    _remove_squares_numba(carpet, iter_count, 0, 0, size, (0,0,0))

    # Applica rotazione prima del resize
    if rotation_angle != 0:
        center = (size / 2, size / 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        carpet = cv2.warpAffine(carpet, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=base_color_bgr)


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
    low_intensity = np.clip(low_freq * 7.0, 0.0, 1.0) # Moltiplicatore calibrato, più sensibile
    mid_intensity = np.clip(mid_freq * 7.0, 0.0, 1.0)
    high_intensity = np.clip(high_freq * 7.0, 0.0, 1.0)

    mixed_frequency_color = (
        low_bgr * low_intensity +
        mid_bgr * mid_intensity +
        high_bgr * high_intensity
    )
    mixed_frequency_color = np.clip(mixed_frequency_color / (low_intensity + mid_intensity + high_intensity + 1e-6), 0, 255)

    mask = np.any(colored_fractal != [0,0,0], axis=-1)

    for c in range(3):
        colored_fractal[mask, c] = np.clip(colored_fractal[mask, c] * (mixed_frequency_color[c] / 255.0) * 1.8, 0, 255) # Moltiplica e satura

    return colored_fractal

# --- FUNZIONI DI DISEGNO FRATTALE PER IL PROCESSING ---

def draw_mandelbrot_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor):
    """Disegna frattale di Mandelbrot reattivo all'audio con movimenti migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)

    # Movimenti costanti e modulati dall'audio e da movement_scale_factor
    time_factor = frame_idx * 0.001 * movement_scale_factor # Velocità base del movimento
    
    # "Pulse" effect sul beat
    beat_pulse_zoom = 0.5 if beat else 0 # Zoom aggiuntivo sul beat
    beat_pulse_iter = 50 if beat else 0 # Iterazioni aggiuntive sul beat

    max_iter = max(50, min(250, int(80 + rms * 180 * movement_scale_factor + beat_pulse_iter)))
    
    # Movimenti più complessi e fluidi
    zoom = (1.5 + rms * 6 * movement_scale_factor + low_freq * 12 * movement_scale_factor + 
            np.sin(time_factor * 2.5) * 0.3 * movement_scale_factor + beat_pulse_zoom) 
    
    move_x = (-0.75 + np.sin(time_factor * 1.5) * 0.25 * movement_scale_factor + # Movimento sinusoidale costante
              mid_freq * 0.15 * movement_scale_factor) # Modulazione audio
    
    move_y = (0.05 + np.cos(time_factor * 1.8) * 0.2 * movement_scale_factor + # Movimento sinusoidale costante
              high_freq * 0.12 * movement_scale_factor) # Modulazione audio
    
    audio_influence = (rms * 2.5 + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor
    
    fractal = mandelbrot_set_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)

    # Color flash sul beat
    if beat and color_settings['use_frequency_colors']:
        flash_color_bgr = np.array(hex_to_bgr("#FFFFFF")) # Bianco o colore vibrante
        # Aumenta l'intensità del flash con l'RMS
        flash_intensity = np.clip(rms * 5.0, 0.2, 0.8) # Tra 20% e 80% di blending
        cv2.addWeighted(fractal, 1 - flash_intensity, flash_color_bgr, flash_intensity, 0, fractal)


    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
        
    alpha = 0.8 if beat else 0.65 # Blending più forte sul beat
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_julia_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor):
    """Disegna frattale di Julia reattivo all'audio con movimenti migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)

    time_factor = frame_idx * 0.0012 * movement_scale_factor # Velocità base del movimento

    # "Pulse" effect sul beat
    beat_pulse_mod = 0.1 if beat else 0 # Modifica aggiuntiva sui parametri C
    beat_pulse_zoom = 0.8 if beat else 0 # Zoom aggiuntivo sul beat

    max_iter = max(50, min(180, int(70 + rms * 100 * movement_scale_factor)))

    # Variazione dei parametri C per Julia per effetti psichedelici e movimenti fluidi
    c_real_base = (-0.7 + np.sin(time_factor * 2.0) * 0.25 * movement_scale_factor +
                   rms * 0.08 * movement_scale_factor + beat_pulse_mod)
    c_imag_base = (0.27015 + np.cos(time_factor * 2.2) * 0.2 * movement_scale_factor +
                   rms * 0.06 * movement_scale_factor + beat_pulse_mod)

    zoom = (1.0 + rms * 2.0 * movement_scale_factor + high_freq * 2.5 * movement_scale_factor +
            np.sin(time_factor * 1.0) * 0.5 * movement_scale_factor + beat_pulse_zoom) # Zoom più dinamico

    audio_mod = (rms * 1.8 + (low_freq * 0.6 + mid_freq * 1.0 + high_freq * 0.4)) * movement_scale_factor # Modulazione combinata

    fractal = julia_set_numba(width, height, max_iter, c_real_base, c_imag_base, zoom, audio_mod)

    # Color flash sul beat
    if beat and color_settings['use_frequency_colors']:
        flash_color_bgr = np.array(hex_to_bgr("#FFC0CB")) # Rosa chiaro o colore vivace
        flash_intensity = np.clip(rms * 4.0, 0.2, 0.7)
        cv2.addWeighted(fractal, 1 - flash_intensity, flash_color_bgr, flash_intensity, 0, fractal)


    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
        
    alpha = 0.9 if beat else 0.75 # Blending più intenso
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_burning_ship_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor):
    """Disegna frattale Burning Ship con movimenti migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)

    time_factor = frame_idx * 0.0008 * movement_scale_factor # Velocità base del movimento

    # "Pulse" effect sul beat
    beat_pulse_zoom = 0.4 if beat else 0
    beat_pulse_iter = 30 if beat else 0

    max_iter = max(40, min(150, int(60 + rms * 80 * movement_scale_factor + beat_pulse_iter)))
    
    zoom = (1.0 + rms * 1.8 * movement_scale_factor + mid_freq * 2.5 * movement_scale_factor +
            np.cos(time_factor * 1.5) * 0.4 * movement_scale_factor + beat_pulse_zoom)
    
    move_x = (-1.8 + np.sin(time_factor * 1.0) * 0.15 * movement_scale_factor + # Movimento costante
              rms * 0.05 * movement_scale_factor)
    
    move_y = (-0.08 + np.cos(time_factor * 1.2) * 0.1 * movement_scale_factor + # Movimento costante
              low_freq * 0.03 * movement_scale_factor)
    
    audio_influence = (rms * 1.5 + high_freq * 0.8) * movement_scale_factor

    fractal = burning_ship_numba(width, height, max_iter, zoom, move_x, move_y, audio_influence)

    # Color flash sul beat
    if beat and color_settings['use_frequency_colors']:
        flash_color_bgr = np.array(hex_to_bgr("#FFD700")) # Oro o colore acceso
        flash_intensity = np.clip(rms * 4.5, 0.2, 0.8)
        cv2.addWeighted(fractal, 1 - flash_intensity, flash_color_bgr, flash_intensity, 0, fractal)

    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
        
    alpha = 0.85 if beat else 0.7 # Leggermente più opaco
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    
    return frame_img

def draw_sierpinski_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, movement_scale_factor):
    """Disegna tappeto di Sierpinski con movimenti migliorati"""
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)

    time_factor = frame_idx * 0.0005 * movement_scale_factor # Velocità base del movimento

    # "Pulse" effect sul beat
    beat_pulse_iter = 1 if beat else 0 # Iterazioni aggiuntive sul beat
    beat_pulse_rot = 5 if beat else 0 # Rotazione aggiuntiva sul beat

    iterations = 3 + int(rms * 4 * movement_scale_factor) + int(low_freq * 3 * movement_scale_factor) + beat_pulse_iter
    audio_scale = (rms + (low_freq + mid_freq + high_freq) / 3.0) * movement_scale_factor

    # Rotazione costante e modulata
    rotation_angle = (np.sin(time_factor * 0.8) * 10 * movement_scale_factor + # Rotazione lenta
                      high_freq * 20 * movement_scale_factor + beat_pulse_rot) # Modulata da audio e beat

    base_carpet_color_bgr = hex_to_bgr(color_settings['background_color'])

    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale, base_carpet_color_bgr, rotation_angle=rotation_angle)

    # Color flash sul beat
    if beat and color_settings['use_frequency_colors']:
        flash_color_bgr = np.array(hex_to_bgr("#9400D3")) # Viola profondo o colore contrastante
        flash_intensity = np.clip(rms * 3.5, 0.2, 0.7)
        cv2.addWeighted(fractal, 1 - flash_intensity, flash_color_bgr, flash_intensity, 0, fractal)

    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
        
    alpha = 0.75 if beat else 0.6 # Blending per Sierpinski
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
st.subheader("🌀 Tipo di Effetto")

fractal_type = st.selectbox(
    "Seleziona l'effetto da generare:",
    [
        "🌀 Mandelbrot Set - Classico e ipnotico",
        "🔥 Julia Set - Dinamico e fluido",
        "🚢 Burning Ship - Forme organiche",
        "📐 Sierpinski Carpet - Geometrico"
    ],
    index=0
)

# --- CONTROLLI MOVIMENTO EFFETTI (REINTRODOTTO) ---
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

                st.info("🌌 Generazione frattali procedurali...")

                for frame_idx in range(frame_count):
                    start_time = frame_idx * frame_duration

                    start_sample = int(start_time * sr)
                    end_sample = start_sample + int(frame_duration * sr)
                    audio_chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                    rms, freq_data = process_frame_data(audio_chunk)
                    beat = np.any((beat_times >= start_time) & (beat_times < start_time + frame_duration))

                    frame_img = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)

                    # Seleziona il tipo di frattale (senza mix casuale)
                    if "Mandelbrot" in fractal_type:
                        frame_img = draw_mandelbrot_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, current_movement_scale_factor)
                    elif "Julia" in fractal_type:
                        frame_img = draw_julia_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, current_movement_scale_factor)
                    elif "Burning Ship" in fractal_type:
                        frame_img = draw_burning_ship_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, current_movement_scale_factor)
                    elif "Sierpinski" in fractal_type:
                        frame_img = draw_sierpinski_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings, current_movement_scale_factor)

                    video_writer.write(frame_img)
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    progress = (frame_idx + 1) / frame_count
                    progress_bar.progress(progress)

                    time.sleep(frame_duration * 0.05)  # Anteprima più veloce

                video_writer.release()
                st.success("✅ Frattali generati con successo!")

                st.info("🎵 Sincronizzazione audio...")
                success, message = merge_video_audio(video_temp, audio_path, video_final)

                if success:
                    st.success("🎉 Video frattale con audio completato!")
                    with open(video_final, "rb") as f:
                        video_bytes = f.read()

                    format_name = selected_format.split(" ")[0].replace(":", "x")
                    fractal_name = fractal_type.split(" ")[1].lower()
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
    **SynestheticFlow** genera frattali matematici complessi sincronizzati con l'audio:

    **🌀 Tipi di Frattali Disponibili:**

    - **Mandelbrot Set**: Il frattale più famoso, genera infinite spirali e forme organiche.
    - **Julia Set**: Forme fluide e dinamiche che cambiano costantemente.
    - **Burning Ship**: Crea strutture che ricordano navi e paesaggi bruciati.
    - **Sierpinski Carpet**: Pattern geometrici auto-simili.

    **🎵 Reattività Audio:**
    - **RMS (Volume)**: Controlla zoom, intensità e velocità di morphing.
    - **Frequenze Basse**: Influenzano movimento orizzontale e parametri base.
    - **Frequenze Medie**: Controllano movimento verticale e dettagli.
    - **Frequenze Acute**: Modulano zoom e distorsioni.
    - **Beat Detection**: Intensifica colori e blending durante i colpi ritmici.

    **⚡ Ottimizzazioni:**
    - Algoritmi compilati con Numba per performance superiori.
    - Calcoli paralleli per rendering in tempo quasi reale.
    - Gestione memoria efficiente per video lunghi.

    **🎨 Sistema Colori Avanzato:**
    - Modulazione colore dinamica basata sull'intensità delle bande di frequenza.
    - Blending intelligente tra sfondo e frattale.
    - Colori predefiniti vivaci per un impatto visivo immediato.

    **Requisiti**: `ffmpeg` installato sul sistema per la fusione audio/video.
    """)
