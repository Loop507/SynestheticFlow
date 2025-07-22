import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time
import os
import subprocess
from numba import jit

# --- CONFIGURAZIONI FORMATO ---
VIDEO_FORMATS = {
    "16:9 (Landscape) - 1280x720": (1280, 720),
    "1:1 (Square) - 720x720": (720, 720), 
    "9:16 (Portrait) - 720x1280": (720, 1280)
}

# --- FUNZIONI FRATTALI AVANZATE ---

@jit(nopython=True)
def mandelbrot_set(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            c_real = (x - width/2) / (zoom * width/4) + move_x
            c_imag = (y - height/2) / (zoom * height/4) + move_y
            c_real += audio_influence * 0.1 * np.sin(x * 0.01)
            c_imag += audio_influence * 0.1 * np.cos(y * 0.01)
            z_real, z_imag = 0, 0
            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*z_real*z_imag + c_imag
                z_real = z_real_new
                iteration += 1
            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]
            else:
                color_val = int(255 * iteration / max_iter)
                fractal[y, x] = [color_val, color_val//2, 255 - color_val]
    return fractal

@jit(nopython=True) 
def julia_set(width, height, max_iter, c_real, c_imag, zoom, audio_mod):
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    c_real_mod = c_real + audio_mod * 0.3
    c_imag_mod = c_imag + audio_mod * 0.2
    for y in range(height):
        for x in range(width):
            z_real = (x - width/2) / (zoom * width/4)
            z_imag = (y - height/2) / (zoom * height/4)
            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real_mod
                z_imag = 2*z_real*z_imag + c_imag_mod
                z_real = z_real_new
                iteration += 1
            if iteration == max_iter:
                fractal[y, x] = [20, 20, 40]
            else:
                t = iteration / max_iter
                fractal[y, x] = [
                    int(255 * abs(np.sin(t * 6.28))),
                    int(255 * abs(np.sin(t * 6.28 + 2.09))), 
                    int(255 * abs(np.sin(t * 6.28 + 4.18)))
                ]
    return fractal

def generate_burning_ship(width, height, max_iter, zoom, move_x, move_y, audio_influence):
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            c_real = (x - width/2) / (zoom * width/4) + move_x
            c_imag = (y - height/2) / (zoom * height/4) + move_y
            c_real += audio_influence * 0.05 * np.sin(x * 0.02 + y * 0.01)
            c_imag += audio_influence * 0.05 * np.cos(x * 0.01 + y * 0.02)
            z_real, z_imag = 0, 0
            iteration = 0
            while iteration < max_iter and z_real*z_real + z_imag*z_imag < 4:
                z_real_new = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*abs(z_real)*abs(z_imag) + c_imag
                z_real = z_real_new
                iteration += 1
            if iteration == max_iter:
                fractal[y, x] = [0, 0, 0]
            else:
                t = iteration / max_iter
                fractal[y, x] = [
                    min(255, int(255 * t * 2)),
                    min(255, int(255 * t * t * 3)),
                    min(255, int(255 * np.sqrt(t) * 1.5))
                ]
    return fractal

def generate_sierpinski_carpet(width, height, iterations, audio_scale):
    size = min(width, height)
    carpet = np.ones((size, size), dtype=np.uint8) * 255
    iter_count = max(1, min(8, int(iterations + audio_scale * 3)))
    def remove_squares(arr, level, x, y, size):
        if level == 0 or size < 3:
            return
        third = size // 3
        for i in range(third):
            for j in range(third):
                if x + third + i < arr.shape[0] and y + third + j < arr.shape[1]:
                    arr[x + third + i, y + third + j] = 0
        for i in range(3):
            for j in range(3):
                if i != 1 or j != 1:
                    remove_squares(arr, level-1, x + i*third, y + j*third, third)
    remove_squares(carpet, iter_count, 0, 0, size)
    fractal = np.zeros((height, width, 3), dtype=np.uint8)
    carpet_resized = cv2.resize(carpet, (width, height))
    for c in range(3):
        fractal[:, :, c] = carpet_resized
    return fractal

def apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings):
    if not color_settings['use_frequency_colors']:
        return fractal
    height, width = fractal.shape[:2]
    colored_fractal = fractal.copy()
    low_bgr = hex_to_bgr(color_settings['low_freq_color'])
    mid_bgr = hex_to_bgr(color_settings['mid_freq_color']) 
    high_bgr = hex_to_bgr(color_settings['high_freq_color'])
    for y in range(height):
        for x in range(width):
            if np.sum(fractal[y, x]) > 0:
                zone = (x // (width // 3)) + (y // (height // 3)) * 3
                if zone % 3 == 0:
                    intensity = min(1.0, low_freq * 300)
                    colored_fractal[y, x] = [
                        min(255, int(fractal[y, x][0] * intensity + low_bgr[0] * (1-intensity))),
                        min(255, int(fractal[y, x][1] * intensity + low_bgr[1] * (1-intensity))),
                        min(255, int(fractal[y, x][2] * intensity + low_bgr[2] * (1-intensity)))
                    ]
                elif zone % 3 == 1:
                    intensity = min(1.0, mid_freq * 300)
                    colored_fractal[y, x] = [
                        min(255, int(fractal[y, x][0] * intensity + mid_bgr[0] * (1-intensity))),
                        min(255, int(fractal[y, x][1] * intensity + mid_bgr[1] * (1-intensity))),
                        min(255, int(fractal[y, x][2] * intensity + mid_bgr[2] * (1-intensity)))
                    ]
                else:
                    intensity = min(1.0, high_freq * 300)
                    colored_fractal[y, x] = [
                        min(255, int(fractal[y, x][0] * intensity + high_bgr[0] * (1-intensity))),
                        min(255, int(fractal[y, x][1] * intensity + high_bgr[1] * (1-intensity))),
                        min(255, int(fractal[y, x][2] * intensity + high_bgr[2] * (1-intensity)))
                    ]
    return colored_fractal

# --- FUNZIONI ORIGINALI MANTENUTE ---
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
    estimated_size = (width * height * fps * duration) / (1024 * 1024)
    return width, height, fps, int(estimated_size)

def analyze_frequency_bands(freq_data):
    if len(freq_data) == 0:
        return 0, 0, 0
    total_bins = len(freq_data)
    low_end = total_bins // 3
    mid_end = (total_bins * 2) // 3
    low_freq = np.mean(freq_data[:low_end]) if low_end > 0 else 0
    mid_freq = np.mean(freq_data[low_end:mid_end]) if mid_end > low_end else 0
    high_freq = np.mean(freq_data[mid_end:]) if total_bins > mid_end else 0
    return low_freq, mid_freq, high_freq

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0
    freq_data = np.abs(np.fft.rfft(audio_chunk)) if len(audio_chunk) > 0 else np.array([])
    return rms, freq_data

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

# --- NUOVE FUNZIONI FRATTALI PER IL PROCESSING ---

def draw_mandelbrot_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    max_iter = max(20, min(100, int(50 + rms * 50)))
    zoom = 100 + rms * 200 + low_freq * 300
    move_x = np.sin(frame_idx * 0.01) * 0.5 + mid_freq * 0.3
    move_y = np.cos(frame_idx * 0.008) * 0.5 + high_freq * 0.2
    audio_influence = rms * 2 + (low_freq + mid_freq + high_freq) / 3
    fractal = mandelbrot_set(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    alpha = 0.8 if beat else 0.6
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def draw_julia_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    max_iter = max(30, min(80, int(40 + rms * 40)))
    c_real = -0.7 + np.sin(frame_idx * 0.02) * 0.3 + low_freq * 0.5
    c_imag = 0.27015 + np.cos(frame_idx * 0.015) * 0.2 + mid_freq * 0.4
    zoom = 150 + rms * 100 + high_freq * 200
    audio_mod = rms + (low_freq + mid_freq + high_freq) / 3
    fractal = julia_set(width, height, max_iter, c_real, c_imag, zoom, audio_mod)
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    alpha = 0.9 if beat else 0.7
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def draw_burning_ship_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    max_iter = max(25, min(70, int(35 + rms * 35)))
    zoom = 200 + rms * 150
    move_x = -1.8 + np.sin(frame_idx * 0.005) * 0.2
    move_y = -0.08 + np.cos(frame_idx * 0.007) * 0.1
    audio_influence = rms * 1.5
    fractal = generate_burning_ship(width, height, max_iter, zoom, move_x, move_y, audio_influence)
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    alpha = 0.85 if beat else 0.65
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def draw_sierpinski_fractal(frame_img, width, height, rms, frame_idx, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    iterations = 4 + int(rms * 2)
    audio_scale = rms + (low_freq + mid_freq + high_freq) / 3
    fractal = generate_sierpinski_carpet(width, height, iterations, audio_scale)
    if color_settings['use_frequency_colors']:
        fractal = apply_frequency_colors_to_fractal(fractal, low_freq, mid_freq, high_freq, color_settings)
    alpha = 0.7 if beat else 0.5
    cv2.addWeighted(frame_img, 1-alpha, fractal, alpha, 0, frame_img)
    return frame_img

def merge_video_audio(video_path, audio_path, output_path):
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

# --- NUOVA FUNZIONE PER GLI EFFETTI VISIVI ---

def apply_visual_effects(frame_img, effects_settings):
    img = frame_img.copy()
    if effects_settings.get("blur", False):
        img = cv2.GaussianBlur(img, (7,7), 0)
    if effects_settings.get("increase_contrast", False):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if effects_settings.get("vignette", False):
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        vignette = np.copy(img)
        for i in range(3):
            vignette[:,:,i] = vignette[:,:,i] * mask
        img = vignette.astype(np.uint8)
    return img

# --- INTERFACCIA STREAMLIT ---

st.title("🎵 Fractal Audio Visualizer")

uploaded_file = st.file_uploader("Carica un file audio (.wav, .mp3)", type=["wav","mp3"])

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    audio_path = prepare_audio_file(uploaded_file, temp_dir)
    y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(f"Durata audio: {duration:.2f} secondi, BPM stimato: {tempo:.2f}")
    
    video_format = st.selectbox("Seleziona formato video", list(VIDEO_FORMATS.keys()))
    width, height = VIDEO_FORMATS[video_format]
    
    # Impostazioni colori frequenze
    st.subheader("🎨 Colori Frequenze")
    use_freq_colors = st.checkbox("Usa colori diversi per bande frequenza", value=True)
    low_freq_color = st.color_picker("Colore Basse Frequenze", "#FF0000")
    mid_freq_color = st.color_picker("Colore Medie Frequenze", "#00FF00")
    high_freq_color = st.color_picker("Colore Alte Frequenze", "#0000FF")
    color_settings = {
        'use_frequency_colors': use_freq_colors,
        'low_freq_color': low_freq_color,
        'mid_freq_color': mid_freq_color,
        'high_freq_color': high_freq_color
    }

    # Scelta tipo fractal
    st.subheader("Tipo di Frattale")
    fractal_type = st.selectbox("Scegli tipo", ["Mandelbrot", "Julia", "Burning Ship", "Sierpinski Carpet"])
    
    # Effetti visivi opzionali
    st.subheader("✨ Effetti Visivi Aggiuntivi")
    col1, col2, col3 = st.columns(3)
    with col1:
        effect_blur = st.checkbox("Blur (sfocatura)", value=False)
    with col2:
        effect_contrast = st.checkbox("Contrasto aumentato", value=False)
    with col3:
        effect_vignette = st.checkbox("Vignettatura", value=False)
    effects_settings = {
        "blur": effect_blur,
        "increase_contrast": effect_contrast,
        "vignette": effect_vignette
    }
    
    # Bottone start
    if st.button("Genera video"):
        fps = 20
        frame_count = int(duration * fps)
        frame_length = int(len(y) / frame_count)
        
        video_path = f"{temp_dir}/output_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        stframe = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(frame_count):
            audio_chunk = y[i*frame_length : (i+1)*frame_length]
            rms, freq_data = process_frame_data(audio_chunk)
            beat = any((beat_times >= i/fps) & (beat_times < (i+1)/fps))
            
            frame_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            if fractal_type == "Mandelbrot":
                frame_img = draw_mandelbrot_fractal(frame_img, width, height, rms, i, beat, freq_data, color_settings)
            elif fractal_type == "Julia":
                frame_img = draw_julia_fractal(frame_img, width, height, rms, i, beat, freq_data, color_settings)
            elif fractal_type == "Burning Ship":
                frame_img = draw_burning_ship_fractal(frame_img, width, height, rms, i, beat, freq_data, color_settings)
            elif fractal_type == "Sierpinski Carpet":
                frame_img = draw_sierpinski_fractal(frame_img, width, height, rms, i, beat, freq_data, color_settings)
            
            # Applica effetti
            frame_img = apply_visual_effects(frame_img, effects_settings)
            
            video_writer.write(frame_img)
            
            if i % 10 == 0:
                progress_bar.progress(min(100, int(i/frame_count*100)))
                stframe.image(frame_img, channels="BGR", caption=f"Frame {i+1}/{frame_count}")
        
        video_writer.release()
        
        # Merge audio e video
        output_final = f"{temp_dir}/final_output.mp4"
        success, msg = merge_video_audio(video_path, audio_path, output_final)
        if success:
            st.success("Video generato con successo!")
            video_file = open(output_final, 'rb').read()
            st.video(video_file)
        else:
            st.error(f"Errore durante il merge audio-video: {msg}")
