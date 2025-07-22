import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import os

# Configurazioni formato video
VIDEO_FORMATS = {
    "16:9 (Landscape) - 1280x720": (1280, 720),
    "1:1 (Square) - 720x720": (720, 720),
    "9:16 (Portrait) - 720x1280": (720, 1280)
}

def prepare_audio_file(uploaded_file, temp_dir):
    audio_path = os.path.join(temp_dir, "input_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    return audio_path

def analyze_audio_minimal(audio_path):
    y, sr = librosa.load(audio_path, sr=11025)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return y, beat_times, tempo, sr

def analyze_frequency_bands(freq_data):
    if len(freq_data) == 0:
        return 0, 0, 0
    freq_data_norm = freq_data / (np.max(freq_data) + 1e-6)
    total_bins = len(freq_data_norm)
    low_end = total_bins // 3
    mid_end = (total_bins * 2) // 3
    low_freq = np.mean(freq_data_norm[:low_end]) if low_end > 0 else 0
    mid_freq = np.mean(freq_data_norm[low_end:mid_end]) if mid_end > low_end else 0
    high_freq = np.mean(freq_data_norm[mid_end:]) if total_bins > mid_end else 0
    return low_freq, mid_freq, high_freq

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0
    windowed_audio_chunk = audio_chunk * np.hanning(len(audio_chunk))
    fft_data = np.abs(np.fft.fft(windowed_audio_chunk))
    return (*analyze_frequency_bands(fft_data), rms)

def generate_video(audio_path, width, height, effect_type, sensitivity):
    y, _, _, sr = analyze_audio_minimal(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = 'output_video.mp4'
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(int(duration * fps)):
        t = i / fps
        audio_chunk = y[int(t * sr):int((t + 1/fps) * sr)]
        low_freq, mid_freq, high_freq, rms = process_frame_data(audio_chunk)
        frame = create_effect_frame(width, height, effect_type, rms, low_freq, mid_freq, high_freq, sensitivity)
        video.write(frame)

    video.release()
    return video_path

def create_effect_frame(width, height, effect_type, rms, low_freq, mid_freq, high_freq, sensitivity):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if effect_type == "Mandelbrot":
        frame[:, :] = [int(low_freq * 255), int(mid_freq * 255), int(high_freq * 255)]
    elif effect_type == "Julia":
        frame[:, :] = [int(high_freq * 255), int(low_freq * 255), int(mid_freq * 255)]
    elif effect_type == "Burning Ship":
        frame[:, :] = [int(mid_freq * 255), int(high_freq * 255), int(low_freq * 255)]
    elif effect_type == "Sierpinski Carpet":
        frame[:, :] = [int((low_freq + mid_freq) * 127), int((mid_freq + high_freq) * 127), int((high_freq + low_freq) * 127)]
    return frame

# Interfaccia Utente Streamlit
st.title("Generatore di Video con Effetti")
uploaded_file = st.file_uploader("Carica un file audio", type=["wav", "mp3"])
effect_type = st.selectbox("Seleziona il tipo di effetto", ["Mandelbrot", "Julia", "Burning Ship", "Sierpinski Carpet"])
video_format = st.selectbox("Seleziona il formato video", list(VIDEO_FORMATS.keys()))
width, height = VIDEO_FORMATS[video_format]
sensitivity = st.slider('Sensibilità Effetti Audio', 0.1, 2.0, 1.0)

if st.button("Genera Video"):
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = prepare_audio_file(uploaded_file, temp_dir)
            video_path = generate_video(audio_path, width, height, effect_type, sensitivity)
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
    else:
        st.error("Carica un file audio prima di generare il video.")
