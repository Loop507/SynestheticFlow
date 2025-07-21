import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time

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
    return y, beat_times, tempo

def get_optimal_settings(duration):
    width, height = 640, 360
    fps = 20
    estimated_size = (width * height * fps * duration) / (1024 * 1024)  # rough estimate
    return width, height, fps, int(estimated_size)

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    freq_data = np.abs(np.fft.rfft(audio_chunk)) if len(audio_chunk) > 0 else np.array([])
    return rms, freq_data

def draw_minimal_mandala(frame_img, width, height, rms, param1, param2, beat, freq_data):
    color = (int(min(255, rms * 500)), 50, 150)
    center = (width // 2, height // 2)
    radius = int(min(height // 3, rms * 150))
    cv2.circle(frame_img, center, radius, color, thickness=3 if beat else 1)
    return frame_img

def draw_minimal_waves(frame_img, width, height, rms, param1, frame_idx, beat, freq_data):
    amplitude = int(min(height // 4, rms * 200))
    color = (150, 50, int(min(255, rms * 500)))
    for x in range(0, width, 20):
        y = int(height / 2 + amplitude * np.sin(0.1 * x + frame_idx * 0.2))
        cv2.circle(frame_img, (x, y), 5, color, -1)
    return frame_img

# --- INTERFACCIA STREAMLIT ---

st.title("Generatore Video Audio Sincronizzato")

uploaded_audio = st.file_uploader("Carica file audio", type=["wav", "mp3", "ogg"])

if uploaded_audio:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = prepare_audio_file(uploaded_audio, temp_dir)
            y, beat_times, tempo = analyze_audio_minimal(audio_path)
            duration = librosa.get_duration(filename=audio_path)
            width, height, fps, est_size = get_optimal_settings(duration)
            
            st.info(f"🎼 BPM: {float(tempo):.0f} | ⏱️ {float(duration):.1f}s | 🎮 {fps} FPS | 📆 ~{est_size} MB")

            if st.button("🎬 CREA VIDEO"):
                video_placeholder = st.empty()
                frame_count = int(duration * fps)
                frame_duration = 1.0 / fps
                
                for frame_idx in range(frame_count):
                    start_time = frame_idx * frame_duration
                    start_sample = int(start_time * 11025)
                    end_sample = start_sample + int(frame_duration * 11025)
                    audio_chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                    rms, freq_data = process_frame_data(audio_chunk)
                    beat = np.any((beat_times >= start_time) & (beat_times < start_time + frame_duration))
                    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    if frame_idx % 2 == 0:
                        frame_img = draw_minimal_mandala(frame_img, width, height, rms, 5, 2, beat, freq_data)
                    else:
                        frame_img = draw_minimal_waves(frame_img, width, height, rms, 5, frame_idx, beat, freq_data)
                    
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
                    time.sleep(frame_duration)
        except Exception as e:
            st.error(f"Errore generazione video: {e}")
