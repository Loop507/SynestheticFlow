import streamlit as st
import numpy as np
import cv2
import librosa
import soundfile as sf
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
from moviepy.editor import AudioFileClip, VideoFileClip

MAX_DURATION = 300
MIN_DURATION = 1.0
MAX_FILE_SIZE = 50 * 1024 * 1024

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def check_file(file) -> Tuple[bool, Optional[str]]:
    if file is None:
        return False, "Nessun file caricato."
    if file.size > MAX_FILE_SIZE:
        return False, "Il file supera la dimensione massima di 50MB."
    try:
        y, sr = librosa.load(file, sr=None, mono=True, duration=MAX_DURATION)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < MIN_DURATION:
            return False, "La durata del file audio è troppo breve."
        return True, None
    except Exception as e:
        return False, f"Errore lettura audio: {str(e)}"

def generate_frames(y, sr, fps):
    st_frame = st.empty()
    frame_count = 0
    total_frames = int(len(y) / sr * fps)
    chunk = int(sr / fps)
    frames = []

    for i in range(0, len(y), chunk):
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        color = int(127 + 127 * np.sin(i / 10000.0))
        frame[:] = (color, color // 2, 255 - color)
        st_frame.image(frame, channels="BGR", caption=f"Frame {frame_count}/{total_frames}")
        frames.append(frame)
        frame_count += 1
    return frames

def save_video(frames, fps, output_path):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()

def merge_audio_video(audio_path, video_path, output_path):
    videoclip = VideoFileClip(video_path)
    audioclip = AudioFileClip(audio_path)
    final = videoclip.set_audio(audioclip)
    final.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)

def cleanup_temp_folder(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

st.set_page_config(page_title="🎨 Generatore Video Astratti", layout="wide")
st.title("🎧 Generatore di Video Astratti Audio-Reattivi")

if not check_ffmpeg():
    st.error("❌ FFMPEG non trovato. Assicurati che sia installato nel sistema.")
    st.stop()

fps_option = st.selectbox("🎞️ Seleziona FPS", [5, 10, 20, 30], index=2)
uploaded_file = st.file_uploader("📤 Carica un file audio (max 50MB)", type=["mp3", "wav", "ogg"])

if uploaded_file:
    is_valid, message = check_file(uploaded_file)
    if is_valid:
        y, sr = librosa.load(uploaded_file, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        st.success("✅ File audio valido.")
        st.write(f"Durata audio: {duration:.2f} secondi | FPS: {fps_option}")

        frames = generate_frames(y, sr, fps_option)

        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, "input_audio.wav")
        video_path = os.path.join(temp_dir, "temp_video.mp4")
        final_path = os.path.join(temp_dir, "final_output.mp4")

        sf.write(audio_path, y, sr)  # Salvataggio audio con soundfile
        save_video(frames, fps_option, video_path)
        merge_audio_video(audio_path, video_path, final_path)

        with open(final_path, "rb") as file:
            st.download_button("📥 Scarica Video", file, file_name="video_astratto.mp4")

        cleanup_temp_folder(temp_dir)
        gc.collect()
    else:
        st.warning(f"⚠️ {message}")
