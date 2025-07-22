import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import moviepy.editor as mp
import shutil
from pathlib import Path
import soundfile as sf

st.set_page_config(page_title="Audio Reactive Fractals", layout="wide")

MAX_DURATION = 300  # durata massima in secondi
MAX_FILE_SIZE_MB = 200

st.title("🎧 Audio Reactive Fractal Generator")

uploaded_audio = st.file_uploader("Carica un file audio", type=["mp3", "wav", "ogg"])
fps_choice = st.selectbox("FPS del video finale", [5, 10, 20, 30], index=2)

def analyze_audio(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return duration, tempo

def create_temp_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())
    return temp_path, temp_dir

def render_preview(duration, bpm):
    st.success(f"✅ Audio caricato correttamente!")
    st.info(f"🎼 Durata: {duration:.2f} sec | BPM stimati: {bpm:.0f}")
    st.video("https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.mp4")

if uploaded_audio is not None:
    file_size_mb = uploaded_audio.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File troppo grande ({file_size_mb:.2f} MB). Max: {MAX_FILE_SIZE_MB} MB")
    else:
        temp_audio_path, temp_dir = create_temp_file(uploaded_audio)
        try:
            duration, bpm = analyze_audio(temp_audio_path)
            if duration > MAX_DURATION:
                st.warning(f"⚠️ Il file supera la durata massima di {MAX_DURATION} secondi.")
            else:
                render_preview(duration, bpm)

                if st.button("🎬 Crea Video"):
                    progress_text = "🎥 Generazione video in corso..."
                    progress_bar = st.progress(0, text=progress_text)

                    # --- Generazione clip video base (placeholder fractal loop) ---
                    output_video_path = os.path.join(temp_dir, "final_video.mp4")
                    img_clip = mp.ColorClip(size=(720, 720), color=(0, 0, 0), duration=duration)
                    img_clip = img_clip.set_fps(fps_choice)

                    # --- Audio ---
                    final_audio = mp.AudioFileClip(temp_audio_path)
                    final_clip = img_clip.set_audio(final_audio)

                    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
                    progress_bar.progress(100, text="✅ Video generato!")

                    st.video(output_video_path)
                    with open(output_video_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Scarica Video",
                            data=f,
                            file_name="frattale_reattivo.mp4",
                            mime="video/mp4"
                        )
        finally:
            shutil.rmtree(temp_dir)
