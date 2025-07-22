import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip

# Costanti
MAX_DURATION = 300  # in secondi
MIN_DURATION = 1.0
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Funzione per creare una colormap psichedelica
def get_psychedelic_colormap():
    t = np.linspace(0, 1, 256)
    r = (np.sin(t * 10 + 4 * np.pi / 3) * 0.5 + 0.5)
    g = (np.sin(t * 10 + 2 * np.pi / 3) * 0.5 + 0.5)
    b = (np.sin(t * 10 + 0) * 0.5 + 0.5)
    return np.stack([r, g, b], axis=1)

# Funzione per applicare la colormap a un frame
def apply_psychedelic_colormap(data, colormap):
    norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_frame = np.zeros((norm_data.shape[0], norm_data.shape[1], 3), dtype=np.uint8)
    for i in range(256):
        mask = norm_data == i
        color_frame[mask] = (colormap[i] * 255).astype(np.uint8)
    return color_frame

# Caricamento e validazione audio
def load_audio(uploaded_file):
    if uploaded_file is None:
        return None, None, "Nessun file caricato."
    if uploaded_file.size > MAX_FILE_SIZE:
        return None, None, "Il file supera il limite di 50 MB."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    y, sr = librosa.load(tmp_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None, None, f"Durata non valida: {duration:.2f}s"
    return y, sr, tmp_path

# Generazione dei frame video
def generate_frames(y, sr, fps=20):
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    colormap = get_psychedelic_colormap()

    frames = []
    hop_length = 512
    frame_duration = sr / fps / hop_length
    num_frames = int(S_dB.shape[1] / frame_duration)

    for i in range(num_frames):
        start = int(i * frame_duration)
        end = start + 1
        if end >= S_dB.shape[1]:
            break
        frame_data = S_dB[:, start:end]
        frame_data = np.repeat(frame_data, 10, axis=1)
        frame_img = apply_psychedelic_colormap(frame_data, colormap)
        frame_img = cv2.resize(frame_img, (640, 480))
        frames.append(frame_img)
    return frames

# Creazione video con audio
def create_video_with_audio(frames, audio_path, fps):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        video_path = tmp_video.name
    clip = ImageSequenceClip(frames, fps=fps)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)
    clip.write_videofile(video_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return video_path

# Streamlit UI
st.title("🎧 Visualizzatore Audio Psichedelico")

uploaded_file = st.file_uploader("Carica un file audio (.wav)", type=["wav"])
fps = st.selectbox("Fotogrammi per secondo (FPS)", options=[5, 10, 20, 30], index=2)

if uploaded_file:
    with st.spinner("🎵 Caricamento audio..."):
        y, sr, tmp_audio_path_or_error = load_audio(uploaded_file)
        if y is None:
            st.error(tmp_audio_path_or_error)
        else:
            st.success(f"✔️ Audio caricato! Durata: {librosa.get_duration(y=y, sr=sr):.2f}s")
            st.audio(uploaded_file)

            if st.button("🎬 Crea Video"):
                with st.spinner("🎨 Generazione frames..."):
                    frames = generate_frames(y, sr, fps)

                with st.spinner("🎥 Creazione video finale..."):
                    video_path = create_video_with_audio(frames, tmp_audio_path_or_error, fps)
                    st.success("✅ Video generato!")
                    with open(video_path, "rb") as f:
                        st.download_button("⬇️ Scarica il Video", f, file_name="output.mp4", mime="video/mp4")
