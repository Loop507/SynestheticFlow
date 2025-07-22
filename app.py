import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time
import os
import subprocess

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

def get_optimal_settings(duration):
    width, height = 640, 360
    fps = 20
    estimated_size = (width * height * fps * duration) / (1024 * 1024)  # rough estimate
    return width, height, fps, int(estimated_size)

def process_frame_data(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0
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

# --- INTERFACCIA STREAMLIT ---

st.title("🎨 **SynestheticFlow**")
st.markdown("*<span style='font-size: 12px;'>by loop507</span>*", unsafe_allow_html=True)

uploaded_audio = st.file_uploader("Carica file audio", type=["wav", "mp3", "ogg"])

if uploaded_audio:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = prepare_audio_file(uploaded_audio, temp_dir)
            y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
            duration = librosa.get_duration(filename=audio_path)
            width, height, fps, est_size = get_optimal_settings(duration)
            
            st.info(f"🎼 BPM: {float(tempo):.0f} | ⏱️ {float(duration):.1f}s | 🎮 {fps} FPS | 📆 ~{est_size} MB")

            if st.button("🎬 CREA VIDEO"):
                # Crea il video senza audio
                video_temp = os.path.join(temp_dir, "video_temp.mp4")
                video_final = os.path.join(temp_dir, "video_with_audio.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_temp, fourcc, fps, (width, height))
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                frame_count = int(duration * fps)
                frame_duration = 1.0 / fps
                
                st.info("🎬 Generazione frames video...")
                
                for frame_idx in range(frame_count):
                    start_time = frame_idx * frame_duration
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + int(frame_duration * sr)
                    audio_chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                    rms, freq_data = process_frame_data(audio_chunk)
                    beat = np.any((beat_times >= start_time) & (beat_times < start_time + frame_duration))
                    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    if frame_idx % 2 == 0:
                        frame_img = draw_minimal_mandala(frame_img, width, height, rms, 5, 2, beat, freq_data)
                    else:
                        frame_img = draw_minimal_waves(frame_img, width, height, rms, 5, frame_idx, beat, freq_data)
                    
                    video_writer.write(frame_img)
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
                    
                    # Aggiorna progress bar
                    progress = (frame_idx + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    # Velocizza la preview (meno sleep)
                    time.sleep(frame_duration * 0.1)  # 10% della durata reale per anteprima veloce
                
                video_writer.release()
                st.success("✅ Frames video generati!")
                
                # Combina video e audio
                st.info("🎵 Sincronizzazione audio...")
                success, message = merge_video_audio(video_temp, audio_path, video_final)
                
                if success:
                    st.success("🎉 Video con audio generato con successo!")
                    with open(video_final, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        "⬇️ Scarica Video Completo", 
                        video_bytes, 
                        file_name="synesthetic_flow_video.mp4", 
                        mime="video/mp4"
                    )
                else:
                    st.warning(f"⚠️ {message}")
                    st.info("📹 Scarica il video senza audio:")
                    with open(video_temp, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        "⬇️ Scarica Video (solo visivo)", 
                        video_bytes, 
                        file_name="synesthetic_flow_video_no_audio.mp4", 
                        mime="video/mp4"
                    )

        except Exception as e:
            st.error(f"❌ Errore generazione video: {e}")

# Informazioni tecniche nell'expander
with st.expander("ℹ️ Informazioni Tecniche"):
    st.markdown("""
    **SynestheticFlow** genera visualizzazioni sincronizzate con l'audio:
    
    - 🎵 **Analisi Audio**: Rileva BPM e beat usando librosa
    - 🎨 **Visualizzazioni**: Mandala e onde che reagiscono al suono
    - 🎬 **Sincronizzazione**: 20 FPS per fluidità ottimale
    - 🔊 **Audio**: Incorporato nel video finale via ffmpeg
    
    **Requisiti**: ffmpeg installato sul sistema per l'audio
    """)
