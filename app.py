import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from pydub import AudioSegment
import librosa
import soundfile as sf
import subprocess
import gc
import shutil
import io
from typing import Tuple, Optional

# Configurazione memoria ridotta
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

st.set_page_config(page_title="🎶 SynestheticFlow", layout="wide")

st.markdown("# 🎶 SynestheticFlow <span style='font-size:0.5em;'>by Loop507</span>", unsafe_allow_html=True)
st.write("Visualizzazioni musicali ottimizzate per 200MB RAM")

# --- FUNZIONI OTTIMIZZATE PER MEMORIA MINIMA ---

@st.cache_data
def get_audio_info(audio_bytes: bytes) -> Tuple[float, int]:
    """Estrae informazioni base dall'audio senza caricarlo tutto."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return len(audio) / 1000.0, audio.frame_rate
    except:
        return 0.0, 22050

def draw_minimal_mandala(frame_img: np.ndarray, width: int, height: int, 
                        rms: float, complexity: int, thickness: int, 
                        beat: bool, freq_data: Tuple[float, float, float]) -> np.ndarray:
    """Mandala ultra-semplificato."""
    cx, cy = width // 2, height // 2
    base_r = min(width, height) // 4
    circles = min(3 + int(rms * complexity), 8)  # Max 8 cerchi
    
    beat_scale = 1.3 if beat else 1.0
    bass, mid, treble = freq_data
    
    colors = [
        (int(255 * treble), int(128 * mid), int(255 * bass)),
        (int(128 * bass), int(255 * treble), int(128 * mid)),
        (int(255 * mid), int(255 * bass), int(128 * treble))
    ]
    
    for i in range(circles):
        r = int(base_r * (0.3 + 0.7 * i / circles) * beat_scale * (1 + rms))
        color = colors[i % 3]
        cv2.circle(frame_img, (cx, cy), r, color, thickness)
    
    return frame_img

def draw_minimal_waves(frame_img: np.ndarray, width: int, height: int,
                      rms: float, complexity: int, frame_idx: int,
                      beat: bool, freq_data: Tuple[float, float, float]) -> np.ndarray:
    """Pattern ondulato ultra-semplificato."""
    # Riduci risoluzione drasticamente per calcoli
    calc_w, calc_h = width // 4, height // 4
    
    # Griglia semplificata
    x = np.linspace(-1, 1, calc_w, dtype=np.float16)  # float16 per memoria
    y = np.linspace(-1, 1, calc_h, dtype=np.float16)
    X, Y = np.meshgrid(x, y)
    
    t = frame_idx * 0.1
    beat_mult = 1.5 if beat else 1.0
    bass, mid, treble = freq_data
    
    # Una sola onda semplice
    wave = np.sin(X * 5 * beat_mult + t + bass * 3) * np.cos(Y * 3 + mid * 2)
    wave = (wave + 1) * 127.5  # Normalizza a 0-255
    
    # Converti a RGB semplice
    hue = (wave + treble * 50) % 180
    rgb_small = np.zeros((calc_h, calc_w, 3), dtype=np.uint8)
    rgb_small[:,:,0] = np.clip(hue + bass * 100, 0, 255)
    rgb_small[:,:,1] = np.clip(wave + mid * 100, 0, 255) 
    rgb_small[:,:,2] = np.clip(255 - hue + treble * 100, 0, 255)
    
    # Ridimensiona velocemente
    rgb_full = cv2.resize(rgb_small, (width, height), interpolation=cv2.INTER_NEAREST)
    frame_img[:] = rgb_full
    
    return frame_img

def analyze_audio_minimal(audio_path: str, max_duration: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analisi audio super-ridotta."""
    try:
        # Carica solo quello che serve con qualità ridotta
        y, sr = librosa.load(audio_path, sr=11025, mono=True, duration=max_duration)  # SR ridotto
        
        # BPM veloce su campione piccolo
        y_beat = y[::8]  # Campiona ogni 8
        tempo, beats = librosa.beat.beat_track(y=y_beat, sr=sr//8, trim=False)
        beat_times = beats * 8 / sr  # Correggi timing
        
        return y, beat_times, tempo
    except Exception as e:
        st.warning(f"Analisi audio semplificata: {e}")
        return np.zeros(max_duration * 11025), np.array([]), 120.0

def process_frame_data(audio_chunk: np.ndarray, sr: int = 11025) -> Tuple[float, Tuple[float, float, float]]:
    """Analisi frame ultra-veloce."""
    if len(audio_chunk) == 0:
        return 0.0, (0.0, 0.0, 0.0)
    
    # RMS veloce
    rms = np.sqrt(np.mean(audio_chunk**2))
    rms_norm = min(1.0, rms * 10)  # Amplifica per visibilità
    
    # Frequenze super-semplificate (solo su 64 campioni max)
    if len(audio_chunk) > 32:
        chunk = audio_chunk[:64] if len(audio_chunk) > 64 else audio_chunk
        fft = np.fft.fft(chunk)
        mag = np.abs(fft[:len(chunk)//2])
        
        # Dividi in 3 bande fisse
        third = len(mag) // 3
        bass = np.mean(mag[:third]) if third > 0 else 0
        mid = np.mean(mag[third:2*third]) if third > 0 else 0
        treble = np.mean(mag[2*third:]) if third > 0 else 0
        
        # Normalizza veloce
        max_mag = max(bass, mid, treble, 1e-6)
        bass = min(1.0, bass / max_mag)
        mid = min(1.0, mid / max_mag) 
        treble = min(1.0, treble / max_mag)
    else:
        bass = mid = treble = 0.0
    
    return rms_norm, (bass, mid, treble)

def check_ffmpeg() -> bool:
    """Verifica FFmpeg."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# --- UI MINIMALISTA ---

uploaded_audio = st.file_uploader("🎵 Audio (MP3/WAV, max 30sec consigliati)", type=["mp3", "wav"])

with st.sidebar:
    st.header("⚙️ Impostazioni")
    
    # Preset rapidi per memoria
    preset = st.selectbox("📊 Preset qualità", [
        "🚀 Ultra-Fast (360p, 15fps)",
        "⚡ Fast (480p, 20fps)", 
        "🎯 Balanced (720p, 15fps)"
    ])
    
    if "Ultra-Fast" in preset:
        width, height, fps = 640, 360, 15
    elif "Fast" in preset:
        width, height, fps = 854, 480, 20
    else:
        width, height, fps = 1280, 720, 15
    
    max_duration = st.slider("⏱️ Durata max (sec)", 5, 30, 20)
    pattern = st.radio("🎨 Pattern", ["Mandala", "Onde"])
    complexity = st.slider("🔧 Complessità", 1, 5, 3)
    thickness = st.slider("📏 Spessore", 1, 3, 2) if pattern == "Mandala" else 1

# Mostra info memoria
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎬 Risoluzione", f"{width}x{height}")
with col2:
    st.metric("⚡ FPS", fps)
with col3:
    if uploaded_audio:
        duration, _ = get_audio_info(uploaded_audio.read())
        uploaded_audio.seek(0)  # Reset per uso successivo
        st.metric("⏱️ Durata", f"{min(duration, max_duration):.1f}s")
    else:
        st.metric("⏱️ Durata", "0s")

if st.button("🚀 **GENERA VIDEO**", type="primary"):
    if not uploaded_audio:
        st.error("⚠️ Carica un file audio!")
        st.stop()
    
    if not check_ffmpeg():
        st.error("❌ FFmpeg richiesto ma non disponibile!")
        st.info("💡 Su Streamlit Cloud: aggiungi 'ffmpeg' in packages.txt")
        st.stop()
    
    # Cleanup memoria iniziale
    gc.collect()
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    try:
        status_text.text("📁 Preparazione file...")
        
        # File temporanei
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_path = os.path.join(temp_dir, "video.mp4")
        final_path = os.path.join(temp_dir, "final.mp4")
        
        # Salva audio
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())
        
        status_text.text("🎵 Analisi audio veloce...")
        
        # Analisi audio ridotta
        audio_data, beat_times, tempo = analyze_audio_minimal(audio_path, max_duration)
        actual_duration = min(len(audio_data) / 11025, max_duration)
        total_frames = int(actual_duration * fps)
        samples_per_frame = len(audio_data) // total_frames
        
        st.info(f"🎼 BPM: {tempo:.0f} | ⏱️ {actual_duration:.1f}s | 🎬 {total_frames} frame")
        
        status_text.text("🎬 Creazione video...")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("❌ Impossibile creare video writer")
            st.stop()
        
        # Generazione frame ottimizzata
        for i in range(total_frames):
            # Estrai chunk audio
            start_idx = i * samples_per_frame
            end_idx = min(start_idx + samples_per_frame, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            # Analizza chunk
            rms, freq_data = process_frame_data(chunk)
            
            # Rileva beat
            frame_time = i / fps
            beat_detected = np.any(np.abs(beat_times - frame_time) < (0.5 / fps))
            
            # Crea frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if pattern == "Mandala":
                frame = draw_minimal_mandala(frame, width, height, rms, complexity, 
                                           thickness, beat_detected, freq_data)
            else:
                frame = draw_minimal_waves(frame, width, height, rms, complexity, 
                                         i, beat_detected, freq_data)
            
            out.write(frame)
            
            # Progresso e cleanup
            if i % 10 == 0:  # Ogni 10 frame
                progress_bar.progress((i + 1) / total_frames)
                status_text.text(f"🎨 Frame {i+1}/{total_frames} ({((i+1)/total_frames*100):.1f}%)")
                gc.collect()  # Forza pulizia memoria
        
        out.release()
        progress_bar.progress(1.0)
        status_text.text("🔧 Unione audio-video...")
        
        # Combina con FFmpeg
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_path,
            '-i', audio_path,
            '-t', str(actual_duration),
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '128k',
            final_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(final_path):
            status_text.text("✅ Video completato!")
            
            # Leggi e offri download
            with open(final_path, 'rb') as f:
                video_bytes = f.read()
            
            file_size_mb = len(video_bytes) / (1024 * 1024)
            st.success(f"🎉 **Video generato!** ({file_size_mb:.1f} MB)")
            
            # Download button
            st.download_button(
                label="⬇️ **SCARICA VIDEO**",
                data=video_bytes,
                file_name=f"synesthetic_{uploaded_audio.name.split('.')[0]}.mp4",
                mime="video/mp4"
            )
            
        else:
            st.error(f"❌ Errore FFmpeg: {result.stderr}")
            
    except Exception as e:
        st.error(f"❌ Errore: {str(e)}")
        if "memory" in str(e).lower():
            st.info("💡 Riduci durata o usa preset Ultra-Fast")
            
    finally:
        # Cleanup finale
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        gc.collect()

else:
    st.info("📝 **Consigli per 200MB RAM:**")
    st.write("""
    - ✅ **Max 20-30 secondi** di audio
    - ✅ **Preset Ultra-Fast** per test
    - ✅ **Chiudi altre app** durante generazione  
    - ✅ **File MP3 < 5MB** consigliati
    """)
    
    with st.expander("🔧 Ottimizzazioni implementate"):
        st.write("""
        **Memoria ridotta:**
        - Float16 per calcoli matematici
        - Risoluzione calcoli divisa per 4
        - Sample rate audio ridotto (11kHz)
        - Max 64 campioni per analisi FFT
        - Cleanup automatico ogni 10 frame
        
        **Performance:**
        - FFmpeg preset ultrafast
        - Codec mp4v per compatibilità
        - Timeout operazioni (60s max)
        - Cache librosa in /tmp
        """)
