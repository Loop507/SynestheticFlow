import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import librosa
import soundfile as sf
import subprocess
import gc
import shutil
import io
import psutil
from typing import Tuple, Optional
import threading
import time

# Configurazione memoria ridotta
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['OMP_NUM_THREADS'] = '1'  # Limita threads per memoria

st.set_page_config(page_title="🎶 SynestheticFlow", layout="wide")

st.markdown("# 🎶 SynestheticFlow <span style='font-size:0.5em;'>by Loop507</span>", unsafe_allow_html=True)
st.write("Visualizzazioni musicali ottimizzate - Max 4 minuti o 200MB")

# --- FALLBACK PER PYDUB ---
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("⚠️ PyDub non disponibile - usando fallback librosa")

# --- FUNZIONI DI CALCOLO DIMENSIONI ---

def estimate_video_size(width: int, height: int, fps: int, duration: float, bitrate_factor: float = 1.0) -> float:
    """Stima la dimensione del video in MB."""
    # Stima basata su bitrate tipici H.264
    pixels_per_second = width * height * fps
    # Bitrate base: ~0.1 bits per pixel per secondo (H.264 efficiente)
    estimated_bitrate_bps = pixels_per_second * 0.1 * bitrate_factor
    # Aggiungi audio (96 kbps)
    total_bitrate_bps = estimated_bitrate_bps + 96000
    # Calcola dimensione in MB
    size_mb = (total_bitrate_bps * duration) / (8 * 1024 * 1024)
    return size_mb

def get_optimal_settings(duration: float, max_size_mb: int = 200) -> Tuple[int, int, int, int]:
    """Calcola impostazioni ottimali per rispettare il limite di dimensione."""
    # Preset 16:9 standard ordinati per qualità crescente
    presets = [
        (426, 240, 8, 0.4),    # 240p Ultra Low
        (640, 360, 10, 0.6),   # 360p Low  
        (854, 480, 12, 0.8),   # 480p Medium
        (1280, 720, 15, 1.0),  # 720p HD
        (1920, 1080, 20, 1.3)  # 1080p Full HD
    ]
    
    for width, height, fps, bitrate_factor in presets:
        estimated_size = estimate_video_size(width, height, fps, duration, bitrate_factor)
        if estimated_size <= max_size_mb:
            return width, height, fps, int(estimated_size)
    
    # Se nessun preset funziona, usa il più basso
    return presets[0][0], presets[0][1], presets[0][2], int(estimate_video_size(*presets[0][:3], duration, presets[0][3]))

def check_memory_available() -> float:
    """Controlla memoria RAM disponibile in MB."""
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except:
        return 500.0  # Valore di default se psutil non funziona

# --- FUNZIONI OTTIMIZZATE (invariate) ---

@st.cache_data
def get_audio_info(audio_bytes: bytes) -> Tuple[float, int]:
    """Estrae informazioni base dall'audio senza caricarlo tutto."""
    try:
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return len(audio) / 1000.0, audio.frame_rate
        else:
            # Fallback con librosa (più lento ma funziona)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                try:
                    y, sr = librosa.load(tmp.name, sr=None, duration=1)  # Solo 1 sec per info
                    duration = librosa.get_duration(filename=tmp.name)
                    return duration, sr
                finally:
                    os.unlink(tmp.name)
    except Exception as e:
        st.warning(f"Errore lettura audio: {e}")
        return 0.0, 22050

def draw_minimal_mandala(frame_img: np.ndarray, width: int, height: int, 
                        rms: float, complexity: int, thickness: int, 
                        beat: bool, freq_data: Tuple[float, float, float]) -> np.ndarray:
    """Mandala ultra-semplificato."""
    try:
        cx, cy = width // 2, height // 2
        base_r = min(width, height) // 4
        circles = min(3 + int(float(rms) * complexity), 8)  # Max 8 cerchi
        
        beat_scale = 1.3 if beat else 1.0
        bass, mid, treble = map(float, freq_data)  # Converti esplicitamente a float
        
        colors = [
            (int(np.clip(255 * treble, 0, 255)), 
             int(np.clip(128 * mid, 0, 255)), 
             int(np.clip(255 * bass, 0, 255))),
            (int(np.clip(128 * bass, 0, 255)), 
             int(np.clip(255 * treble, 0, 255)), 
             int(np.clip(128 * mid, 0, 255))),
            (int(np.clip(255 * mid, 0, 255)), 
             int(np.clip(255 * bass, 0, 255)), 
             int(np.clip(128 * treble, 0, 255)))
        ]
        
        for i in range(circles):
            if circles > 0:  # Evita divisione per zero
                r = int(base_r * (0.3 + 0.7 * i / circles) * beat_scale * (1 + float(rms)))
                r = max(1, r)  # Assicurati che il raggio sia almeno 1
                color = colors[i % 3]
                cv2.circle(frame_img, (cx, cy), r, color, thickness)
        
    except Exception as e:
        # Fallback: cerchio semplice
        cx, cy = width // 2, height // 2
        r = min(width, height) // 6
        cv2.circle(frame_img, (cx, cy), r, (128, 128, 128), thickness)
    
    return frame_img

def draw_minimal_waves(frame_img: np.ndarray, width: int, height: int,
                      rms: float, complexity: int, frame_idx: int,
                      beat: bool, freq_data: Tuple[float, float, float]) -> np.ndarray:
    """Pattern ondulato ultra-semplificato."""
    try:
        # Riduci risoluzione drasticamente per calcoli
        calc_w, calc_h = width // 4, height // 4
        
        # Griglia semplificata
        x = np.linspace(-1, 1, calc_w, dtype=np.float32)
        y = np.linspace(-1, 1, calc_h, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        t = float(frame_idx * 0.1)
        beat_mult = 1.5 if beat else 1.0
        bass, mid, treble = map(float, freq_data)
        
        # Una sola onda semplice
        wave = np.sin(X * 5.0 * beat_mult + t + bass * 3.0) * np.cos(Y * 3.0 + mid * 2.0)
        wave = np.clip((wave + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        # Converti a RGB semplice
        hue = np.clip((wave.astype(np.float32) + treble * 50.0) % 180.0, 0, 255)
        
        rgb_small = np.zeros((calc_h, calc_w, 3), dtype=np.uint8)
        rgb_small[:,:,0] = np.clip(hue + bass * 100.0, 0, 255).astype(np.uint8)
        rgb_small[:,:,1] = np.clip(wave.astype(np.float32) + mid * 100.0, 0, 255).astype(np.uint8)
        rgb_small[:,:,2] = np.clip(255.0 - hue + treble * 100.0, 0, 255).astype(np.uint8)
        
        # Ridimensiona velocemente
        rgb_full = cv2.resize(rgb_small, (width, height), interpolation=cv2.INTER_NEAREST)
        frame_img[:] = rgb_full
        
    except Exception as e:
        # Fallback: riempi con colore solido
        bass, mid, treble = map(float, freq_data)
        color = [
            int(np.clip(bass * 255, 0, 255)),
            int(np.clip(mid * 255, 0, 255)), 
            int(np.clip(treble * 255, 0, 255))
        ]
        frame_img[:] = color
    
    return frame_img

def analyze_audio_minimal(audio_path: str, max_duration: int = 240) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analisi audio super-ridotta - fino a 4 minuti."""
    try:
        # Carica solo quello che serve con qualità ridotta
        y, sr = librosa.load(audio_path, sr=11025, mono=True, duration=max_duration)
        
        # Pulisci memoria
        gc.collect()
        
        # BPM veloce su campione piccolo
        if len(y) > 1000:
            y_beat = y[::8]  # Campiona ogni 8
            tempo, beats = librosa.beat.beat_track(y=y_beat, sr=sr//8, trim=False)
            beat_times = beats * 8 / sr  # Correggi timing
        else:
            tempo, beat_times = 120.0, np.array([])
        
        return y, beat_times, tempo
    except Exception as e:
        st.warning(f"Analisi audio semplificata: {e}")
        return np.zeros(max_duration * 11025), np.array([]), 120.0

def process_frame_data(audio_chunk: np.ndarray, sr: int = 11025) -> Tuple[float, Tuple[float, float, float]]:
    """Analisi frame ultra-veloce."""
    try:
        if len(audio_chunk) == 0:
            return 0.0, (0.0, 0.0, 0.0)
        
        # RMS veloce - converti esplicitamente a float
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)))
        rms_norm = min(1.0, rms * 10.0)  # Amplifica per visibilità
        
        # Frequenze super-semplificate (solo su 64 campioni max)
        if len(audio_chunk) > 32:
            chunk = audio_chunk[:64] if len(audio_chunk) > 64 else audio_chunk
            chunk = chunk.astype(np.float32)
            
            fft = np.fft.rfft(chunk)  # Più efficiente per segnali reali
            mag = np.abs(fft).astype(np.float32)
            
            # Dividi in 3 bande fisse
            third = len(mag) // 3
            if third > 0:
                bass = float(np.mean(mag[:third]))
                mid = float(np.mean(mag[third:2*third]))
                treble = float(np.mean(mag[2*third:]))
                
                # Normalizza veloce
                max_mag = max(bass, mid, treble, 1e-6)
                bass = min(1.0, bass / max_mag)
                mid = min(1.0, mid / max_mag) 
                treble = min(1.0, treble / max_mag)
            else:
                bass = mid = treble = 0.0
        else:
            bass = mid = treble = 0.0
        
        return rms_norm, (bass, mid, treble)
        
    except Exception as e:
        return 0.0, (0.0, 0.0, 0.0)

def check_ffmpeg() -> bool:
    """Verifica FFmpeg."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def prepare_audio_file(uploaded_audio, temp_dir: str) -> str:
    """Prepara file audio ottimizzato."""
    audio_path = os.path.join(temp_dir, "audio.wav")
    
    try:
        # Prova prima con pydub se disponibile
        if PYDUB_AVAILABLE:
            audio_bytes = uploaded_audio.read()
            uploaded_audio.seek(0)
            
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            # Converti a mono e riduci sample rate per memoria
            audio = audio.set_channels(1).set_frame_rate(22050)
            audio.export(audio_path, format="wav")
        else:
            # Fallback: salva raw e converti con librosa
            raw_path = os.path.join(temp_dir, "raw_audio")
            with open(raw_path, "wb") as f:
                f.write(uploaded_audio.read())
            
            # Carica e risalva con librosa
            y, sr = librosa.load(raw_path, sr=22050, mono=True)
            sf.write(audio_path, y, sr)
            os.remove(raw_path)
            
        return audio_path
    except Exception as e:
        st.error(f"Errore preparazione audio: {e}")
        raise

# --- UI OTTIMIZZATA ---

uploaded_audio = st.file_uploader("🎵 Audio (MP3/WAV, max 4 minuti)", type=["mp3", "wav"])

# Calcola memoria disponibile
available_memory = check_memory_available()

with st.sidebar:
    st.header("⚙️ Impostazioni")
    
    # Selettore durata massima
    max_duration = st.slider("⏱️ Durata max", 5, 240, 60, step=5, 
                            help="In secondi (max 4 minuti)")
    
    # Limite dimensione file
    max_size = st.selectbox("📦 Dimensione max", [50, 100, 150, 200], 
                           index=3, help="Dimensione massima video in MB")
    
    # Se abbiamo un audio, calcola impostazioni ottimali
    if uploaded_audio:
        try:
            duration, _ = get_audio_info(uploaded_audio.read())
            uploaded_audio.seek(0)
            actual_duration = min(duration, max_duration)
            
            # Calcola impostazioni ottimali
            opt_width, opt_height, opt_fps, est_size = get_optimal_settings(actual_duration, max_size)
            
            st.write(f"**🎯 Impostazioni auto:**")
            st.write(f"📐 {opt_width}x{opt_height} ({opt_height}p)")
            st.write(f"🎬 {opt_fps} FPS")
            st.write(f"📦 ~{est_size}MB")
            
            # Opzione manuale
            manual_mode = st.checkbox("✏️ Impostazioni manuali")
            
            if manual_mode:
                preset = st.selectbox("📊 Qualità", [
                    "240p (426x240)",
                    "360p (640x360)", 
                    "480p (854x480)",
                    "720p (1280x720)",
                    "1080p (1920x1080)"
                ])
                
                if "240p" in preset:
                    width, height, fps = 426, 240, 8
                elif "360p" in preset:
                    width, height, fps = 640, 360, 10
                elif "480p" in preset:
                    width, height, fps = 854, 480, 12
                elif "720p" in preset:
                    width, height, fps = 1280, 720, 15
                else:  # 1080p
                    width, height, fps = 1920, 1080, 20
                    
                fps = st.slider("🎬 FPS", 5, 30, fps)
                
                # Mostra stima dimensione
                est_size_manual = estimate_video_size(width, height, fps, actual_duration)
                if est_size_manual > max_size:
                    st.error(f"⚠️ Stima {est_size_manual:.0f}MB > {max_size}MB limite!")
                else:
                    st.success(f"✅ Stima {est_size_manual:.0f}MB")
            else:
                width, height, fps = opt_width, opt_height, opt_fps
                
        except:
            # Default se non riusciamo a leggere l'audio
            width, height, fps = 640, 360, 10
            st.write("📊 Default: 360p, 10fps")
    else:
        width, height, fps = 640, 360, 10
        st.write("📊 Default: 360p, 10fps")
    
    # Altre opzioni
    st.markdown("---")
    pattern = st.radio("🎨 Pattern", ["Mandala", "Onde"])
    complexity = st.slider("🔧 Complessità", 1, 5, 3)
    thickness = st.slider("📏 Spessore", 1, 5, 2) if pattern == "Mandala" else 1

# Status sistema
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Status Sistema")
st.sidebar.write(f"💾 RAM libera: {available_memory:.0f}MB")
st.sidebar.write(f"🔧 PyDub: {'✅' if PYDUB_AVAILABLE else '❌'}")
st.sidebar.write(f"🎬 FFmpeg: {'✅' if check_ffmpeg() else '❌'}")

# Avvisi memoria
if available_memory < 300:
    st.sidebar.error("⚠️ Memoria bassa! Usa impostazioni minime")
elif available_memory < 500:
    st.sidebar.warning("⚠️ Memoria limitata - raccomandati 360p/480p")

# --- METRICHE AGGIORNATE CON CAST SICURI ---
st.info(f"🎼 BPM: {float(tempo):.0f} | ⏱️ {float(actual_duration):.1f}s | 🎮 {total_frames} frame | 📆 ~{float(final_estimated_size):.0f}MB")

# Metriche principali
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎮 Risoluzione", f"{width}x{height}")
with col2:
    st.metric("⚡ FPS", fps)
with col3:
    if uploaded_audio:
        try:
            duration, _ = get_audio_info(uploaded_audio.read())
            uploaded_audio.seek(0)
            duration_float = float(duration) if hasattr(duration, '__float__') else float(np.asscalar(np.asarray(duration)))
            st.metric("⏱️ Durata", f"{min(duration_float, float(max_duration)):.1f}s")
        except:
            st.metric("⏱️ Durata", "N/A")
    else:
        st.metric("⏱️ Durata", "0s")
with col4:
    st.metric("📆 Limite", f"{max_size}MB")

# Bottone principale
if st.button("🚀 **GENERA VIDEO**", type="primary"):
    if not uploaded_audio:
        st.error("⚠️ Carica un file audio!")
        st.stop()
    
    if not check_ffmpeg():
        st.error("❌ FFmpeg richiesto ma non disponibile!")
        st.info("💡 Su Streamlit Cloud: aggiungi 'ffmpeg' in packages.txt")
        st.stop()
    
    # Controllo memoria
    if available_memory < 200:
        st.error("❌ Memoria insufficiente (< 200MB)")
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
        video_path = os.path.join(temp_dir, "video.mp4")
        final_path = os.path.join(temp_dir, "final.mp4")
        
        # Prepara audio
        audio_path = prepare_audio_file(uploaded_audio, temp_dir)
        progress_bar.progress(0.05)
        
        status_text.text("🎵 Analisi audio...")
        
        # Analisi audio
        audio_data, beat_times, tempo = analyze_audio_minimal(audio_path, max_duration)
        actual_duration = min(len(audio_data) / 11025, max_duration)
        total_frames = int(actual_duration * fps)
        
        if total_frames == 0:
            st.error("❌ Audio troppo corto o non valido")
            st.stop()
            
        samples_per_frame = len(audio_data) // total_frames if total_frames > 0 else len(audio_data)
        
        # Stima finale dimensione
        final_estimated_size = estimate_video_size(width, height, fps, actual_duration)
        
        st.info(f"🎼 BPM: {tempo:.0f} | ⏱️ {actual_duration:.1f}s | 🎬 {total_frames} frame | 📦 ~{final_estimated_size:.0f}MB")
        progress_bar.progress(0.1)
        
        if final_estimated_size > max_size * 1.2:  # 20% tolleranza
            st.warning(f"⚠️ Video potrebbe superare {max_size}MB - continuando comunque")
        
        status_text.text("🎬 Creazione video...")
        
        # Video writer ottimizzato
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            st.error("❌ Impossibile creare video writer")
            st.stop()
        
        # Generazione frame con controllo memoria
        memory_check_interval = max(20, total_frames // 20)  # Controlla memoria più spesso per video lunghi
        
        for i in range(total_frames):
            try:
                # Controllo memoria periodico
                if i % memory_check_interval == 0:
                    current_memory = check_memory_available()
                    if current_memory < 150:  # Se scende sotto 150MB
                        st.warning("⚠️ Memoria critica - forzando cleanup")
                        gc.collect()
                
                # Estrai chunk audio
                start_idx = i * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Analizza chunk
                rms, freq_data = process_frame_data(chunk)
                
                # Rileva beat
                frame_time = float(i) / float(fps)
                beat_detected = (len(beat_times) > 0 and 
                               np.any(np.abs(beat_times - frame_time) < (0.5 / fps)))
                
                # Crea frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                if pattern == "Mandala":
                    frame = draw_minimal_mandala(frame, width, height, rms, complexity, 
                                               thickness, beat_detected, freq_data)
                else:
                    frame = draw_minimal_waves(frame, width, height, rms, complexity, 
                                             i, beat_detected, freq_data)
                
                out.write(frame)
                
                # Progresso
                if i % 10 == 0:
                    progress_val = 0.1 + 0.7 * (i + 1) / total_frames
                    progress_bar.progress(progress_val)
                    
                    # Mostra ETA per video lunghi
                    if total_frames > 1000:
                        eta_seconds = (total_frames - i) * (time.time() - start_time if 'start_time' in locals() else 0) / max(i, 1)
                        eta_text = f" (ETA: {eta_seconds/60:.1f}min)" if eta_seconds > 60 else f" (ETA: {eta_seconds:.0f}s)"
                    else:
                        eta_text = ""
                    
                    status_text.text(f"🎨 Frame {i+1}/{total_frames} ({((i+1)/total_frames*100):.1f}%){eta_text}")
                    
                if i % 50 == 0:  # Cleanup più frequente per video lunghi
                    gc.collect()
                    
            except Exception as e:
                st.warning(f"Errore frame {i}: {e}")
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(frame)
            
            # Salva start time per ETA
            if i == 0:
                start_time = time.time()
        
        out.release()
        if 'out' in locals():
            del out
        gc.collect()
        
        progress_bar.progress(0.8)
        status_text.text("🔧 Encoding finale...")
        
        # FFmpeg con settings adattivi
        crf = '23' if final_estimated_size < 100 else '28'  # Qualità adattiva
        preset = 'fast' if total_frames < 3000 else 'ultrafast'  # Velocità adattiva
        
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_path,
            '-i', audio_path,
            '-t', str(actual_duration),
            '-c:v', 'libx264', '-preset', preset, '-crf', crf,
            '-c:a', 'aac', '-b:a', '96k',
            '-movflags', '+faststart',
            '-maxrate', '2M', '-bufsize', '4M',  # Limita bitrate per controllo dimensione
            final_path
        ]
        
        # Timeout adattivo (più tempo per video lunghi)
        timeout = min(600, max(120, total_frames // 10))  # 2-10 minuti
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0 and os.path.exists(final_path):
            progress_bar.progress(1.0)
            status_text.text("✅ Video completato!")
            
            # Leggi e verifica dimensione
            with open(final_path, 'rb') as f:
                video_bytes = f.read()
            
            file_size_mb = len(video_bytes) / (1024 * 1024)
            
            if file_size_mb <= max_size:
                st.success(f"🎉 **Video generato!** ({file_size_mb:.1f} MB ✅)")
            else:
                st.warning(f"⚠️ **Video generato** ({file_size_mb:.1f} MB) - supera limite {max_size}MB")
            
            # Preview condizionale
            if file_size_mb < 20:  # Preview solo per file piccoli
                st.video(video_bytes)
            else:
                st.info("📹 File troppo grande per preview - usa il download")
            
            # Download button
            filename = f"synesthetic_{uploaded_audio.name.split('.')[0]}_{int(actual_duration)}s.mp4"
            st.download_button(
                label=f"⬇️ **SCARICA VIDEO** ({file_size_mb:.1f}MB)",
                data=video_bytes,
                file_name=filename,
                mime="video/mp4"
            )
            
        else:
            st.error(f"❌ Errore FFmpeg: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        st.error("⏱️ Timeout durante encoding - video troppo lungo o complesso")
    except MemoryError:
        st.error("💾 Memoria insufficiente - riduci durata o qualità")
    except Exception as e:
        st.error(f"❌ Errore: {str(e)}")
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            st.info("💡 Memoria insufficiente - riduci durata o qualità")
            
    finally:
        # Cleanup finale
        try:
            if 'out' in locals():
                out.release()
                del out
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        gc.collect()

else:
    st.info("📝 **Novità - Video fino a 4 minuti o 200MB:**")
    st.write("""
    - ✅ **Fino a 4 minuti** di audio supportati
    - ✅ **Controllo automatico dimensioni** - calcolo ottimale qualità/dimensione
    - ✅ **Risoluzioni standard 16:9**: 240p, 360p, 480p, 720p, 1080p
    - ✅ **Controllo memoria dinamico** durante elaborazione
    - ✅ **ETA progress** per video lunghi
    - ⚠️ **Raccomandato**: file MP3 < 10MB per performance ottimali
    """)
    
    with st.expander("🎯 Sistema adattivo qualità/dimensione"):
        st.write("""
        **Il sistema calcola automaticamente:**
        - **Risoluzione ottimale** per rispettare il limite di dimensione
        - **FPS adattivo** (8-20 fps) in base alla durata
        - **Bitrate dinamico** per ottimizzare qualità/dimensione
        - **CRF adattivo** (23 per file piccoli, 28 per grandi)
        - **Preset FFmpeg** (fast/ultrafast) in base al numero di frame
        
        **Risoluzioni 16:9 supportate:**
        - 240p: 426 × 240 (ultra-low, ~0.4MB/min)
        - 360p: 640 × 360 (low, ~0.6MB/min)
        - 480p: 854 × 480 (medium, ~0.8MB/min)  
        - 720p: 1280 × 720 (HD, ~1.0MB/min)
        - 1080p: 1920 × 1080 (Full HD, ~1.3MB/min)
        """)
        
    with st.expander("🚨 Risoluzione problemi avanzata"):
        st.write("""
        **Errori video lunghi:**
        - **"Timeout encoding"** → Riduci qualità o durata, sistema troppo lento
        - **"Memory error durante frame X"** → Riduci risoluzione o complessità
        - **"File size > limit"** → Sistema sceglierà automaticamente qualità più bassa
        
        **Ottimizzazioni per video lunghi:**
        - Cleanup memoria ogni 50 frame (vs 20 per video corti)
        - ETA dinamico mostrato durante elaborazione
        - Timeout FFmpeg adattivo (2-10 minuti)
        - Controllo memoria critica ogni N frame
        
        **Requirements per 4 minuti:**
        ```
        streamlit
        numpy
        opencv-python-headless
        librosa
        soundfile
        pydub  
        psutil
        ```
        """)
