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
    pixels_per_second = width * height * fps
    estimated_bitrate_bps = pixels_per_second * 0.1 * bitrate_factor
    total_bitrate_bps = estimated_bitrate_bps + 96000
    size_mb = (total_bitrate_bps * duration) / (8 * 1024 * 1024)
    return size_mb

def get_optimal_settings(duration: float, max_size_mb: int = 200) -> Tuple[int, int, int, int]:
    presets = [
        (426, 240, 8, 0.4),
        (640, 360, 10, 0.6),
        (854, 480, 12, 0.8),
        (1280, 720, 15, 1.0),
        (1920, 1080, 20, 1.3)
    ]
    
    for width, height, fps, bitrate_factor in presets:
        estimated_size = estimate_video_size(width, height, fps, duration, bitrate_factor)
        if estimated_size <= max_size_mb:
            return width, height, fps, int(estimated_size)
    
    return presets[0][0], presets[0][1], presets[0][2], int(estimate_video_size(*presets[0][:3], duration, presets[0][3]))

def check_memory_available() -> float:
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except:
        return 500.0

@st.cache_data
def get_audio_info(audio_bytes: bytes) -> Tuple[float, int]:
    try:
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return len(audio) / 1000.0, audio.frame_rate
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                try:
                    y, sr = librosa.load(tmp.name, sr=None, duration=1)
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
    try:
        cx, cy = width // 2, height // 2
        base_r = min(width, height) // 4
        circles = min(3 + int(float(rms) * complexity), 8)
        
        beat_scale = 1.3 if beat else 1.0
        bass, mid, treble = map(float, freq_data)
        
        colors = [
            (int(np.clip(255 * treble, 0, 255)), int(np.clip(128 * mid, 0, 255)), int(np.clip(255 * bass, 0, 255))),
            (int(np.clip(128 * bass, 0, 255)), int(np.clip(255 * treble, 0, 255)), int(np.clip(128 * mid, 0, 255))),
            (int(np.clip(255 * mid, 0, 255)), int(np.clip(255 * bass, 0, 255)), int(np.clip(128 * treble, 0, 255)))
        ]
        
        for i in range(circles):
            if circles > 0:
                r = int(base_r * (0.3 + 0.7 * i / circles) * beat_scale * (1 + float(rms)))
                r = max(1, r)
                color = colors[i % 3]
                cv2.circle(frame_img, (cx, cy), r, color, thickness)
        
    except Exception:
        cx, cy = width // 2, height // 2
        r = min(width, height) // 6
        cv2.circle(frame_img, (cx, cy), r, (128, 128, 128), thickness)
    
    return frame_img

def draw_minimal_waves(frame_img: np.ndarray, width: int, height: int,
                      rms: float, complexity: int, frame_idx: int,
                      beat: bool, freq_data: Tuple[float, float, float]) -> np.ndarray:
    try:
        calc_w, calc_h = width // 4, height // 4
        
        x = np.linspace(-1, 1, calc_w, dtype=np.float32)
        y = np.linspace(-1, 1, calc_h, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        t = float(frame_idx * 0.1)
        beat_mult = 1.5 if beat else 1.0
        bass, mid, treble = map(float, freq_data)
        
        wave = np.sin(X * 5.0 * beat_mult + t + bass * 3.0) * np.cos(Y * 3.0 + mid * 2.0)
        wave = np.clip((wave + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        hue = np.clip((wave.astype(np.float32) + treble * 50.0) % 180.0, 0, 255)
        
        rgb_small = np.zeros((calc_h, calc_w, 3), dtype=np.uint8)
        rgb_small[:,:,0] = np.clip(hue + bass * 100.0, 0, 255).astype(np.uint8)
        rgb_small[:,:,1] = np.clip(wave.astype(np.float32) + mid * 100.0, 0, 255).astype(np.uint8)
        rgb_small[:,:,2] = np.clip(255.0 - hue + treble * 100.0, 0, 255).astype(np.uint8)
        
        rgb_full = cv2.resize(rgb_small, (width, height), interpolation=cv2.INTER_NEAREST)
        frame_img[:] = rgb_full
        
    except Exception:
        bass, mid, treble = map(float, freq_data)
        color = [
            int(np.clip(bass * 255, 0, 255)),
            int(np.clip(mid * 255, 0, 255)), 
            int(np.clip(treble * 255, 0, 255))
        ]
        frame_img[:] = color
    
    return frame_img

def analyze_audio_minimal(audio_path: str, max_duration: int = 240) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        y, sr = librosa.load(audio_path, sr=11025, mono=True, duration=max_duration)
        gc.collect()
        
        if len(y) > 1000:
            y_beat = y[::8]
            tempo, beats = librosa.beat.beat_track(y=y_beat, sr=sr//8, trim=False)
            beat_times = beats * 8 / sr
        else:
            tempo, beat_times = 120.0, np.array([])
        
        return y, beat_times, tempo
    except Exception as e:
        st.warning(f"Analisi audio semplificata: {e}")
        return np.zeros(max_duration * 11025), np.array([]), 120.0

def process_frame_data(audio_chunk: np.ndarray, sr: int = 11025) -> Tuple[float, Tuple[float, float, float]]:
    try:
        if len(audio_chunk) == 0:
            return 0.0, (0.0, 0.0, 0.0)
        
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)))
        rms_norm = min(1.0, rms * 10.0)
        
        if len(audio_chunk) > 32:
            chunk = audio_chunk[:64] if len(audio_chunk) > 64 else audio_chunk
            chunk = chunk.astype(np.float32)
            
            fft = np.fft.rfft(chunk)
            mag = np.abs(fft).astype(np.float32)
            
            third = len(mag) // 3
            if third > 0:
                bass = float(np.mean(mag[:third]))
                mid = float(np.mean(mag[third:2*third]))
                treble = float(np.mean(mag[2*third:]))
                
                max_mag = max(bass, mid, treble, 1e-6)
                bass = min(1.0, bass / max_mag)
                mid = min(1.0, mid / max_mag)
                treble = min(1.0, treble / max_mag)
            else:
                bass = mid = treble = 0.0
        else:
            bass = mid = treble = 0.0
        
        return rms_norm, (bass, mid, treble)
        
    except Exception:
        return 0.0, (0.0, 0.0, 0.0)

def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def prepare_audio_file(uploaded_audio, temp_dir: str) -> str:
    """Prepara file audio ottimizzato, convertendo con ffmpeg per evitare errori di decoding."""
    audio_path = os.path.join(temp_dir, "audio.wav")
    orig_path = os.path.join(temp_dir, uploaded_audio.name)

    try:
        # Salva il file originale
        with open(orig_path, "wb") as f:
            f.write(uploaded_audio.read())
        uploaded_audio.seek(0)
        
        # Converti in wav mono 22050 Hz con ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", orig_path,
            "-ac", "1",      # mono
            "-ar", "22050",  # sample rate
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.strip()}")
        
        return audio_path

    except Exception as e:
        st.error(f"Errore preparazione audio: {e}")
        raise

def main():
    uploaded_audio = st.file_uploader("Carica file audio (max 4 min, 200MB)", type=["wav", "mp3", "ogg", "flac", "m4a"])
    if not uploaded_audio:
        st.info("Carica un file audio per iniziare.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepara audio
            audio_path = prepare_audio_file(uploaded_audio, temp_dir)
            
            # Analizza audio
            y, beat_times, tempo = analyze_audio_minimal(audio_path)
            duration = librosa.get_duration(filename=audio_path)
            width, height, fps, est_size = get_optimal_settings(duration)
            
            st.info(f"🎼 BPM: {float(tempo):.0f} | ⏱️ {float(duration):.1f}s | 🎮 {fps} FPS | 📆 ~{est_size} MB")
            
            # Visualizzazione placeholder
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
                
                # Usa mandala o waves a rotazione
                if frame_idx % 2 == 0:
                    frame_img = draw_minimal_mandala(frame_img, width, height, rms, 5, 2, beat, freq_data)
                else:
                    frame_img = draw_minimal_waves(frame_img, width, height, rms, 5, frame_idx, beat, freq_data)
                
                frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(frame_duration)
            
        except Exception as e:
            st.error(f"Errore generazione video: {e}")

if __name__ == "__main__":
    if not check_ffmpeg():
        st.error("⚠️ FFmpeg non trovato. Assicurati che sia installato e nel PATH.")
    else:
        main()
