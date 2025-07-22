import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time
import os
import subprocess

# --- CONFIGURAZIONI FORMATO ---
VIDEO_FORMATS = {
    "16:9 (Landscape) - 1280x720": (1280, 720),
    "1:1 (Square) - 720x720": (720, 720), 
    "9:16 (Portrait) - 720x1280": (720, 1280)
}

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

def get_optimal_settings(duration, width, height):
    fps = 20
    estimated_size = (width * height * fps * duration) / (1024 * 1024)  # rough estimate
    return width, height, fps, int(estimated_size)

def analyze_frequency_bands(freq_data):
    """Analizza le bande di frequenza (basse, medie, acute)"""
    if len(freq_data) == 0:
        return 0, 0, 0
    
    # Dividi lo spettro in 3 bande
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
    """Converte colore hex in formato BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR format

def get_frequency_colors(low_freq, mid_freq, high_freq, low_color, mid_color, high_color, intensity_multiplier=300):
    """Calcola i colori basati sulle frequenze"""
    low_bgr = hex_to_bgr(low_color)
    mid_bgr = hex_to_bgr(mid_color)
    high_bgr = hex_to_bgr(high_color)
    
    # Normalizza e applica intensità
    low_intensity = min(255, int(low_freq * intensity_multiplier))
    mid_intensity = min(255, int(mid_freq * intensity_multiplier))
    high_intensity = min(255, int(high_freq * intensity_multiplier))
    
    # Mescola i colori basandosi sull'intensità delle frequenze
    mixed_color = (
        min(255, int((low_bgr[0] * low_intensity + mid_bgr[0] * mid_intensity + high_bgr[0] * high_intensity) / 255)),
        min(255, int((low_bgr[1] * low_intensity + mid_bgr[1] * mid_intensity + high_bgr[1] * high_intensity) / 255)),
        min(255, int((low_bgr[2] * low_intensity + mid_bgr[2] * mid_intensity + high_bgr[2] * high_intensity) / 255))
    )
    
    return mixed_color, (low_bgr, mid_bgr, high_bgr), (low_intensity, mid_intensity, high_intensity)

def draw_minimal_mandala(frame_img, width, height, rms, param1, param2, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    if color_settings['use_frequency_colors']:
        main_color, freq_colors, intensities = get_frequency_colors(
            low_freq, mid_freq, high_freq,
            color_settings['low_freq_color'],
            color_settings['mid_freq_color'], 
            color_settings['high_freq_color']
        )
    else:
        main_color = (int(min(255, rms * 500)), 50, 150)
    
    center = (width // 2, height // 2)
    max_radius = min(width, height) // 6
    base_radius = int(min(max_radius, rms * 150))
    
    # Disegna cerchi concentrici per diverse frequenze
    if color_settings['use_frequency_colors']:
        # Cerchio esterno per frequenze basse
        low_radius = base_radius + int(low_freq * 50)
        cv2.circle(frame_img, center, min(low_radius, max_radius), freq_colors[0], thickness=3 if beat else 1)
        
        # Cerchio medio per frequenze medie  
        mid_radius = int(base_radius * 0.7) + int(mid_freq * 30)
        cv2.circle(frame_img, center, min(mid_radius, max_radius), freq_colors[1], thickness=2)
        
        # Cerchio interno per frequenze acute
        high_radius = int(base_radius * 0.4) + int(high_freq * 20)
        cv2.circle(frame_img, center, min(high_radius, max_radius), freq_colors[2], thickness=4 if beat else 2)
    else:
        cv2.circle(frame_img, center, base_radius, main_color, thickness=3 if beat else 1)
    
    return frame_img

def draw_minimal_waves(frame_img, width, height, rms, param1, frame_idx, beat, freq_data, color_settings):
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    
    amplitude = int(min(height // 4, rms * 200))
    step = max(10, width // 50)
    
    for x in range(0, width, step):
        if color_settings['use_frequency_colors']:
            # Usa frequenze diverse per onde diverse
            low_y = int(height / 2 + amplitude * np.sin(0.05 * x + frame_idx * 0.1) * (1 + low_freq))
            mid_y = int(height / 2 + amplitude * np.sin(0.1 * x + frame_idx * 0.2) * (1 + mid_freq))
            high_y = int(height / 2 + amplitude * np.sin(0.2 * x + frame_idx * 0.3) * (1 + high_freq))
            
            circle_radius = max(2, min(8, width // 160))
            
            # Disegna onde per ogni banda di frequenza
            low_color = hex_to_bgr(color_settings['low_freq_color'])
            mid_color = hex_to_bgr(color_settings['mid_freq_color'])
            high_color = hex_to_bgr(color_settings['high_freq_color'])
            
            if low_freq > 0.01:
                cv2.circle(frame_img, (x, max(0, min(height-1, low_y))), circle_radius, low_color, -1)
            if mid_freq > 0.01:
                cv2.circle(frame_img, (x, max(0, min(height-1, mid_y))), circle_radius//2, mid_color, -1)
            if high_freq > 0.01:
                cv2.circle(frame_img, (x, max(0, min(height-1, high_y))), max(1, circle_radius//3), high_color, -1)
        else:
            y = int(height / 2 + amplitude * np.sin(0.1 * x + frame_idx * 0.2))
            color = (150, 50, int(min(255, rms * 500)))
            circle_radius = max(2, min(8, width // 160))
            cv2.circle(frame_img, (x, y), circle_radius, color, -1)
    
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

# Selezione formato video
st.subheader("📐 Formato Video")
selected_format = st.selectbox(
    "Scegli il rapporto di aspetto:",
    list(VIDEO_FORMATS.keys()),
    index=0  # Default 16:9
)

width, height = VIDEO_FORMATS[selected_format]
st.info(f"📺 Formato selezionato: **{selected_format}** - Risoluzione: {width}x{height}px")

# --- CONTROLLI COLORI ---
st.subheader("🎨 Controlli Colori")

col1, col2 = st.columns(2)

with col1:
    use_frequency_colors = st.checkbox("🌈 Usa colori basati su frequenze", value=True)
    background_color = st.color_picker("🖤 Colore sfondo", value="#000000")

with col2:
    if use_frequency_colors:
        st.markdown("**Colori Frequenze:**")
        low_freq_color = st.color_picker("🔴 Frequenze Basse", value="#FF0000")
        mid_freq_color = st.color_picker("🟢 Frequenze Medie", value="#00FF00") 
        high_freq_color = st.color_picker("🔵 Frequenze Acute", value="#0080FF")
    else:
        low_freq_color = "#FF0000"
        mid_freq_color = "#00FF00"
        high_freq_color = "#0080FF"
        st.info("Abilita 'Colori basati su frequenze' per personalizzare")

# Salva impostazioni colori
color_settings = {
    'use_frequency_colors': use_frequency_colors,
    'background_color': background_color,
    'low_freq_color': low_freq_color,
    'mid_freq_color': mid_freq_color,
    'high_freq_color': high_freq_color
}

uploaded_audio = st.file_uploader("🎵 Carica file audio", type=["wav", "mp3", "ogg"])

if uploaded_audio:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = prepare_audio_file(uploaded_audio, temp_dir)
            y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
            duration = librosa.get_duration(filename=audio_path)
            width, height, fps, est_size = get_optimal_settings(duration, width, height)
            
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
                
                # Colore di sfondo
                bg_color_bgr = hex_to_bgr(color_settings['background_color'])
                
                st.info("🎬 Generazione frames video...")
                
                for frame_idx in range(frame_count):
                    start_time = frame_idx * frame_duration
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + int(frame_duration * sr)
                    audio_chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                    rms, freq_data = process_frame_data(audio_chunk)
                    beat = np.any((beat_times >= start_time) & (beat_times < start_time + frame_duration))
                    
                    # Crea frame con colore di sfondo personalizzato
                    frame_img = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)
                    
                    if frame_idx % 2 == 0:
                        frame_img = draw_minimal_mandala(frame_img, width, height, rms, 5, 2, beat, freq_data, color_settings)
                    else:
                        frame_img = draw_minimal_waves(frame_img, width, height, rms, 5, frame_idx, beat, freq_data, color_settings)
                    
                    video_writer.write(frame_img)
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
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
                    
                    # Nome file basato sul formato selezionato
                    format_name = selected_format.split(" ")[0].replace(":", "x")
                    filename = f"synesthetic_flow_{format_name}_{width}x{height}.mp4"
                    
                    st.download_button(
                        "⬇️ Scarica Video Completo", 
                        video_bytes, 
                        file_name=filename, 
                        mime="video/mp4"
                    )
                else:
                    st.warning(f"⚠️ {message}")
                    st.info("📹 Scarica il video senza audio:")
                    with open(video_temp, "rb") as f:
                        video_bytes = f.read()
                    
                    format_name = selected_format.split(" ")[0].replace(":", "x")
                    filename = f"synesthetic_flow_{format_name}_{width}x{height}_no_audio.mp4"
                    
                    st.download_button(
                        "⬇️ Scarica Video (solo visivo)", 
                        video_bytes, 
                        file_name=filename, 
                        mime="video/mp4"
                    )

        except Exception as e:
            st.error(f"❌ Errore generazione video: {e}")

# Informazioni tecniche nell'expander
with st.expander("ℹ️ Informazioni Tecniche"):
    st.markdown("""
    **SynestheticFlow Enhanced** genera visualizzazioni avanzate sincronizzate con l'audio:
    
    - 🎵 **Analisi Audio**: Rileva BPM, beat e analizza frequenze (basse, medie, acute)
    - 🌈 **Colori Dinamici**: Colori che reagiscono alle diverse bande di frequenza
    - 🎨 **Personalizzazione**: Controllo completo su colori di frequenze e sfondo  
    - 📐 **Formati**: 3 rapporti ottimizzati (16:9, 1:1, 9:16)
    - 🎬 **Sincronizzazione**: 20 FPS per fluidità ottimale
    - 🔊 **Audio**: Incorporato nel video finale via ffmpeg
    
    **Nuove Funzionalità**:
    - **Analisi Frequenze**: Separa basse (bassi), medie (voci), acute (hi-hat)
    - **Colori Reattivi**: Ogni banda di frequenza ha un colore personalizzabile
    - **Sfondo Personalizzabile**: Scegli il colore di sfondo del video
    - **Visualizzazioni Avanzate**: Mandala e onde multi-frequenza
    
    **Formati disponibili**:
    - **16:9 (Landscape)**: 1280x720px - Ideale per YouTube, monitor widescreen
    - **1:1 (Square)**: 720x720px - Perfetto per Instagram post
    - **9:16 (Portrait)**: 720x1280px - Ottimizzato per TikTok, Instagram Stories
    
    **Requisiti**: ffmpeg installato sul sistema per l'audio
    """)
