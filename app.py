import streamlit as st
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import tempfile
from pydub import AudioSegment
import librosa
import soundfile as sf
import subprocess
import math

st.set_page_config(page_title="🎶 SynestheticFlow", layout="wide")

st.markdown(
    """
    # 🎶 SynestheticFlow <span style="font-size:0.5em;">by Loop507</span>
    """,
    unsafe_allow_html=True
)

st.write("Crea visualizzazioni dinamiche e pattern reattivi alla tua musica!")

# --- FUNZIONI DI DISEGNO DEI PATTERN ---

def draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames):
    """Disegna un pattern di cerchi concentrici che reagisce al volume, con dinamiche potenziate."""
    center_x, center_y = frame_width // 2, frame_height // 2

    base_radius = min(frame_width, frame_height) // 3
    num_circles = int(5 + normalized_rms * visual_complexity * 3)

    for j in range(num_circles):
        radius = int(base_radius * (0.8 + normalized_rms * 0.9) * (j / num_circles + 0.1))
        
        color_val = int(255 * (normalized_rms + j / num_circles) / 1.5)
        circle_color = (min(255, color_val), min(255, 100 + color_val // 2), min(255, 200 + color_val))
        cv2.circle(frame_img, (center_x, center_y), radius, circle_color, line_thickness)
    return frame_img

def draw_fluid_gradient_pattern(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, i, total_frames):
    """
    Genera un pattern fluido e organico basato su gradienti di colore generati matematicamente,
    simulando le immagini di riferimento. Opera a livello di pixel.
    """
    # Genera una griglia di coordinate per tutti i pixel del frame
    y_coords, x_coords = np.indices((frame_height, frame_width))

    # Normalizza le coordinate a un intervallo da -1 a 1 per facilitare i calcoli matematici
    x_norm = (x_coords / frame_width) * 2 - 1
    y_norm = (y_coords / frame_height) * 2 - 1

    # Componente temporale per l'animazione, influenzata dal volume per maggiore dinamicità
    t = i * 0.01 + normalized_rms * 0.5 

    # --- Generazione del campo d'onda complesso ---
    # Combinazione di più funzioni d'onda (seno/coseno) per creare complessità e fluidità
    # Variazioni spaziali e temporali

    # Onda di base che si muove in diagonale
    wave_field1 = np.sin(x_norm * 10 + y_norm * 8 + t * 5 + normalized_rms * 3) * 0.5
    
    # Onda radiale che pulsa dal centro
    r = np.sqrt(x_norm**2 + y_norm**2)
    angle = np.arctan2(y_norm, x_norm)
    wave_field2 = np.cos(r * (12 + visual_complexity) + t * (4 + normalized_rms * 2)) * 0.6
    
    # Onda che crea un effetto vortice/spirale
    swirl_strength = 0.5 + normalized_rms * 0.3
    swirl_x = x_norm * np.cos(angle + t * 0.5) - y_norm * np.sin(angle + t * 0.5)
    swirl_y = x_norm * np.sin(angle + t * 0.5) + y_norm * np.cos(angle + t * 0.5)
    wave_field3 = np.sin(swirl_x * (8 + visual_complexity) + swirl_y * (7 + visual_complexity) + t * 6) * 0.7

    # Onda aggiuntiva per maggiore complessità e intersezioni
    wave_field4 = np.cos(x_norm * (visual_complexity * 2) + t * 2 + np.sin(y_norm * 5)) * 0.4

    # Combina tutti i campi d'onda
    # L'intensità complessiva delle onde è influenzata dal volume
    total_wave_value = (wave_field1 + wave_field2 + wave_field3 + wave_field4) * (0.8 + normalized_rms * 0.7)

    # Normalizza il valore complessivo a un intervallo utile per la mappatura del colore (es. 0-1)
    normalized_value = np.clip(total_wave_value, -3, 3) # Limita i valori estremi
    normalized_value = (normalized_value + 3) / 6 # Rescala a 0-1

    # --- Mappatura del colore (usando HSV per gradienti vividi e controllati) ---
    # Tonalità (Hue): Varia in base al valore dell'onda, al tempo e al volume
    hue = (normalized_value * 200 + t * 30 + normalized_rms * 60) % 180 # Intervallo HSV hue è 0-179

    # Saturazione: Alta per colori vivaci, con leggera influenza del volume
    saturation = 200 + (normalized_rms * 55) # Sempre molto saturo
    saturation = np.clip(saturation, 0, 255)

    # Valore (Brightness): Pulsazione più intensa con il volume, con variazione basata sull'onda
    value = (normalized_value * 180 + normalized_rms * 70).astype(np.uint8) # Luminosità di base + influenza RMS
    value = np.clip(value, 0, 255) # Assicura che i valori rientrino nel range valido

    # Crea un'immagine HSV stackando i canali
    hsv_image = np.stack([hue.astype(np.uint8), saturation * np.ones_like(hue, dtype=np.uint8), value], axis=-1)
    
    # Converte l'immagine da HSV a BGR (formato usato da OpenCV)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Applica l'immagine generata al frame attuale
    frame_img[:] = bgr_image

    return frame_img

# --- UI per Caricamento Audio e Controlli (Spostati nella Sidebar) ---

uploaded_audio = st.file_uploader("Carica un file audio (MP3, WAV)", type=["mp3", "wav"])

st.sidebar.subheader("Impostazioni Generazione Visual")
num_frames_per_second = st.sidebar.slider("Frame al secondo (FPS)", 15, 60, 24)
visual_complexity = st.sidebar.slider("Complessità visiva", 1, 10, 5)
# line_thickness è meno rilevante per i pattern basati su pixel, ma lo lascio per Mandala Semplice
line_thickness = st.sidebar.slider("Spessore linee (per Mandala Semplice)", 1, 5, 2)
pattern_type = st.sidebar.selectbox("Tipo di Pattern", ["Mandala Semplice (Cerchi)", "Flusso Ondulato a Gradiente"])


generate_button = st.button("✨ Genera Visual e Video")

if generate_button and uploaded_audio is not None:
    st.info("Inizio elaborazione... potrebbe volerci del tempo.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
            tmp_audio_file.write(uploaded_audio.read())
            audio_path = tmp_audio_file.name

        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio)
        audio_duration_sec = audio_duration_ms / 1000.0

        st.write(f"Audio caricato: **{uploaded_audio.name}**")
        st.write(f"Durata audio: **{audio_duration_sec:.2f}** secondi")

        total_frames = int(audio_duration_sec * num_frames_per_second)
        st.write(f"Genererò circa **{total_frames}** frame per il video.")

        all_visual_frames = []
        
        frame_width, frame_height = 1280, 720
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        y, sr = sf.read(audio_path)

        samples_per_frame = int(sr / num_frames_per_second)
        
        for i in range(total_frames):
            start_sample = i * samples_per_frame
            end_sample = min((i + 1) * samples_per_frame, len(y))
            
            if start_sample >= len(y):
                break

            audio_chunk = y[start_sample:end_sample]

            rms_energy = np.sqrt(np.mean(audio_chunk**2))
            # Normalizzazione RMS più robusta per evitare divisioni per zero anche con volumi molto bassi
            normalized_rms = np.log10(rms_energy + 1e-10) / np.log10(1.0 + 1e-10) 
            normalized_rms = np.clip(normalized_rms, 0, 1)

            frame_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            if pattern_type == "Mandala Semplice (Cerchi)":
                frame_img = draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames)
            elif pattern_type == "Flusso Ondulato a Gradiente": # Rinominato
                frame_img = draw_fluid_gradient_pattern(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, i, total_frames)
            
            all_visual_frames.append(frame_img)
            progress_bar.progress((i + 1) / total_frames)
            progress_text.text(f"Generazione frame: {i+1}/{total_frames}")

        progress_text.empty()


        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_output:
            video_filepath = tmp_video_output.name

        writer = imageio.get_writer(video_filepath, fps=num_frames_per_second)
        for frame in all_visual_frames:
            writer.append_data(frame)
        writer.close()

        final_video_filepath = video_filepath.replace(".mp4", "_final.mp4")
        
        command = [
            'ffmpeg',
            '-i', video_filepath,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            final_video_filepath
        ]
        
        try:
            st.write("Combinazione video e audio...")
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            st.success("✅ Video con audio generato con successo!")

            with open(final_video_filepath, "rb") as f:
                st.download_button(
                    label="⬇️ Scarica il tuo Video Visual",
                    data=f,
                    file_name="synesthetic_flow_visual.mp4",
                    mime="video/mp4"
                )

        except subprocess.CalledProcessError as e:
            st.error(f"Errore durante la combinazione di video/audio (FFmpeg): {e.stderr}")
            st.info("Assicurati che FFmpeg sia installato e configurato correttamente nel tuo ambiente.")
        except FileNotFoundError:
            st.error("FFmpeg non trovato. Assicurati che sia installato e nel PATH del tuo ambiente.")


    except Exception as e:
        st.error(f"Si è verificato un errore generale nell'elaborazione: {e}")
        st.error(f"Dettagli: {str(e)}")
        if "No such file or directory" in str(e) and ("ffprobe" in str(e).lower() or "ffmpeg" in str(e).lower()):
            st.error("Sembra che FFmpeg/ffprobe non sia disponibile. Controlla le configurazioni del tuo ambiente (`packages.txt`).")
        elif "unsupported format" in str(e).lower() and uploaded_audio:
            st.error(f"Il formato del file audio caricato ({uploaded_audio.type}) potrebbe non essere supportato. Prova un MP3 o WAV standard.")
        elif "Could not find a backend" in str(e) and "imageio" in str(e).lower():
            st.error("Sembra che il backend per imageio non sia installato correttamente. Controlla `requirements.txt` (`imageio[ffmpeg]`).")


    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'video_filepath' in locals() and os.path.exists(video_filepath):
            os.remove(video_filepath)
        if 'final_video_filepath' in locals() and os.path.exists(final_video_filepath):
            if not os.path.islink(final_video_filepath):
                os.remove(final_video_filepath)

else:
    st.info("Carica un file audio (MP3 o WAV) e premi 'Genera Visual e Video' per creare la tua visualizzazione!")
