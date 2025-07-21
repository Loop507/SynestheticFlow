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
    """Disegna un pattern di cerchi concentrici che reagisce al volume."""
    center_x, center_y = frame_width // 2, frame_height // 2

    # Scala base del raggio in base alla dimensione del frame
    base_radius = min(frame_width, frame_height) // 4
    num_circles = int(3 + normalized_rms * visual_complexity * 2)

    for j in range(num_circles):
        radius = int(base_radius * (1 + normalized_rms * 0.7) * (j / num_circles + 0.1))
        color_val = int(255 * (normalized_rms + j / num_circles) / 2)
        circle_color = (min(255, color_val), min(255, 50 + color_val // 3), min(255, 200 + color_val // 2))
        cv2.circle(frame_img, (center_x, center_y), radius, circle_color, line_thickness)
    return frame_img

def draw_radial_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames):
    """Disegna un pattern radiale tipo fiore/stella che reagisce al volume, con dinamiche e dettagli potenziati, e ondulazioni."""
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # --- Sfondo pulsante e più dinamico ---
    bg_color_val = int(normalized_rms * 100)
    frame_img[:] = (bg_color_val, bg_color_val, bg_color_val)

    # --- Parametri dinamici basati su complessità e volume ---
    num_petals_base = int(4 + visual_complexity * 2)
    num_petals_volume = int(num_petals_base + normalized_rms * 12) # Aumenta il numero di "petali" con il volume, più aggressivo
    num_petals_volume = max(num_petals_base, num_petals_volume if num_petals_volume % 2 == 0 else num_petals_volume + 1)

    max_length_base = min(frame_width, frame_height) // 2 * (0.5 + visual_complexity / 20)
    max_length = int(max_length_base * (0.7 + normalized_rms * 0.3))

    # --- Rotazione e pulsazione più complesse ---
    rotation_speed = 0.05 + normalized_rms * 0.2 + (visual_complexity * 0.01)
    current_rotation_offset = (i / total_frames) * (2 * math.pi) * rotation_speed + (normalized_rms * math.pi / 4)

    # --- Parametri per l'ondulazione ---
    wave_frequency = 4 + visual_complexity # Più onde con complessità
    wave_amplitude = int(10 + normalized_rms * 30 * (visual_complexity / 10)) # Ampiezza dell'onda che pulsa con il volume e complessità
    
    # --- Disegno dello strato principale dei petali/raggi con ondulazione ---
    for j in range(num_petals_volume):
        angle = (2 * math.pi / num_petals_volume) * j + current_rotation_offset
        
        # Disegna una linea curva (simulata con molti piccoli segmenti)
        prev_point = (center_x, center_y)
        num_segments = 20 # Numero di segmenti per disegnare ogni petalo come una curva
        
        for k in range(num_segments + 1):
            segment_t = k / num_segments
            
            # Calcola la lunghezza radiale per il segmento
            length_radial = max_length * segment_t
            
            # Aggiungi un'ondulazione al raggio
            wave_offset = wave_amplitude * math.sin(angle * wave_frequency + i * 0.1 + segment_t * math.pi * 2)
            
            # Calcola il punto lungo l'angolo, con l'offset dell'onda perpendicolare
            current_angle_with_offset = angle + wave_offset / length_radial # Converte l'offset in angolo se necessario

            # Calcola le coordinate del punto con ondulazione
            current_x = int(center_x + length_radial * math.cos(current_angle_with_offset))
            current_y = int(center_y + length_radial * math.sin(current_angle_with_offset))
            
            current_point = (current_x, current_y)
            
            # Colore primario vibrante (influenzato dal volume e dall'angolo)
            hue = int((normalized_rms * 180 + (angle / (2 * math.pi)) * 360) % 180)
            saturation = int(200 + normalized_rms * 55)
            value = int(150 + normalized_rms * 100)
            
            hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            if k > 0:
                cv2.line(frame_img, prev_point, current_point, bgr_color, line_thickness)
            prev_point = current_point

    # --- Disegno di uno strato secondario per maggiore complessità ---
    num_petals_secondary = int(num_petals_volume * 1.5)
    secondary_rotation_offset = current_rotation_offset * 0.5 + math.pi / 4
    
    for j in range(num_petals_secondary):
        angle = (2 * math.pi / num_petals_secondary) * j + secondary_rotation_offset
        
        prev_point = (center_x, center_y)
        num_segments_secondary = 15
        
        for k in range(num_segments_secondary + 1):
            segment_t = k / num_segments_secondary
            length_radial_secondary = max_length * 0.6 * segment_t
            
            wave_offset_secondary = wave_amplitude * 0.7 * math.cos(angle * wave_frequency * 1.5 + i * 0.05 + segment_t * math.pi * 3)
            
            current_angle_with_offset_secondary = angle + wave_offset_secondary / length_radial_secondary if length_radial_secondary > 0 else angle
            
            current_x_secondary = int(center_x + length_radial_secondary * math.cos(current_angle_with_offset_secondary))
            current_y_secondary = int(center_y + length_radial_secondary * math.sin(current_angle_with_offset_secondary))
            
            current_point_secondary = (current_x_secondary, current_y_secondary)
            
            hue_secondary = int((hue + 90) % 180)
            saturation_secondary = int(255 - normalized_rms * 50)
            value_secondary = int(100 + normalized_rms * 155)
            
            hsv_color_secondary = np.array([[[hue_secondary, saturation_secondary, value_secondary]]], dtype=np.uint8)
            bgr_color_secondary = cv2.cvtColor(hsv_color_secondary, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            if k > 0:
                cv2.line(frame_img, prev_point, current_point_secondary, bgr_color_secondary, max(1, line_thickness - 1))
            prev_point = current_point_secondary


    # --- Cerchio centrale pulsante più grande e con bordi ---
    circle_radius_main = int(30 + normalized_rms * 100)
    circle_radius_border = int(circle_radius_main * 1.1)
    
    cv2.circle(frame_img, (center_x, center_y), circle_radius_border, (50, 50, 50), -1)
    cv2.circle(frame_img, (center_x, center_y), circle_radius_main, (255, 255, 255), -1)

    # --- Anello esterno sottile che pulsa ---
    outer_ring_radius = int(min(frame_width, frame_height) // 2 * (0.8 + normalized_rms * 0.1))
    cv2.circle(frame_img, (center_x, center_y), outer_ring_radius, (255, 255, 0), max(1, line_thickness + 1))

    return frame_img

# --- UI per Caricamento Audio e Controlli (Spostati nella Sidebar) ---

uploaded_audio = st.file_uploader("Carica un file audio (MP3, WAV)", type=["mp3", "wav"])

st.sidebar.subheader("Impostazioni Generazione Visual")
num_frames_per_second = st.sidebar.slider("Frame al secondo (FPS)", 15, 60, 24)
visual_complexity = st.sidebar.slider("Complessità visiva", 1, 10, 5)
line_thickness = st.sidebar.slider("Spessore linee", 1, 5, 2)
pattern_type = st.sidebar.selectbox("Tipo di Pattern", ["Mandala Semplice (Cerchi)", "Mandala Radiale (Fiori)"])


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
        
        # --- Modifica la risoluzione dei frame qui ---
        frame_width, frame_height = 1280, 720 # Nuova risoluzione: 1280x720 (HD)
        
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
            normalized_rms = np.log10(rms_energy + 1e-7) / np.log10(1.0 + 1e-7) # Aggiustato a 1e-7
            normalized_rms = np.clip(normalized_rms, 0, 1)

            frame_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            if pattern_type == "Mandala Semplice (Cerchi)":
                frame_img = draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames)
            elif pattern_type == "Mandala Radiale (Fiori)":
                frame_img = draw_radial_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames)
            
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
