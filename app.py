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
import math # Aggiunto per funzioni matematiche come sin/cos

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

    base_radius = min(frame_width, frame_height) // 4
    # Più volume = più cerchi, più grandi
    num_circles = int(3 + normalized_rms * visual_complexity * 2) 

    for j in range(num_circles):
        # Il raggio del cerchio varia con il volume e l'indice del cerchio
        radius = int(base_radius * (1 + normalized_rms * 0.7) * (j / num_circles + 0.1))

        # Il colore cambia con il volume e l'indice del cerchio
        # Un gradiente di colore che si intensifica
        color_val = int(255 * (normalized_rms + j / num_circles) / 2)
        circle_color = (min(255, color_val), min(255, 50 + color_val // 3), min(255, 200 + color_val // 2)) 
        
        cv2.circle(frame_img, (center_x, center_y), radius, circle_color, line_thickness)

    return frame_img

def draw_radial_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames):
    """Disegna un pattern radiale tipo fiore/stella che reagisce al volume."""
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # Colore di sfondo che pulsa leggermente
    bg_color_val = int(normalized_rms * 50) # da 0 a 50
    frame_img[:] = (bg_color_val, bg_color_val, bg_color_val) # Sfondo grigio scuro che pulsa

    # Numero di "petali" o raggi, influenzato da complessità e volume
    num_petals = int(4 + visual_complexity * 2 * (1 + normalized_rms)) 
    num_petals = max(4, num_petals if num_petals % 2 == 0 else num_petals + 1) # Assicurati che sia pari e almeno 4

    max_length = min(frame_width, frame_height) // 2 * (0.8 + normalized_rms * 0.2) # La lunghezza massima pulsa
    
    # Variazione della rotazione nel tempo per un effetto dinamico
    rotation_speed = 0.05 + normalized_rms * 0.1 # Più volume, più rotazione
    current_rotation_offset = (i / total_frames) * (2 * math.pi) * rotation_speed

    for j in range(num_petals):
        angle = (2 * math.pi / num_petals) * j + current_rotation_offset

        # La lunghezza delle linee/petali pulsa con il volume
        length = int(max_length * (0.5 + normalized_rms * 0.5))

        # Punto finale del raggio
        end_x = int(center_x + length * math.cos(angle))
        end_y = int(center_y + length * math.sin(angle))
        
        # Colore che cambia con il volume e l'angolo
        # Usa HSV per un cambio colore più fluido
        hue = int((normalized_rms * 180 + (angle / (2 * math.pi)) * 180) % 180) # Hue da 0-180 in OpenCV
        saturation = int(255 * (0.5 + normalized_rms * 0.5))
        value = int(200 + normalized_rms * 55) # Luminosità che pulsa
        
        # Converte HSV in BGR per OpenCV
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
        
        cv2.line(frame_img, (center_x, center_y), (end_x, end_y), bgr_color, line_thickness)
        
        # Disegna anche un cerchio al centro che pulsa
        circle_radius = int(20 + normalized_rms * 50)
        cv2.circle(frame_img, (center_x, center_y), circle_radius, (255, 255, 255), -1) # Cerchio bianco pieno

    return frame_img

# --- UI per Caricamento Audio e Controlli ---
uploaded_audio = st.file_uploader("Carica un file audio (MP3, WAV)", type=["mp3", "wav"])

st.subheader("Impostazioni Generazione Visual")
# Controlli per i pattern
num_frames_per_second = st.slider("Frame al secondo (FPS)", 15, 60, 24)
visual_complexity = st.slider("Complessità visiva", 1, 10, 5) 
line_thickness = st.slider("Spessore linee", 1, 5, 2)
# Nuovo controllo per il tipo di pattern
pattern_type = st.selectbox("Tipo di Pattern", ["Mandala Semplice (Cerchi)", "Mandala Radiale (Fiori)"])


generate_button = st.button("✨ Genera Visual e Video")

if generate_button and uploaded_audio is not None:
    st.info("Inizio elaborazione... potrebbe volerci del tempo.")

    try:
        # --- SALVATAGGIO TEMPORANEO AUDIO ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
            tmp_audio_file.write(uploaded_audio.read())
            audio_path = tmp_audio_file.name

        # --- CARICAMENTO E ANALISI AUDIO (BASE) ---
        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio) 
        audio_duration_sec = audio_duration_ms / 1000.0

        st.write(f"Audio caricato: **{uploaded_audio.name}**")
        st.write(f"Durata audio: **{audio_duration_sec:.2f}** secondi")

        total_frames = int(audio_duration_sec * num_frames_per_second)
        st.write(f"Genererò circa **{total_frames}** frame per il video.")

        all_visual_frames = []
        frame_width, frame_height = 800, 800 # Puoi rendere questi valori configurabili con slider in futuro
        
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

            # --- ANALISI AUDIO PER IL FRAME CORRENTE ---
            rms_energy = np.sqrt(np.mean(audio_chunk**2)) 
            normalized_rms = np.log10(rms_energy + 1e-6) / np.log10(1.0 + 1e-6)
            normalized_rms = np.clip(normalized_rms, 0, 1)

            # --- DISEGNO DEL MANDALA/PATTERN (ORA DINAMICO) ---
            frame_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8) # Crea un frame nero di base

            if pattern_type == "Mandala Semplice (Cerchi)":
                frame_img = draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames)
            elif pattern_type == "Mandala Radiale (Fiori)":
                frame_img = draw_radial_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames)
            
            all_visual_frames.append(frame_img)
            progress_bar.progress((i + 1) / total_frames)
            progress_text.text(f"Generazione frame: {i+1}/{total_frames}")

        progress_text.empty() # Pulisci il testo di progresso alla fine


        # --- GENERAZIONE VIDEO SOLO VISUALE ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_output:
            video_filepath = tmp_video_output.name

        writer = imageio.get_writer(video_filepath, fps=num_frames_per_second)
        for frame in all_visual_frames:
            writer.append_data(frame)
        writer.close()

        # --- COMBINAZIONE VIDEO E AUDIO (USANDO FFmpeg) ---
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
            st.info("Assicurati che FFmpeg sia installato e configurato correttamente nel tuo ambiente Glitch.")
        except FileNotFoundError:
            st.error("FFmpeg non trovato. Assicurati che sia installato e nel PATH del tuo ambiente Glitch.")


    except Exception as e:
        st.error(f"Si è verificato un errore generale nell'elaborazione: {e}")
        if "No such file or directory" in str(e) and "ffprobe" in str(e).lower():
            st.error("Sembra che ffprobe (parte di FFmpeg) non sia disponibile. Controlla le configurazioni del tuo ambiente.")
        if "unsupported format" in str(e).lower() and uploaded_audio:
            st.error(f"Il formato del file audio caricato ({uploaded_audio.type}) potrebbe non essere supportato da pydub o librosa. Prova un MP3 o WAV standard.")

    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'video_filepath' in locals() and os.path.exists(video_filepath):
            os.remove(video_filepath)
        # final_video_filepath non rimosso qui per consentire il download
else:
    st.info("Carica un file audio (MP3 o WAV) e premi 'Genera Visual e Video' per creare la tua visualizzazione!")
