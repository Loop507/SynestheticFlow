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

def draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames, is_beat_in_frame, bass_energy, mid_energy, treble_energy):
    """Disegna un pattern di cerchi concentrici che reagisce al volume, con dinamiche potenziate."""
    center_x, center_y = frame_width // 2, frame_height // 2

    base_radius = min(frame_width, frame_height) // 3
    num_circles = int(5 + normalized_rms * visual_complexity * 3)

    # Aggiungi una reazione al beat: aumenta temporaneamente la dimensione o lo spessore
    beat_effect_scale = 1.0 + (1.5 * is_beat_in_frame) # Se è beat, scala di 1.5x
    
    # Colore influenzato da RMS e frequenze
    r_color = int(255 * (normalized_rms + treble_energy) / 2)
    g_color = int(255 * (normalized_rms + mid_energy) / 2)
    b_color = int(255 * (normalized_rms + bass_energy) / 2)

    for j in range(num_circles):
        radius = int(base_radius * (0.8 + normalized_rms * 0.9) * (j / num_circles + 0.1) * beat_effect_scale)
        
        # Colore influenzato dalla posizione del cerchio e dalle frequenze
        color_val = int(255 * (normalized_rms + j / num_circles) / 1.5)
        
        # Combinazione di colori basata sulle energie delle frequenze
        circle_color = (
            min(255, r_color + int(treble_energy * 100)), 
            min(255, g_color + int(mid_energy * 100)), 
            min(255, b_color + int(bass_energy * 100))
        )
        cv2.circle(frame_img, (center_x, center_y), radius, circle_color, line_thickness)
    return frame_img

def draw_fluid_gradient_pattern(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, i, total_frames, is_beat_in_frame, bass_energy, mid_energy, treble_energy):
    """
    Genera un pattern fluido e organico basato su gradienti di colore generati matematicamente,
    simulando le immagini di riferimento. Opera a livello di pixel.
    Ora reagisce a BPM, Volume e Frequenze.
    """
    y_coords, x_coords = np.indices((frame_height, frame_width))

    x_norm = (x_coords / frame_width) * 2 - 1
    y_norm = (y_coords / frame_height) * 2 - 1

    # Componente temporale per l'animazione, influenzata dal volume e dal beat
    t = i * 0.01 + normalized_rms * 0.5 
    
    # Aggiungi un impulso visivo in corrispondenza del beat
    beat_impulse = 1.0 + (2.0 * is_beat_in_frame) # Forte impatto sul beat

    # --- Generazione del campo d'onda complesso ---
    # Le onde sono influenzate dalla complessità visiva, dal volume e dalle frequenze
    
    # Onda di base che si muove in diagonale, influenzata dal beat e dai medi
    wave_field1 = np.sin(x_norm * (10 * beat_impulse) + y_norm * 8 + t * 5 + normalized_rms * 3 + mid_energy * 5) * 0.5
    
    # Onda radiale che pulsa dal centro, influenzata dal beat e dai bassi
    r = np.sqrt(x_norm**2 + y_norm**2)
    angle = np.arctan2(y_norm, x_norm)
    wave_field2 = np.cos(r * (12 + visual_complexity * beat_impulse) + t * (4 + normalized_rms * 2) + bass_energy * 8) * 0.6
    
    # Onda che crea un effetto vortice/spirale, influenzata dagli alti
    swirl_strength = 0.5 + normalized_rms * 0.3 + treble_energy * 0.2
    swirl_x = x_norm * np.cos(angle + t * 0.5 * swirl_strength) - y_norm * np.sin(angle + t * 0.5 * swirl_strength)
    swirl_y = x_norm * np.sin(angle + t * 0.5 * swirl_strength) + y_norm * np.cos(angle + t * 0.5 * swirl_strength)
    wave_field3 = np.sin(swirl_x * (8 + visual_complexity) + swirl_y * (7 + visual_complexity) + t * 6 + treble_energy * 7) * 0.7

    # Onda aggiuntiva per maggiore complessità e intersezioni, influenzata da tutte le frequenze
    wave_field4 = np.cos(x_norm * (visual_complexity * 2) + t * 2 + np.sin(y_norm * 5) + (bass_energy + mid_energy + treble_energy) * 3) * 0.4

    # Combina tutti i campi d'onda. L'intensità complessiva è influenzata dal volume e dal beat
    total_wave_value = (wave_field1 + wave_field2 + wave_field3 + wave_field4) * (0.8 + normalized_rms * 0.7 + is_beat_in_frame * 0.5)

    normalized_value = np.clip(total_wave_value, -3, 3)
    normalized_value = (normalized_value + 3) / 6

    # --- Calcolo e casting esplicito a np.uint8 per tutti i canali HSV ---
    # I colori sono influenzati dalle energie delle frequenze
    
    # Tonalità (Hue): Varia con l'onda, il tempo, il volume e le frequenze
    hue_float = (normalized_value * 200 + t * 30 + normalized_rms * 60 + bass_energy * 50 + mid_energy * 30 + treble_energy * 70) % 180
    
    # Saturazione: Alta per colori vivaci, con influenza di volume e frequenze
    saturation_float = 200 + (normalized_rms * 55) + (bass_energy + mid_energy + treble_energy) * 30
    
    # Valore (Brightness): Pulsazione più intensa con il volume e il beat, variazione basata sull'onda
    value_float = (normalized_value * 180 + normalized_rms * 70 + is_beat_in_frame * 80)
    
    # CLIPPAGGIO E CASTING A UINT8: Essenziale per OpenCV
    hue_uint8 = np.clip(hue_float, 0, 179).astype(np.uint8) # Hue range: 0-179
    saturation_uint8 = np.clip(saturation_float, 0, 255).astype(np.uint8) # Saturation range: 0-255
    value_uint8 = np.clip(value_float, 0, 255).astype(np.uint8) # Value range: 0-255

    hsv_image = np.stack([hue_uint8, np.full_like(hue_uint8, saturation_uint8), value_uint8], axis=-1)
    
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    frame_img[:] = bgr_image

    return frame_img

# --- UI per Caricamento Audio e Controlli (Spostati nella Sidebar) ---

uploaded_audio = st.file_uploader("Carica un file audio (MP3, WAV)", type=["mp3", "wav"])

st.sidebar.subheader("Impostazioni Generazione Visual")
num_frames_per_second = st.sidebar.slider("Frame al secondo (FPS)", 15, 60, 24)
visual_complexity = st.sidebar.slider("Complessità visiva", 1, 10, 5)
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
        
        # Carica l'intero file audio per l'analisi di BPM e frequenze
        y, sr = sf.read(audio_path)

        # --- Pre-calcolo BPM e Beat ---
        # librosa.beat.beat_track può richiedere un po' di tempo per file lunghi
        st.info("Analisi del battito (BPM)...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=100, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        st.write(f"BPM stimato: **{tempo:.2f}**")
        st.write(f"Trovati **{len(beat_times)}** battiti.")

        samples_per_frame = int(sr / num_frames_per_second)
        
        # Frequenze per le bande (esempi, possono essere ottimizzate)
        # Queste sono le frequenze di taglio, relative alla frequenza di campionamento
        bass_hz_end = 250
        mid_hz_end = 4000
        # Treble va da mid_hz_end fino alla frequenza di Nyquist (sr/2)

        for i in range(total_frames):
            start_sample = i * samples_per_frame
            end_sample = min((i + 1) * samples_per_frame, len(y))
            
            if start_sample >= len(y):
                break

            audio_chunk = y[start_sample:end_sample]

            # --- Calcolo RMS (Volume) ---
            rms_energy = np.sqrt(np.mean(audio_chunk**2))
            normalized_rms = np.log10(rms_energy + 1e-10) / np.log10(1.0 + 1e-10) 
            normalized_rms = np.clip(normalized_rms, 0, 1)

            # --- Rilevamento Beat nel Frame Corrente ---
            current_frame_time = i / num_frames_per_second
            next_frame_time = (i + 1) / num_frames_per_second
            is_beat_in_frame = False
            # Controlla se c'è un battito nell'intervallo di tempo del frame
            for bt in beat_times:
                if current_frame_time <= bt < next_frame_time:
                    is_beat_in_frame = True
                    break

            # --- Analisi Frequenze (Bassi, Medi, Alti) ---
            # Esegui la trasformata di Fourier sull'audio chunk
            # Utilizziamo np.fft.fft per semplicità su piccoli chunk
            if len(audio_chunk) > 0:
                fft_output = np.fft.fft(audio_chunk)
                # Calcola le ampiezze (magnitudine) e prendi solo la parte positiva (simmetrica)
                magnitude = np.abs(fft_output[:len(audio_chunk) // 2])
                
                # Calcola le frequenze corrispondenti
                frequencies = np.linspace(0, sr / 2, len(magnitude))

                # Indici per le bande di frequenza
                bass_idx = np.where(frequencies < bass_hz_end)
                mid_idx = np.where((frequencies >= bass_hz_end) & (frequencies < mid_hz_end))
                treble_idx = np.where(frequencies >= mid_hz_end)

                # Calcola l'energia (media delle magnitudini) per ogni banda
                bass_energy = np.mean(magnitude[bass_idx]) if len(bass_idx[0]) > 0 else 0
                mid_energy = np.mean(magnitude[mid_idx]) if len(mid_idx[0]) > 0 else 0
                treble_energy = np.mean(magnitude[treble_idx]) if len(treble_idx[0]) > 0 else 0
                
                # Normalizza l'energia delle bande a un intervallo 0-1. 
                # Questi valori 'max' sono empirici e possono essere regolati.
                max_mag = np.max(magnitude) if len(magnitude) > 0 else 1
                bass_energy = np.clip(bass_energy / (max_mag + 1e-10) * 2, 0, 1) # Moltiplica per aumentare sensibilità se necessario
                mid_energy = np.clip(mid_energy / (max_mag + 1e-10) * 2, 0, 1)
                treble_energy = np.clip(treble_energy / (max_mag + 1e-10) * 2, 0, 1)
            else:
                bass_energy, mid_energy, treble_energy = 0, 0, 0


            frame_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            if pattern_type == "Mandala Semplice (Cerchi)":
                frame_img = draw_simple_mandala(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, line_thickness, i, total_frames, is_beat_in_frame, bass_energy, mid_energy, treble_energy)
            elif pattern_type == "Flusso Ondulato a Gradiente":
                frame_img = draw_fluid_gradient_pattern(frame_img, frame_width, frame_height, normalized_rms, visual_complexity, i, total_frames, is_beat_in_frame, bass_energy, mid_energy, treble_energy)
            
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
        elif "Cannot install on Python version 3.13.5; only versions >=3.6,<3.10 are supported." in str(e):
             st.error("Si è verificato un problema con la versione di Python richiesta da alcune librerie (probabilmente Numba/llvmlite, dipendenze di librosa).")
             st.info("Per favore, verifica il file `requirements.txt` e `packages.txt` nel tuo repository per assicurarti che librosa sia compatibile con l'ambiente Streamlit (che potrebbe usare Python 3.13.5). Potrebbe essere necessario specificare versioni più vecchie di librosa/numba/llvmlite se la versione corrente non supporta Python 3.13.")
             st.info("Una soluzione rapida potrebbe essere aggiungere `python_version = \"3.9\"` nel tuo `.streamlit/config.toml` per forzare una versione Python più vecchia se possibile, oppure bloccare le versioni di `numba` e `llvmlite` in `requirements.txt` a quelle note per funzionare con 3.13.5, se esistono.")


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
