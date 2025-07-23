import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import time
import os
import subprocess
from numba import jit
import random

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
        return 0.0, 0.0, 0.0

    # Normalizza i dati di frequenza per evitare valori eccessivi
    max_val = np.max(freq_data)
    if max_val > 1e-6:
        freq_data_norm = freq_data / max_val
    else:
        return 0.0, 0.0, 0.0

    total_bins = len(freq_data_norm)
    # Aumentiamo la sensibilità per le bande
    low_end = max(1, int(total_bins * 0.25)) # Fino a ~2.7kHz per 11025/2
    mid_end = max(low_end + 1, int(total_bins * 0.6)) # Fino a ~6.6kHz
    
    low_freq = np.mean(freq_data_norm[:low_end]) if low_end > 0 else 0.0
    mid_freq = np.mean(freq_data_norm[low_end:mid_end]) if mid_end > low_end else 0.0
    high_freq = np.mean(freq_data_norm[mid_end:]) if total_bins > mid_end else 0.0

    return float(low_freq), float(mid_freq), float(high_freq)

def process_frame_data(audio_chunk):
    if len(audio_chunk) == 0:
        return 0.0, np.array([])
    
    rms = np.sqrt(np.mean(audio_chunk ** 2))

    # Applica una finestra per FFT per ridurre artefatti
    if len(audio_chunk) > 1:
        windowed_audio_chunk = audio_chunk * np.hanning(len(audio_chunk))
        freq_data = np.abs(np.fft.rfft(windowed_audio_chunk))
    else:
        freq_data = np.array([])

    return float(rms), freq_data

def hex_to_bgr(hex_color):
    """Converte colore hex in formato BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 0)
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # BGR format
    except ValueError:
        return (0, 0, 0)

# --- FUNZIONI DI SUPPORTO PER SINCRONIZZAZIONE BPM ---

def calculate_bpm_phase(current_time, tempo, sync_type, beat_times, bmp_settings):
    """Calcola la fase corrente basata sul BPM per sincronizzazione precisa"""
    if not bmp_settings['enabled'] or len(beat_times) == 0 or tempo <= 0:
        return 0.0, False, 1.0
    
    # Trova il beat più vicino
    beat_distances = np.abs(beat_times - current_time)
    nearest_beat_idx = np.argmin(beat_distances)
    nearest_beat_time = beat_times[nearest_beat_idx]
    
    # Calcola la distanza temporale dal beat più vicino
    time_from_beat = current_time - nearest_beat_time
    
    # Determina se siamo "su un beat"
    beat_window = 60.0 / (tempo * 8)
    is_on_beat = abs(time_from_beat) <= beat_window
    
    # Calcola la fase in base al tipo di sincronizzazione
    beat_duration = 60.0 / tempo
    
    sync_multipliers = {
        "Beat principale (1/1)": 1.0,
        "Mezzo beat (1/2)": 2.0,
        "Doppio beat (2/1)": 0.5,
        "Terzine (1/3)": 3.0
    }
    
    multiplier = sync_multipliers.get(sync_type, 1.0)
    phase_duration = beat_duration / multiplier
    
    # Calcola la fase (0 a 2π nel periodo del beat)
    if phase_duration > 0:
        time_in_cycle = (current_time % phase_duration)
        phase = (time_in_cycle / phase_duration) * 2 * np.pi
    else:
        phase = 0.0
    
    # Calcola l'intensità del beat
    if beat_window > 0:
        beat_intensity = max(0.1, 1.0 - (abs(time_from_beat) / beat_window)) if is_on_beat else 0.1
    else:
        beat_intensity = 0.1
    
    return float(phase), is_on_beat, float(beat_intensity)

def apply_bpm_movement_modulation(base_value, phase, beat_intensity, modulation_type, bmp_settings):
    """Applica modulazione basata sui BPM a un valore base"""
    if not bmp_settings['enabled']:
        return float(base_value)
    
    intensity = bmp_settings['beat_response_intensity']
    
    modulation = 0.0
    if modulation_type == 'sine':
        modulation = np.sin(phase) * intensity * beat_intensity
    elif modulation_type == 'cosine':
        modulation = np.cos(phase) * intensity * beat_intensity
    elif modulation_type == 'pulse':
        modulation = (np.sin(phase) ** 2) * intensity * beat_intensity
    elif modulation_type == 'sawtooth':
        normalized_phase = (phase / (2 * np.pi)) % 1
        modulation = (normalized_phase * 2 - 1) * intensity * beat_intensity
    
    if bmp_settings['smooth_transitions']:
        smooth_factor = 0.5
        modulation *= smooth_factor
    
    return float(base_value + modulation)

# --- FUNZIONE PER PATTERN GEOMETRICO ---
def draw_geometric_pattern_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings, selected_pattern_mode, line_thickness_control, particles_settings):
    """
    Genera un pattern geometrico reattivo all'audio e ai BPM con trasformazioni significative.
    Ora seleziona il pattern in base a `selected_pattern_mode`.
    """
    low_freq, mid_freq, high_freq = analyze_frequency_bands(freq_data)
    phase, is_on_beat, beat_intensity = calculate_bpm_phase(current_time, tempo, bmp_settings['movement_sync_type'], beat_times, bmp_settings)

    effective_rms = rms * movement_scale_factor
    effective_low_freq = low_freq * movement_scale_factor
    effective_mid_freq = mid_freq * movement_scale_factor
    effective_high_freq = high_freq * movement_scale_factor

    pattern_mode = selected_pattern_mode
        
    # Dimensioni delle celle (per pattern 4 e 5)
    cell_size_base = 70 
    cell_size_mod_audio = effective_rms * 15 
    cell_size_mod_bpm = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bmp_settings) * 15 if bmp_settings['enabled'] else 0
    current_cell_size = int(cell_size_base + cell_size_mod_audio + cell_size_mod_bpm)
    current_cell_size = max(20, min(120, current_cell_size))

    # Spessore del bordo (per pattern 4 e 5)
    border_thickness_base = 1
    border_thickness_mod = effective_rms * 2 + (beat_intensity * bmp_settings['beat_response_intensity'] * 3 if bmp_settings['enabled'] else 0)
    current_border_thickness = int(border_thickness_base + border_thickness_mod)
    current_border_thickness = max(1, min(15, current_border_thickness))


    # Logica per i colori reattivi alle frequenze
    base_fill_color_bgr = np.array(hex_to_bgr(color_settings['background_color']), dtype=np.float32)
    
    # Se i colori di frequenza sono abilitati, calcola un colore dinamico
    if color_settings['use_frequency_colors']:
        low_bgr = np.array(hex_to_bgr(color_settings['low_freq_color']), dtype=np.float32)
        mid_bgr = np.array(hex_to_bgr(color_settings['mid_freq_color']), dtype=np.float32)
        high_bgr = np.array(hex_to_bgr(color_settings['high_freq_color']), dtype=np.float32)

        # Applica l'intensità del colore per le particelle
        color_intensity_multiplier = 1.0
        if pattern_mode == 6: # Se sono particelle, usa il loro slider di intensità
            color_intensity_multiplier = particles_settings['color_intensity']

        # Mix i colori in base all'intensità delle frequenze e il nuovo multiplier
        mixed_dynamic_color = (
            low_bgr * effective_low_freq * 5.0 * color_intensity_multiplier +
            mid_bgr * effective_mid_freq * 5.0 * color_intensity_multiplier +
            high_bgr * effective_high_freq * 5.0 * color_intensity_multiplier
        )
        # Assicurati che il colore non superi 255 e abbia una base del colore di sfondo
        mixed_dynamic_color = np.clip(mixed_dynamic_color + base_fill_color_bgr * 0.2, 0, 255).astype(np.uint8) 
        effective_pattern_color = tuple(mixed_dynamic_color.tolist())
    else: # Usa i colori di base se non si usano i colori di frequenza
        effective_pattern_color = hex_to_bgr(color_settings['background_color']) 


    # --- Differenti modalità di pattern per le trasformazioni ---
    if pattern_mode == 4: # Effetto "Geometric Random Burst"
        for y in range(0, height, current_cell_size):
            for x in range(0, width, current_cell_size):
                x1, y1 = x, y
                x2, y2 = x + current_cell_size, y + current_cell_size
                
                # Effetti di movimento casuali
                offset_x = int(np.sin(current_time * 3 + x * 0.005) * effective_rms * 10)
                offset_y = int(np.cos(current_time * 3 + y * 0.005) * effective_rms * 10)

                # Probabilità di disegnare un elemento, aumenta con l'RMS e l'intensità del beat
                draw_probability = 0.05 + effective_rms * 0.2 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.3 if bmp_settings['enabled'] else 0)
                draw_probability = min(0.6, draw_probability) # Limita la probabilità massima

                if random.random() < draw_probability:
                    # Scegli casualmente il tipo di forma
                    shape_type = random.choice(['circle', 'rectangle', 'line', 'triangle'])
                    
                    # Colore randomizzato che deriva da effective_pattern_color
                    rand_color_mod = np.array([random.randint(-30, 30) for _ in range(3)])
                    rand_color = tuple(np.clip(np.array(effective_pattern_color) + rand_color_mod, 0, 255).tolist())

                    # Posizione casuale all'interno della cella (o anche fuori leggermente)
                    rand_x = x1 + random.randint(-current_cell_size // 4, current_cell_size)
                    rand_y = y1 + random.randint(-current_cell_size // 4, current_cell_size)
                    
                    # Dimensione casuale influenzata dall'RMS
                    rand_size = int(random.uniform(5, current_cell_size * 0.8) * (1 + effective_rms * 0.5))
                    rand_thickness = int(random.uniform(1, current_border_thickness * 2))
                    
                    if shape_type == 'circle':
                        cv2.circle(frame_img, (rand_x, rand_y), rand_size // 2, rand_color, rand_thickness)
                    elif shape_type == 'rectangle':
                        cv2.rectangle(frame_img, (rand_x, rand_y), (rand_x + rand_size, rand_y + rand_size), rand_color, rand_thickness)
                    elif shape_type == 'line':
                        x_end = rand_x + int(random.uniform(-rand_size, rand_size))
                        y_end = rand_y + int(random.uniform(-rand_size, rand_size))
                        cv2.line(frame_img, (rand_x, rand_y), (x_end, y_end), rand_color, rand_thickness)
                    elif shape_type == 'triangle':
                        p1 = (rand_x, rand_y)
                        p2 = (rand_x + int(rand_size * random.uniform(0.5, 1.5)), rand_y + int(rand_size * random.uniform(-0.5, 0.5)))
                        p3 = (rand_x + int(rand_size * random.uniform(-0.5, 0.5)), rand_y + int(rand_size * random.uniform(0.5, 1.5)))
                        pts = np.array([p1, p2, p3], np.int32)
                        cv2.polylines(frame_img, [pts.reshape((-1, 1, 2))], True, rand_color, rand_thickness)
                        # Aggiungiamo anche il riempimento per alcuni triangoli
                        if random.random() < 0.5:
                             cv2.fillPoly(frame_img, [pts.reshape((-1, 1, 2))], rand_color)
            
    elif pattern_mode == 5: # Effetto "Linee Scomposte"
        # Applica il controllo dello spessore delle linee dal nuovo slider qui specificamente
        current_line_thickness_glitch = int(border_thickness_base + border_thickness_mod + line_thickness_control)
        current_line_thickness_glitch = max(1, min(15, current_line_thickness_glitch)) 

        for y in range(0, height, current_cell_size):
            for x in range(0, width, current_cell_size):
                x1, y1 = x, y
                x2, y2 = x + current_cell_size, y + current_cell_size
                
                # Sfondo della cella: sempre il colore di sfondo scelto
                bg_for_lines = hex_to_bgr(color_settings['background_color'])
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), bg_for_lines, -1)

                # Colore delle linee: usa effective_pattern_color se freq_colors attivi, altrimenti bianco/nero
                if color_settings['use_frequency_colors']:
                    line_color_for_lines = effective_pattern_color
                else:
                    line_color_for_lines = (0, 0, 0) # Default: Linee nere
                    if np.mean(bg_for_lines) < 50: # Se il colore di sfondo è quasi nero
                        line_color_for_lines = (255, 255, 255) # Linee bianche
                
                # Intensità di "rottura" basata sulle alte frequenze e RMS
                break_intensity = np.clip(effective_high_freq * 4.0 + effective_rms * 2.0, 0.0, 1.0)
                
                # Numero di linee verticali all'interno della cella
                num_lines = max(2, int(current_cell_size / 15)) 
                
                for i in range(num_lines):
                    line_x = x1 + int(i * (current_cell_size / num_lines))
                    
                    # Se l'intensità di rottura è alta, spezziamo la linea
                    if random.random() < break_intensity * 0.7: # Probabilità di rottura
                        num_segments = max(2, int(break_intensity * 5)) # Più forte = più segmenti
                        segment_height = current_cell_size / num_segments
                        
                        for s in range(num_segments):
                            y_start_segment = y1 + int(s * segment_height)
                            y_end_segment = y1 + int((s + 1) * segment_height)
                            
                            # Aggiungi un offset casuale orizzontale per l'effetto "glitch"
                            glitch_offset = int(random.uniform(-5, 5) * break_intensity * 10)
                            
                            # Riduci la lunghezza del segmento casualmente per "spazi"
                            segment_shrink = random.uniform(0.1, 0.8) if random.random() < break_intensity else 1.0
                            y_start_segment += int((1 - segment_shrink) * segment_height / 2)
                            y_end_segment -= int((1 - segment_shrink) * segment_height / 2)

                            cv2.line(frame_img, (line_x + glitch_offset, y_start_segment), 
                                     (line_x + glitch_offset, y_end_segment), 
                                     line_color_for_lines, max(1, current_line_thickness_glitch // 2))
                    else:
                        # Linea intera (o leggermente spostata)
                        glitch_offset = int(random.uniform(-2, 2) * break_intensity * 5)
                        cv2.line(frame_img, (line_x + glitch_offset, y1), (line_x + glitch_offset, y2), 
                                 line_color_for_lines, current_line_thickness_glitch)

    elif pattern_mode == 6: # Nuovo Effetto "Particelle Reattive"
        num_particles_to_draw = particles_settings['quantity'] # Usa il valore dallo slider
        
        # Le particelle possono essere influenzate dal tempo per il loro movimento di base
        time_factor_x = np.sin(current_time * 0.5) * width * 0.1
        time_factor_y = np.cos(current_time * 0.5) * height * 0.1

        # Reattività BPM sul raggruppamento/dispersione
        bpm_dispersion = apply_bpm_movement_modulation(1.0, phase, beat_intensity, 'pulse', bmp_settings) * 0.5 # Aumenta la dispersione sul beat

        for i in range(num_particles_to_draw):
            # Posizione iniziale casuale per le particelle
            # Modulata da RMS e tempo per un movimento più interessante
            # Aggiunto un fattore casuale più ampio per il movimento di base
            x_pos = int(width * (0.5 + 0.4 * np.sin(i * 0.1 + current_time * 0.8 + effective_rms * 5.0 + random.uniform(-0.5, 0.5))))
            y_pos = int(height * (0.5 + 0.4 * np.cos(i * 0.15 + current_time * 0.7 + effective_rms * 5.0 + random.uniform(-0.5, 0.5))))
            
            # Applica una dispersione casuale e modulata dal BPM
            x_pos += int(random.uniform(-1, 1) * 70 * bpm_dispersion * (1 + effective_rms * 2)) # Aumentata la casualità
            y_pos += int(random.uniform(-1, 1) * 70 * bpm_dispersion * (1 + effective_rms * 2))

            # Assicurati che le particelle rimangano entro i bordi del frame
            x_pos = np.clip(x_pos, 0, width - 1)
            y_pos = np.clip(y_pos, 0, height - 1)

            # Dimensione delle particelle con un fattore casuale aggiuntivo
            particle_size = max(1, int(1 + effective_rms * 10 + beat_intensity * bmp_settings['beat_response_intensity'] * 5 + random.uniform(0, particles_settings['randomness_scale'])))
            
            # Colore delle particelle: usa effective_pattern_color se freq_colors attivi, altrimenti bianco
            particle_color = effective_pattern_color if color_settings['use_frequency_colors'] else (255, 255, 255)

            # Disegna la particella come un cerchio
            cv2.circle(frame_img, (x_pos, y_pos), particle_size, particle_color, -1) # -1 per riempire il cerchio
            
    # Alpha blending per questo layer (generalizzato per tutti i pattern)
    alpha = min(0.95, 0.7 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.4 if bmp_settings['enabled'] else 0)) 
    
    # Se è l'effetto particelle, potremmo usare un alpha leggermente inferiore per un look più etereo
    if pattern_mode == 6:
        alpha = min(0.95, 0.5 + effective_rms * 0.3 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.2 if bmp_settings['enabled'] else 0)) 
        
    geometric_pattern_layer = frame_img.copy() # Copia il frame con il pattern
    cv2.addWeighted(frame_img, 1-alpha, geometric_pattern_layer, alpha, 0, frame_img)

    return frame_img

# --- FUNZIONI DI MERGE VIDEO/AUDIO ---
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

# --- INTERFACCIA STREAMLIT MIGLIORATA ---

st.title("🌌 **SynestheticFlow**")
st.markdown("*<span style='font-size: 12px;'>Visualizer by loop507</span>*", unsafe_allow_html=True)

# File upload section (Moved to main area)
st.header("🎵 Carica brano per iniziare")
uploaded_file = st.file_uploader(
    "Scegli un file audio", 
    type=['mp3', 'wav', 'flac', 'm4a', 'ogg']
)

# Video format selection
st.sidebar.header("📺 Formato Video")
selected_format = st.sidebar.selectbox(
    "Scegli formato:",
    list(VIDEO_FORMATS.keys())
)
width, height = VIDEO_FORMATS[selected_format]

# Pattern type selection
st.sidebar.header("🌀 Tipo Visualizzazione")
# Mapping delle stringhe della selectbox ai numeri dei pattern
pattern_options = {
    "Geometric Random Burst": 4,
    "Linee Scomposte (Glitch)": 5,
    "Particelle Reattive": 6 
}
selected_pattern_name = st.sidebar.selectbox(
    "Scegli visualizzazione:",
    list(pattern_options.keys())
)
# Ottieni il numero del pattern selezionato
selected_pattern_mode = pattern_options[selected_pattern_name]


# Movement settings
st.sidebar.header("🎬 Movimento")
movement_scale = st.sidebar.slider(
    "Intensità movimento (generale)", 
    min_value=0.0, 
    max_value=3.0, 
    value=1.0, 
    step=0.1
)

# Controlli specifici per "Linee Scomposte"
if selected_pattern_mode == 5: 
    st.sidebar.subheader("Linee Scomposte - Controlli")
    line_thickness_control = st.sidebar.slider(
        "Spessore base linee",
        min_value=0,
        max_value=10, 
        value=0, 
        step=1
    )
else:
    line_thickness_control = 0 # Imposta a 0 se l'effetto non è Linee Scomposte

# Controlli specifici per "Particelle Reattive"
particles_settings = {
    'quantity': 1500, # Valore di default
    'color_intensity': 1.0, # Valore di default
    'randomness_scale': 0 # Valore di default
}
if selected_pattern_mode == 6:
    st.sidebar.subheader("Particelle Reattive - Controlli")
    particles_settings['quantity'] = st.sidebar.slider(
        "Quantità Particelle",
        min_value=100,
        max_value=5000, # Puoi aumentare questo per schermi pieni
        value=1500, 
        step=100
    )
    particles_settings['color_intensity'] = st.sidebar.slider(
        "Intensità Colore Particelle",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    particles_settings['randomness_scale'] = st.sidebar.slider(
        "Scala casualità movimento/dimensione",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


# BPM Sync Settings
st.sidebar.header("🎵 Sincronizzazione BPM")
bmp_settings = {
    'enabled': st.sidebar.checkbox("Abilita sync BPM", value=True),
    'movement_sync_type': st.sidebar.selectbox(
        "Tipo sincronizzazione:",
        ["Beat principale (1/1)", "Mezzo beat (1/2)", "Doppio beat (2/1)", "Terzine (1/3)"]
    ),
    'beat_response_intensity': st.sidebar.slider(
        "Intensità risposta beat", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.8, 
        step=0.1
    ),
    'smooth_transitions': st.sidebar.checkbox("Transizioni smooth", value=True)
}

# Color settings
st.sidebar.header("🎨 Colori")
color_settings = {
    'use_frequency_colors': st.sidebar.checkbox("Usa colori frequenza", value=True),
    'background_color': st.sidebar.color_picker("Colore sfondo", "#000000"),
    'low_freq_color': st.sidebar.color_picker("Frequenze basse", "#FF0000"),
    'mid_freq_color': st.sidebar.color_picker("Frequenze medie", "#00FF00"),
    'high_freq_color': st.sidebar.color_picker("Frequenze acute", "#0000FF")
}
st.sidebar.markdown("<small>*I colori delle frequenze influenzeranno la colorazione dinamica dei pattern e delle particelle. Per 'Linee Scomposte', se i colori di frequenza sono disabilitati, le linee saranno bianco/nero per contrasto.*</small>", unsafe_allow_html=True)


# Processing section
if uploaded_file is not None:
    st.success(f"File caricato: {uploaded_file.name}")
    
    # Show video settings info
    est_width, est_height, est_fps, est_size = get_optimal_settings(60, width, height)  # assuming 60s duration
    st.info(f"Impostazioni video: {width}x{height} @ {est_fps}fps (stima: ~{est_size}MB)")
    
    # Process button
    if st.button("🚀 Genera Video", type="primary"):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Prepare audio
                status_text.text("Preparazione audio...")
                audio_path = prepare_audio_file(uploaded_file, temp_dir)
                progress_bar.progress(10)
                
                # Step 2: Analyze audio
                status_text.text("Analisi audio...")
                y, beat_times, tempo, sr = analyze_audio_minimal(audio_path)
                duration = len(y) / sr
                progress_bar.progress(20)
                
                # Step 3: Setup video
                status_text.text("Inizializzazione video...")
                fps = 20
                total_frames = int(duration * fps)
                
                # Video writer setup
                temp_video_path = f"{temp_dir}/temp_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                
                progress_bar.progress(30)
                
                # Step 4: Generate frames
                status_text.text("Generazione frames...")
                
                for frame_idx in range(total_frames):
                    current_time = frame_idx / fps
                    
                    # Calculate audio data for current frame
                    start_sample = int(current_time * sr)
                    end_sample = int((current_time + 1/fps) * sr)
                    audio_chunk = y[start_sample:end_sample] if start_sample < len(y) else np.array([])
                    
                    # Process audio data
                    rms, freq_data = process_frame_data(audio_chunk)
                    
                    # Create base frame
                    frame_img = np.full((height, width, 3), hex_to_bgr(color_settings['background_color']), dtype=np.uint8)
                    
                    # Apply geometric pattern
                    frame_img = draw_geometric_pattern_bpm_sync(
                        frame_img, width, height, rms, current_time, beat_times, 
                        tempo, freq_data, color_settings, movement_scale, bmp_settings, 
                        selected_pattern_mode, line_thickness_control, particles_settings 
                    )
                    
                    # Write frame
                    out.write(frame_img)
                    
                    # Update progress
                    progress = 30 + int((frame_idx / total_frames) * 60)
                    progress_bar.progress(progress)
                    
                    if frame_idx % 20 == 0:  # Update status every 20 frames
                        status_text.text(f"Frame {frame_idx}/{total_frames} ({current_time:.1f}s)")
                
                # Release video writer
                out.release()
                progress_bar.progress(90)
                
                # Step 5: Merge audio and video
                status_text.text("Merge audio/video...")
                final_output_path = f"{temp_dir}/synesthetic_output.mp4"
                success, message = merge_video_audio(temp_video_path, audio_path, final_output_path)
                
                if success:
                    progress_bar.progress(100)
                    status_text.text("✅ Video completato!")
                    
                    # Read the final video file
                    with open(final_output_path, 'rb') as f:
                        video_data = f.read()
                    
                    # Provide download
                    st.download_button(
                        label="📥 Scarica Video",
                        data=video_data,
                        file_name=f"synesthetic_{selected_pattern_name.lower().replace(' ', '_').replace('(','').replace(')','')}_{int(time.time())}.mp4", 
                        mime="video/mp4"
                    )
                    
                    # Show video preview
                    st.video(final_output_path)
                    
                else:
                    st.error(f"Errore nel merge: {message}")
                    
            except Exception as e:
                st.error(f"Errore durante la generazione: {str(e)}")
                
else:
    st.info("👆 Carica un file audio per iniziare")

# Footer info
st.markdown("---")
st.markdown("""
### 📖 Come usare:
1. **Carica** un file audio (MP3, WAV, etc.)
2. **Imposta** formato video
3. **Scegli** la visualizzazione tra "Geometric Random Burst", "Linee Scomposte (Glitch)" e "Particelle Reattive".
4. **Personalizza** intensità movimento, sincronizzazione BPM, **controlli specifici per ogni effetto** e colori.
5. **Genera** il tuo video artistico!

### 🎵 Caratteristiche BPM:
- **Sincronizzazione automatica** sul tempo del brano
- **Modulazione dinamica** di zoom, movimento e colori
- **Diversi tipi di sync**: beat principale, mezzi beat, terzine
- **Transizioni smooth** per effetti fluidi

### 🌀 Visualizzazioni disponibili:
- **Geometric Random Burst**: Un'esplosione dinamica di forme geometriche casuali che reagiscono all'audio e ai colori delle frequenze.
- **Linee Scomposte (Glitch)**: Linee verticali che si "rompono" e glitchano in base all'audio, ora con colori reattivi alle frequenze (o bianco/nero per contrasto).
- **Particelle Reattive**: Una nuvola di particelle dinamiche che si muovono, pulsano e cambiano colore in base al volume e alle frequenze dell'audio, creando un'esperienza fluida e organica. Ora con **controllo su quantità, intensità del colore e casualità del movimento!**
""")
