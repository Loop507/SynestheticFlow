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
def draw_geometric_pattern_bpm_sync(frame_img, width, height, rms, current_time, beat_times, tempo, freq_data, color_settings, movement_scale_factor, bmp_settings, selected_pattern_mode, glitch_settings, particles_settings, burst_settings, waves_settings):
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
        
    # Dimensioni delle celle (per pattern 4, 5)
    cell_size_base = 70 
    cell_size_mod_audio = effective_rms * 15 
    cell_size_mod_bpm = apply_bpm_movement_modulation(0, phase, beat_intensity, 'pulse', bmp_settings) * 15 if bmp_settings['enabled'] else 0
    current_cell_size = int(cell_size_base + cell_size_mod_audio + cell_size_mod_bpm)
    current_cell_size = max(20, min(120, current_cell_size))

    # Spessore del bordo (per pattern 4, 5)
    border_thickness_base = 1
    border_thickness_mod = effective_rms * 2 + (beat_intensity * bmp_settings['beat_response_intensity'] * 3 if bmp_settings['enabled'] else 0)
    current_border_thickness = int(border_thickness_base + border_thickness_mod)
    current_border_thickness = max(1, min(15, current_border_thickness))


    # Pre-calcola i colori base per le frequenze per la selezione per-elemento
    base_low_bgr = np.array(hex_to_bgr(color_settings['low_freq_color']), dtype=np.float32)
    base_mid_bgr = np.array(hex_to_bgr(color_settings['mid_freq_color']), dtype=np.float32)
    base_high_bgr = np.array(hex_to_bgr(color_settings['high_freq_color']), dtype=np.float32)

    # Funzione helper per ottenere il colore di un singolo elemento in base alla dominanza delle frequenze
    def get_dynamic_element_color():
        if color_settings['use_frequency_colors']:
            low_weight = effective_low_freq * 5.0 
            mid_weight = effective_mid_freq * 5.0 
            high_weight = effective_high_freq * 5.0 
            total_weight = low_weight + mid_weight + high_weight

            if total_weight > 0:
                prob_low = low_weight / total_weight
                prob_mid = mid_weight / total_weight

                rand_choice = random.uniform(0, 1)
                
                selected_base_color = None
                if rand_choice < prob_low:
                    selected_base_color = base_low_bgr
                elif rand_choice < prob_low + prob_mid:
                    selected_base_color = base_mid_bgr
                else:
                    selected_base_color = base_high_bgr
                
                if selected_base_color is not None:
                    color_mod_factor = color_settings['element_color_intensity'] * (1.0 + random.uniform(-0.2, 0.2))
                    return tuple(np.clip(selected_base_color * color_mod_factor, 0, 255).astype(np.uint8).tolist())
        return (255, 255, 255) # Default white if no frequency colors or no sound

    # --- Differenti modalità di pattern per le trasformazioni ---
    if pattern_mode == 4: # Effetto "Geometric Random Burst"
        # Probabilità di disegnare un elemento, basata sul nuovo slider "Quantità Figure Burst"
        # Convertiamo la scala 0-100 in una probabilità 0.0-0.7 (massimo)
        draw_probability_user = burst_settings['figure_quantity'] / 100.0 * 0.7 
        draw_probability_audio = effective_rms * 0.2 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.3 if bmp_settings['enabled'] else 0)
        final_draw_probability = min(0.7, draw_probability_user + draw_probability_audio) # Limita la probabilità massima

        selected_figure_type = burst_settings['figure_type']
        
        for y in range(0, height, current_cell_size):
            for x in range(0, width, current_cell_size):
                x1, y1 = x, y
                x2, y2 = x + current_cell_size, y + current_cell_size
                
                # Effetti di movimento casuali
                offset_x = int(np.sin(current_time * 3 + x * 0.005) * effective_rms * 10)
                offset_y = int(np.cos(current_time * 3 + y * 0.005) * effective_rms * 10)

                if random.random() < final_draw_probability:
                    # Scegli il tipo di forma
                    shape_type = selected_figure_type
                    if selected_figure_type == 'Casuale':
                        shape_type = random.choice(['Cerchio', 'Rettangolo', 'Linea', 'Triangolo']) # Aggiornato nomi

                    current_element_color = get_dynamic_element_color() if color_settings['use_frequency_colors'] else hex_to_bgr(color_settings['background_color']) 
                    if not color_settings['use_frequency_colors']: # Se i colori freq non sono usati, randomizza il colore di sfondo leggermente
                        rand_color_mod = np.array([random.randint(-30, 30) for _ in range(3)])
                        current_element_color = tuple(np.clip(np.array(current_element_color) + rand_color_mod, 0, 255).tolist())


                    # Posizione casuale all'interno della cella (o anche fuori leggermente)
                    rand_x = x1 + random.randint(-current_cell_size // 4, current_cell_size)
                    rand_y = y1 + random.randint(-current_cell_size // 4, current_cell_size)
                    
                    # Dimensione casuale influenzata dall'RMS
                    rand_size = int(random.uniform(5, current_cell_size * 0.8) * (1 + effective_rms * 0.5))
                    rand_thickness = int(random.uniform(1, current_border_thickness * 2))
                    
                    if shape_type == 'Cerchio':
                        cv2.circle(frame_img, (rand_x, rand_y), rand_size // 2, current_element_color, rand_thickness)
                    elif shape_type == 'Rettangolo':
                        cv2.rectangle(frame_img, (rand_x, rand_y), (rand_x + rand_size, rand_y + rand_size), current_element_color, rand_thickness)
                    elif shape_type == 'Linea':
                        x_end = rand_x + int(random.uniform(-rand_size, rand_size))
                        y_end = rand_y + int(random.uniform(-rand_size, rand_size))
                        cv2.line(frame_img, (rand_x, rand_y), (x_end, y_end), current_element_color, rand_thickness)
                    elif shape_type == 'Triangolo':
                        p1 = (rand_x, rand_y)
                        p2 = (rand_x + int(rand_size * random.uniform(0.5, 1.5)), rand_y + int(rand_size * random.uniform(-0.5, 0.5)))
                        p3 = (rand_x + int(rand_size * random.uniform(-0.5, 0.5)), rand_y + int(rand_size * random.uniform(0.5, 1.5)))
                        pts = np.array([p1, p2, p3], np.int32)
                        cv2.polylines(frame_img, [pts.reshape((-1, 1, 2))], True, current_element_color, rand_thickness)
                        # Aggiungiamo anche il riempimento per alcuni triangoli
                        if random.random() < 0.5:
                             cv2.fillPoly(frame_img, [pts.reshape((-1, 1, 2))], current_element_color)
            
    elif pattern_mode == 5: # Effetto "Linee Scomposte"
        current_line_thickness_glitch = int(border_thickness_base + border_thickness_mod + glitch_settings['line_thickness'])
        current_line_thickness_glitch = max(1, min(15, current_line_thickness_glitch)) 

        line_orientation = glitch_settings['orientation']

        for y_cell in range(0, height, current_cell_size):
            for x_cell in range(0, width, current_cell_size):
                x1_cell, y1_cell = x_cell, y_cell
                x2_cell, y2_cell = x_cell + current_cell_size, y_cell + current_cell_size
                
                # Sfondo della cella: sempre il colore di sfondo scelto
                bg_for_lines = hex_to_bgr(color_settings['background_color'])
                cv2.rectangle(frame_img, (x1_cell, y1_cell), (x2_cell, y2_cell), bg_for_lines, -1)

                # Intensità di "rottura" basata sulle alte frequenze e RMS
                break_intensity = np.clip(effective_high_freq * 4.0 + effective_rms * 2.0, 0.0, 1.0)
                
                # Scegli l'orientamento per questa cella se è "Entrambi"
                current_orientation_choice = line_orientation
                if line_orientation == 'Entrambi':
                    current_orientation_choice = random.choice(['Verticale', 'Orizzontale'])

                if current_orientation_choice == 'Verticale' or current_orientation_choice == 'Entrambi':
                    # Numero di linee verticali all'interno della cella
                    num_lines_vertical = max(2, int(current_cell_size / 15)) 
                    
                    for i in range(num_lines_vertical):
                        line_x = x1_cell + int(i * (current_cell_size / num_lines_vertical))
                        
                        line_color_for_lines = get_dynamic_element_color() if color_settings['use_frequency_colors'] else ( (0, 0, 0) if np.mean(bg_for_lines) >= 50 else (255, 255, 255) )
                        
                        # Se l'intensità di rottura è alta, spezziamo la linea
                        if random.random() < break_intensity * 0.7: # Probabilità di rottura
                            num_segments = max(2, int(break_intensity * 5)) # Più forte = più segmenti
                            segment_height = current_cell_size / num_segments
                            
                            for s in range(num_segments):
                                y_start_segment = y1_cell + int(s * segment_height)
                                y_end_segment = y1_cell + int((s + 1) * segment_height)
                                
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
                            cv2.line(frame_img, (line_x + glitch_offset, y1_cell), (line_x + glitch_offset, y2_cell), 
                                     line_color_for_lines, current_line_thickness_glitch)
                
                if current_orientation_choice == 'Orizzontale' or current_orientation_choice == 'Entrambi':
                    # Numero di linee orizzontali all'interno della cella
                    num_lines_horizontal = max(2, int(current_cell_size / 15))

                    for i in range(num_lines_horizontal):
                        line_y = y1_cell + int(i * (current_cell_size / num_lines_horizontal))

                        line_color_for_lines = get_dynamic_element_color() if color_settings['use_frequency_colors'] else ( (0, 0, 0) if np.mean(bg_for_lines) >= 50 else (255, 255, 255) )

                        if random.random() < break_intensity * 0.7:
                            num_segments = max(2, int(break_intensity * 5))
                            segment_width = current_cell_size / num_segments

                            for s in range(num_segments):
                                x_start_segment = x1_cell + int(s * segment_width)
                                x_end_segment = x1_cell + int((s + 1) * segment_width)

                                glitch_offset = int(random.uniform(-5, 5) * break_intensity * 10)

                                segment_shrink = random.uniform(0.1, 0.8) if random.random() < break_intensity else 1.0
                                x_start_segment += int((1 - segment_shrink) * segment_width / 2)
                                x_end_segment -= int((1 - segment_shrink) * segment_width / 2)
                                
                                cv2.line(frame_img, (x_start_segment, line_y + glitch_offset),
                                         (x_end_segment, line_y + glitch_offset),
                                         line_color_for_lines, max(1, current_line_thickness_glitch // 2))
                        else:
                            glitch_offset = int(random.uniform(-2, 2) * break_intensity * 5)
                            cv2.line(frame_img, (x1_cell, line_y + glitch_offset), (x2_cell, line_y + glitch_offset),
                                     line_color_for_lines, current_line_thickness_glitch)


    elif pattern_mode == 6: # Effetto "Particelle Reattive"
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
            
            # Colore delle particelle: Logica per colori individuali basati su frequenza
            particle_color = get_dynamic_element_color() if color_settings['use_frequency_colors'] else (255, 255, 255) # Default white

            # Disegna la particella come un cerchio
            cv2.circle(frame_img, (x_pos, y_pos), particle_size, particle_color, -1) # -1 per riempire il cerchio

    
    elif pattern_mode == 9: # Nuovo effetto "Onde Astratte"
        num_waves = waves_settings['quantity']
        wave_amplitude_base = waves_settings['amplitude_scale'] * 20 # Scala base dell'ampiezza
        wave_frequency_base = waves_settings['frequency_scale'] * 0.005 # Scala base della frequenza spaziale
        wave_speed_base = waves_settings['speed'] * 0.5 # Scala base della velocità temporale
        line_thickness_waves = max(1, waves_settings['line_thickness'])

        # Modulazioni da audio e BPM
        modulated_amplitude = wave_amplitude_base * (1 + effective_rms * 3 + (beat_intensity * bmp_settings['beat_response_intensity'] * 5 if bmp_settings['enabled'] else 0))
        modulated_frequency = wave_frequency_base * (1 + effective_high_freq * 2)
        modulated_speed = wave_speed_base * (1 + effective_mid_freq * 1) # Modula la velocità anche con le medie frequenze

        # Colori per la gradazione
        color_start = base_low_bgr # Usiamo i colori delle basse frequenze come inizio della gradazione
        color_end = base_high_bgr # Usiamo i colori delle alte frequenze come fine della gradazione

        # Se i colori delle frequenze non sono usati, usa un gradiente fisso (blu-rosa come nell'esempio)
        if not color_settings['use_frequency_colors']:
            color_start = np.array([255, 0, 255], dtype=np.float32) # Magenta (BGR)
            color_end = np.array([255, 255, 0], dtype=np.float32)   # Ciano (BGR)
            # Potremmo anche aggiungere controlli per questi colori fissi, se l'utente vuole personalizzare.

        for i in range(num_waves):
            points = []
            # Calcola la posizione Y base della linea
            # Distribuisci le linee uniformemente lungo l'altezza
            base_y_position = int(height * (i / (num_waves - 1))) if num_waves > 1 else height // 2

            # Offset verticale per creare un effetto "spostamento" generale
            global_offset_y = int(np.sin(current_time * modulated_speed + i * 0.1) * modulated_amplitude * 0.5)

            for x in range(0, width + 1, 5): # Disegna punti ogni 5 pixel per una linea fluida
                # Calcola l'offset Y per il punto corrente basato su una funzione sinusoidale
                # La fase dell'onda dipende dalla posizione X e dal tempo
                wave_y_offset = np.sin(x * modulated_frequency + current_time * modulated_speed) * modulated_amplitude
                
                # Applica un leggero offset casuale per un aspetto più "organico" o "vettoriale" granuloso
                random_offset = random.uniform(-waves_settings['randomness_spread'], waves_settings['randomness_spread'])

                y_point = int(base_y_position + wave_y_offset + global_offset_y + random_offset)
                
                # Clampa i punti entro i limiti del frame
                y_point = np.clip(y_point, 0, height - 1)
                
                points.append((x, y_point))
            
            # Colore della linea: interpolazione tra i colori start e end
            # `t` è un fattore di interpolazione da 0 a 1 basato sull'indice della linea
            t_color = i / (num_waves - 1) if num_waves > 1 else 0.5
            line_color_bgr = tuple(np.clip(color_start * (1 - t_color) + color_end * t_color, 0, 255).astype(np.uint8).tolist())

            # Disegna la polilinea
            if len(points) > 1:
                cv2.polylines(frame_img, [np.array(points, np.int32).reshape((-1, 1, 2))], False, line_color_bgr, line_thickness_waves, cv2.LINE_AA)
            
    # Alpha blending per questo layer (generalizzato per tutti i pattern)
    alpha = min(0.95, 0.7 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.4 if bmp_settings['enabled'] else 0)) 
    
    # Se è l'effetto particelle, potremmo usare un alpha leggermente inferiore per un look più etereo
    if pattern_mode == 6:
        alpha = min(0.95, 0.5 + effective_rms * 0.3 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.2 if bmp_settings['enabled'] else 0)) 
    elif pattern_mode == 9: # Alpha per le onde astratte, può essere leggermente più denso
        alpha = min(0.98, 0.8 + effective_rms * 0.15 + (beat_intensity * bmp_settings['beat_response_intensity'] * 0.25 if bmp_settings['enabled'] else 0))


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
    "Particelle Reattive": 6,
    "Onde Astratte": 9 # Nuovo effetto
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
glitch_settings = {
    'line_thickness': 0,
    'orientation': 'Verticale' # Default
}
if selected_pattern_mode == 5: 
    st.sidebar.subheader("Linee Scomposte - Controlli")
    glitch_settings['line_thickness'] = st.sidebar.slider(
        "Spessore base linee",
        min_value=0,
        max_value=10, 
        value=0, 
        step=1
    )
    glitch_settings['orientation'] = st.sidebar.selectbox(
        "Orientamento Linee",
        ['Verticale', 'Orizzontale', 'Entrambi']
    )


# Controlli specifici per "Particelle Reattive"
particles_settings = {
    'quantity': 1500, # Valore di default
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
    particles_settings['randomness_scale'] = st.sidebar.slider(
        "Scala casualità movimento/dimensione",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

# Controlli specifici per "Geometric Random Burst"
burst_settings = {
    'figure_quantity': 50, # Default per la probabilità
    'figure_type': 'Casuale' # Default per il tipo di figura
}
if selected_pattern_mode == 4:
    st.sidebar.subheader("Geometric Random Burst - Controlli")
    burst_settings['figure_quantity'] = st.sidebar.slider(
        "Quantità Figure Burst",
        min_value=1,
        max_value=100, # Rappresenta una percentuale della massima probabilità di disegno
        value=50,
        step=1
    )
    burst_settings['figure_type'] = st.sidebar.selectbox(
        "Tipo Figura Burst",
        ['Casuale', 'Cerchio', 'Rettangolo', 'Linea', 'Triangolo']
    )


# Controlli specifici per "Onde Astratte"
waves_settings = {
    'quantity': 30,
    'amplitude_scale': 1.0,
    'frequency_scale': 1.0,
    'speed': 1.0,
    'line_thickness': 2,
    'randomness_spread': 0 # Nuovo controllo per la casualità
}
if selected_pattern_mode == 9:
    st.sidebar.subheader("Onde Astratte - Controlli")
    waves_settings['quantity'] = st.sidebar.slider(
        "Quantità Onde",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Numero di linee ondulate visualizzate."
    )
    waves_settings['amplitude_scale'] = st.sidebar.slider(
        "Scala Ampiezza Onde",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Intensità di quanto le onde si curvano (modulata da audio)."
    )
    waves_settings['frequency_scale'] = st.sidebar.slider(
        "Frequenza Onde",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Densità delle 'creste' o 'valli' delle onde (modulata da audio)."
    )
    waves_settings['speed'] = st.sidebar.slider(
        "Velocità Onde",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Velocità di movimento delle onde sullo schermo."
    )
    waves_settings['line_thickness'] = st.sidebar.slider(
        "Spessore Linee Onde",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Spessore delle linee ondulate."
    )
    waves_settings['randomness_spread'] = st.sidebar.slider(
        "Dispersione Casuale Onde",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Aggiunge un leggero offset casuale a ogni punto per un look più organico o 'granuloso'."
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
    'use_frequency_colors': st.sidebar.checkbox("Usa colori frequenza (per elemento)", value=True),
    'background_color': st.sidebar.color_picker("Colore sfondo", "#000000"),
    'low_freq_color': st.sidebar.color_picker("Frequenze basse", "#FF0000"),
    'mid_freq_color': st.sidebar.color_picker("Frequenze medie", "#00FF00"),
    'high_freq_color': st.sidebar.color_picker("Frequenze acute", "#0000FF"),
    'element_color_intensity': st.sidebar.slider(
        "Intensità Colore Elementi",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
}
st.sidebar.markdown("<small>*I colori delle frequenze influenzeranno la colorazione dinamica di **ogni singolo elemento** disegnato in tutti i pattern, se abilitato. Per 'Linee Scomposte', se i colori di frequenza sono disabilitati, le linee saranno bianco/nero per contrasto. Le **'Onde Astratte' useranno i colori delle frequenze basse/acute per una gradazione** se abilitato, altrimenti un gradiente fisso.</small>", unsafe_allow_html=True)


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
                        selected_pattern_mode, glitch_settings, particles_settings, burst_settings, waves_settings
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
1.  **Carica** un file audio (MP3, WAV, etc.)
2.  **Imposta** formato video
3.  **Scegli** la visualizzazione tra "Geometric Random Burst", "Linee Scomposte (Glitch)", "Particelle Reattive" e il nuovo **"Onde Astratte"**.
4.  **Personalizza** intensità movimento, sincronizzazione BPM, **controlli specifici per ogni effetto** e colori.
5.  **Genera** il tuo video artistico!

### 🎵 Caratteristiche BPM:
-   **Sincronizzazione automatica** sul tempo del brano
-   **Modulazione dinamica** di zoom, movimento e colori
-   **Diversi tipi di sync**: beat principale, mezzi beat, terzine
-   **Transizioni smooth** per effetti fluidi

### 🌀 Visualizzazioni disponibili:
-   **Geometric Random Burst**: Un'esplosione dinamica di forme geometriche casuali che reagiscono all'audio e ai colori delle frequenze. Ora con **colori per elemento basati sulle frequenze, controllo della quantità e selezione del tipo di figura!**
-   **Linee Scomposte (Glitch)**: Linee che si "rompono" e glitchano in base all'audio. Ora con **colori per elemento reattivi alle frequenze (o bianco/nero per contrasto) e scelta dell'orientamento (verticale, orizzontale o entrambi)!**
-   **Particelle Reattive**: Una nuvola di particelle dinamiche che si muovono, pulsano e cambiano colore in base al volume e alle frequenze dell'audio, creando un'esperienza fluida e organica. Ora con **controllo su quantità, intensità del colore e casualità del movimento, e colori individuali per particella in base alla frequenza!**
-   **Onde Astratte**: Un nuovo pattern di linee ondulate e fluide che attraversano lo schermo, reagendo al volume e alle frequenze dell'audio. Le linee mostrano una **gradazione di colore** basata sui colori delle frequenze (o un gradiente fisso se disabilitati), per un effetto visivo moderno e organico.
""")
