import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
from NoteName2NoteNumber import name2number_int

# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Programa principal para el preprocesado y segmentación de audio
# para la generación del .csv empleado como dataset
# -----------------------------------------------------------------------------

MIDI_FOLDER = "dataset/DATASET_BAJO/dataCaso2/midi"
AUDIO_FOLDER = "dataset/DATASET_BAJO/dataCaso2/audio"
OUTPUT_FOLDER = "dataset/DATASET_BAJO/output"

WINDOW_SIZE = 1024
DESPLAZAMIENTO = 0.001  # desplazamiento negativo de la ventana de recorte en segundos
SAMPLE_RATE = 48_000
FIG_SIZE = (12, 7)
OUTPUT_NAME = "BassDataset2"


def getAttackTime(file_path):
    # Lee el archivo midi que se encuentra en "file_path" y devuelve un vector con los
    # instantes de
    archivo = pretty_midi.PrettyMIDI(file_path)
    t_attack = []
    for instrumentos in archivo.instruments:
        for nota in instrumentos.notes:
            t_attack.append(nota.start)
    return t_attack


def getStartEndSamples(t_inicio, s_duracion):  # t en segundos y s en muestras
    # Cálculo de la ventana temporal del transitorio inicial
    s_inicio = int(np.ceil(t_inicio * SAMPLE_RATE))
    # s_duracion = int(np.ceil(t_duracion * SAMPLE_RATE))
    # t_final = t_inicio + t_duracion
    s_final = s_inicio + s_duracion
    return s_inicio, s_final


def getWindowedSamples(samples, t, desplazamiento, samples_offset=0):
    s_inicio, s_final = getStartEndSamples(
        t_inicio=t - desplazamiento, s_duracion=WINDOW_SIZE
    )
    window = samples[s_inicio + samples_offset : s_final + samples_offset]
    return window


def getExtendedWindowedSamples(samples, t, num_wind):
    s_inicio, s_final = getStartEndSamples(
        t_inicio=(t - WINDOW_SIZE / SAMPLE_RATE), s_duracion=(WINDOW_SIZE * num_wind)
    )
    window = samples[s_inicio:s_final]
    return window


def transientMarkSamples(t_trans, window_size):
    mk_samples = np.zeros(window_size)
    s_trans_ini = int(np.ceil(t_trans * SAMPLE_RATE))  # muestras = s * s^-1
    s_trans_end = s_trans_ini + window_size
    mk_samples[s_trans_ini - 1] = 0.9
    mk_samples[s_trans_end - 1] = 0.9
    return mk_samples


def plotTimeMulti(samples1, samples2, samples3, samples4, samples_transitorio, sr):

    figure = plt.figure(figsize=FIG_SIZE)
    ax1 = figure.subplots()

    x_ax = np.zeros(len(samples1))  # Simular el eje horizontal en 0

    librosa.display.waveshow(samples1, sr=sr, ax=ax1, color="r", label="T1")
    librosa.display.waveshow(samples2, sr=sr, ax=ax1, color="g", label="T2")
    librosa.display.waveshow(samples3, sr=sr, ax=ax1, color="b", label="T3")
    librosa.display.waveshow(samples4, sr=sr, ax=ax1, color="r", label="T4")
    librosa.display.waveshow(
        samples_transitorio, sr=sr, ax=ax1, color="#4A235A", label="Transitorio"
    )
    librosa.display.waveshow(x_ax, sr=sr, ax=ax1, color="#000000", alpha=0.5)
    ax1.legend()

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (lineal)")
    plt.show()


def formatFileInfo(file):
    filename, filextension = os.path.splitext(file)
    audio_filepath = AUDIO_FOLDER + os.sep + filename + ".wav"
    midi_filepath = MIDI_FOLDER + os.sep + filename + ".mid"
    note_number = name2number_int(filename)
    return audio_filepath, midi_filepath, note_number


def main():

    # 1. Para cada nota abrir archivo wav y midi
    #    Recorrer carpeta wav y para cada nota abrir el archivo midi asociado
    # 2. Recorrer todo el archivo recortando cada nota en función del código
    #    de tiempo del midi teniendo en cuenta el retardo de compensación
    # 3. Almacenar cada nota en el dataframe con su correspondient etiqueta
    # 4. Almacenar el dataframe

    data = []

    for file in os.listdir(AUDIO_FOLDER):

        if file[0] == ".":  # Omitir archivos ocultos que pueda haber en la carpeta
            continue

        audio_filepath, midi_filepath, note_number = formatFileInfo(file)
        T_attack = getAttackTime(
            midi_filepath
        )  # Extraer tiempos de ataque a partir del midi
        print(
            f"Cargando recortando el archivo: {audio_filepath}\nNumero de nota midi: {note_number}\nTotal de {len(T_attack)} transitorios\n"
        )

        samples, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)

        for t in T_attack:  # Recorrer todos los transitorios
            w_samples = getWindowedSamples(
                samples, t, DESPLAZAMIENTO
            )  # para coger la segunda ventana temporal
            w_samples = np.array(w_samples)  # quitar si no va
            data.append(
                {"label": note_number, "audio": w_samples.tolist()}
            )  # w_samples.tolist()

        print("Archivos recortados correctamente")
        # print(data["audio"][0])

    df = pd.DataFrame(data=data)
    print(df.head())

    print(f"Escribiendo archivo: {OUTPUT_FOLDER}/{OUTPUT_NAME}.csv")
    df.to_csv(f"{OUTPUT_FOLDER}/{OUTPUT_NAME}.csv")
    print("Archivo escrito CORRECTAMENTE")


if __name__ == "__main__":
    main()
