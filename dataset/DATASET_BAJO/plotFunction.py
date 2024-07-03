import librosa
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Función auxiliar para generar gráficas
# -----------------------------------------------------------------------------

FIG_SIZE = (12, 7)


def plotTimeMulti(
    samples1,
    label1,
    samples2,
    label2,
    samples3,
    label3,
    samples4,
    label4,
    samples5,
    label5,
    sr,
):

    figure = plt.figure(figsize=FIG_SIZE)
    ax1 = figure.subplots()

    x_ax = np.zeros(len(samples1))  # Simular el eje horizontal en 0

    librosa.display.waveshow(samples1, sr=sr, ax=ax1, color="r", label=label1)
    librosa.display.waveshow(samples2, sr=sr, ax=ax1, color="g", label=label2)
    librosa.display.waveshow(samples3, sr=sr, ax=ax1, color="b", label=label3)
    librosa.display.waveshow(samples4, sr=sr, ax=ax1, color="r", label=label4)
    librosa.display.waveshow(samples5, sr=sr, ax=ax1, color="#000000", label=label5)
    librosa.display.waveshow(x_ax, sr=sr, ax=ax1, color="#000000", alpha=0.5)
    ax1.legend()

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (lineal)")
    plt.show()


def plotTimeMulti12(
    samples1,
    label1,
    samples2,
    label2,
    samples3,
    label3,
    samples4,
    label4,
    samples5,
    label5,
    samples6,
    label6,
    samples7,
    label7,
    samples8,
    label8,
    samples9,
    label9,
    samples10,
    label10,
    samples11,
    label11,
    samples12,
    label12,
    sr,
):

    figure = plt.figure(figsize=FIG_SIZE)
    ax1 = figure.subplots()

    x_ax = np.zeros(len(samples1))  # Simular el eje horizontal en 0

    librosa.display.waveshow(samples1, sr=sr, ax=ax1, label=label1)
    librosa.display.waveshow(samples2, sr=sr, ax=ax1, label=label2)
    librosa.display.waveshow(samples3, sr=sr, ax=ax1, label=label3)
    librosa.display.waveshow(samples4, sr=sr, ax=ax1, label=label4)
    librosa.display.waveshow(samples5, sr=sr, ax=ax1, label=label5)
    librosa.display.waveshow(samples6, sr=sr, ax=ax1, label=label6)
    librosa.display.waveshow(samples7, sr=sr, ax=ax1, label=label7)
    librosa.display.waveshow(samples8, sr=sr, ax=ax1, label=label8)
    librosa.display.waveshow(samples9, sr=sr, ax=ax1, label=label9)
    librosa.display.waveshow(samples10, sr=sr, ax=ax1, label=label10)
    librosa.display.waveshow(samples11, sr=sr, ax=ax1, label=label11)
    librosa.display.waveshow(samples12, sr=sr, ax=ax1, label=label12)
    librosa.display.waveshow(x_ax, sr=sr, ax=ax1, color="#000000", alpha=0.5)
    ax1.legend()

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (lineal)")
    plt.show()
