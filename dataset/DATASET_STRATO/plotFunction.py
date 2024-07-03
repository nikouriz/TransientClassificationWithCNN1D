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
