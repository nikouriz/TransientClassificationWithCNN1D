import ast
import random as rand

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotFunction import plotTimeMulti, plotTimeMulti12

# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Programa auxiliar para el checkeo del archivo .csv
# -----------------------------------------------------------------------------

CSV_PATH = "dataset/DATASET_BAJO/output/BassDataset2.csv"
SAMPLE_RATE = 48_000
FIG_SIZE = (12, 7)


def main():
    df = pd.read_csv(CSV_PATH, index_col=0)
    # print(df.head())

    # Typecast de la columna de audio, csv almacena float como str
    df["audio"] = df["audio"].apply(ast.literal_eval)

    data = df.to_dict()
    labels = []
    i = 0
    rand_selection = []

    while i < 12:
        rn = rand.randint(0, len(data["label"]) - 1)
        label = data["label"][rn]
        if label not in labels:
            labels.append(label)
            samples = data["audio"][rn]
            # print(samples[0])
            rand_selection.append({"label": label, "audio": samples})
            i += 1
        else:
            continue

    print("Se ha salido del bucle")
    df = pd.DataFrame(data=rand_selection)
    df["audio"] = df["audio"].apply(np.array)

    rand_selection = df.to_dict()

    plotTimeMulti12(
        samples1=rand_selection["audio"][0],
        label1=rand_selection["label"][0],
        samples2=rand_selection["audio"][1],
        label2=rand_selection["label"][1],
        samples3=rand_selection["audio"][2],
        label3=rand_selection["label"][2],
        samples4=rand_selection["audio"][3],
        label4=rand_selection["label"][3],
        samples5=rand_selection["audio"][4],
        label5=rand_selection["label"][4],
        samples6=rand_selection["audio"][5],
        label6=rand_selection["label"][5],
        samples7=rand_selection["audio"][6],
        label7=rand_selection["label"][6],
        samples8=rand_selection["audio"][7],
        label8=rand_selection["label"][7],
        samples9=rand_selection["audio"][8],
        label9=rand_selection["label"][8],
        samples10=rand_selection["audio"][9],
        label10=rand_selection["label"][9],
        samples11=rand_selection["audio"][10],
        label11=rand_selection["label"][10],
        samples12=rand_selection["audio"][11],
        label12=rand_selection["label"][11],
        sr=SAMPLE_RATE,
    )


if __name__ == "__main__":
    main()
