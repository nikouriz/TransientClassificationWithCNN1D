import ast

import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, random_split

# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Función para definir el comportamiento del objeto Dataset
# -----------------------------------------------------------------------------


class StratoDataset(Dataset):

    def __init__(self, csv_path, device):
        self.data_frame = pd.read_csv(csv_path, index_col=0)
        self.data_frame["audio"] = self.data_frame["audio"].apply(ast.literal_eval)
        self.device = device
        # self.expected_sr = expected_sr
        # self.expected_len = expected_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        label = self._get_label(index)  # label -> numpy.int64
        audio_samp = self._get_audio_samp(index)  # audio_samp -> list
        # audio_samp = torch.unsqueeze(audio_samp, 1)
        # audio_samp = torch.permute(audio_samp, (1, 0, 2))
        # [1, 64, 512] -> [64, 1, 512]
        # audio_samp = audio_samp.permute(()) # testear si implementar aqui o fuera de la clase
        return audio_samp, label

    def _get_audio_samp(self, index):
        audio = self.data_frame.iloc[index, 1]
        audio = np.asarray(audio)
        audio = torch.from_numpy(audio.astype(np.float32))
        audio = audio.to(self.device)
        return audio

    def _get_label(self, index):

        label = int(self.data_frame.iloc[index, [0]])
        id = label - 40
        id = torch.tensor(id, dtype=torch.long)  # dtype=torch.int8
        return id


if __name__ == "__main__":

    CSV_PATH = "DatasetPreprocessing/output/data1.csv"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    dataset = StratoDataset(CSV_PATH, device)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        print(targets)
