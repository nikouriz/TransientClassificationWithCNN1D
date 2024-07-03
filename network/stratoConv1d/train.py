import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cnn1d import CNN
from stratodataset import StratoDataset
from torch import nn
from torch.utils.data import DataLoader, random_split

# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Programa de entrenamiento de la red
# -----------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
CSV_PATH = "network/stratoConv1d/datasets/StratoDatset.csv"
OUTPUT_PATH = "network/stratoConv1d/output"
NUM_LABELS = 5


def create_data_loader(data, batch_size, shuffle):
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader


def id2label(id):
    return id + 40


def mse_loss(y_pred, y_targ):
    # MSE = 1/n*sum(y_pred - y_targ)^2
    y_targ = tensor2matrix(y_targ).detach().numpy()
    y_pred = y_pred.detach().numpy()
    n = 0
    suma = 0

    for fila_pred, fila_targ in zip(y_pred, y_targ):
        for val_pred, val_targ in zip(fila_pred, fila_targ):
            n += 1
            dif = (val_pred - val_targ) ** 2
            suma += dif
    mse = suma / n

    return mse


def tensor2matrix(y_targ):
    y_targ_matrix = np.zeros((len(y_targ), NUM_LABELS), dtype=np.float32)
    for i, fila in enumerate(y_targ_matrix):
        target_index = y_targ[i]
        fila[target_index] = 1
    y_targ_tensor = torch.from_numpy(y_targ_matrix)
    return y_targ_tensor


def shape_input(input, target):
    input, target = input[None, :, :].to(device), target.to(device)
    input = torch.permute(input, (1, 0, 2))
    target = torch.squeeze(target)
    return input, target


def train_test_split(dataset, train_relative_size):
    train_size = int(train_relative_size * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    return train_data, test_data


def train(model, train_dataloader, test_dataloader, loss_fn, optimiser, device, epochs):

    # n_total_steps = len(train_dataloader)
    global mse_train
    global mse_train_withtest
    global accuracy_train
    global accuracy_train_withtest

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        mse_epoch = []
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        for i, (input, target) in enumerate(train_dataloader):

            # adaptación de dimension y formato de los tensores
            # # [1, 64, 512] -> [64, 1, 512]
            input, target = shape_input(input=input, target=target)

            optimiser.zero_grad()
            outputs = model(input)  # sin softmax: logits

            # calcular error
            loss = loss_fn(outputs, target)

            # calcular MSE y almacenarlo
            probabilities = F.softmax(
                outputs, dim=1
            )  # se aplica softmax para obtener las probabilidades y calcular mse
            _, predictions = torch.max(probabilities, 1)
            mse_epoch.append(mse_loss(probabilities, target))

            # backpropagate error, actualizar pesos
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            running_correct += (predictions == target).sum().item()
            running_samples += target.shape[0]

        mse_epoch_total = sum(mse_epoch) / len(mse_epoch)
        accuracy = running_correct / running_samples
        mse_train.append(mse_epoch_total)
        accuracy_train.append(accuracy)  # método anterior

        with torch.no_grad():
            acc_over_test, mse_over_test = test(
                model=model, test_dataloader=test_dataloader, device=device, log=False
            )
        accuracy_train_withtest.append(acc_over_test)
        mse_train_withtest.append(mse_over_test)

        print(f"loss: {loss.item():.4f}")
        print(f"accuracy: {(100.0*acc_over_test):.2f} %")
        print("---------------------------")

    print("Finished training")


def test(model, test_dataloader, device, log):

    # global mse_test
    global accuracy_test
    n_correct = 0
    n_samples = 0
    mse_test = []

    for input, target in test_dataloader:
        # adaptación de dimension y formato de los tensores
        input, target = input[None, :, :].to(device), target.to(device)
        input = torch.permute(input, (1, 0, 2))
        target = torch.squeeze(target)

        # paso de datos por la red
        outputs = model(input)
        probabilities = F.softmax(outputs, dim=1)  # calcular probabilidades

        _, predictions = torch.max(probabilities, 1)  # predicción de salida

        # conteo de muestras y aciertos
        n_samples += target.shape[0]
        n_correct += (predictions == target).sum().item()

        # calcular MSE y almacenarlo
        mse = mse_loss(probabilities, target)
        temp_accuracy = ((predictions == target).sum().item()) / target.shape[0]
        mse_test.append(mse)
        accuracy_test.append(temp_accuracy)
        # running_samples += target.shape[0]

    mse_average = sum(mse_test) / len(mse_test)
    accuracy = n_correct / n_samples
    if log:
        print(f"Accuracy = {(100*accuracy):.2f} %")
    return accuracy, mse_average


def plot_metrics(
    mse_train, mse_train_withtest, accuracy_train, accuracy_train_withtest
):
    figure, axs = plt.subplots(2, 1)
    figure.set_figheight(8)
    figure.set_figwidth(12)
    axs[0].plot(
        [i for i in range(1, len(mse_train) + 1)],
        mse_train,
        color="r",
    )
    axs[0].plot(
        [i for i in range(1, len(mse_train_withtest) + 1)],
        mse_train_withtest,
        color="b",
    )
    axs[0].legend(
        ("MSE Train", "MSE Test"),
        loc="upper right",
    )
    axs[0].set_title("MSE durante Entrenamiento")
    plt.xlabel("Nº Epoch")
    plt.ylabel("")

    # ACCURACY PLOT
    axs[1].plot(
        [i for i in range(1, len(accuracy_train) + 1)],
        accuracy_train,
        color="r",
    )
    axs[1].plot(
        [i for i in range(1, len(accuracy_train_withtest) + 1)],
        accuracy_train_withtest,
        color="b",
    )
    axs[1].legend(
        ("Accuracy Train", "Accuracy Test"),
        loc="lower right",
    )
    axs[1].set_title("Accuracy durante Entrenamiento")
    plt.xlabel("Nº Epoch")
    plt.show()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    dataset = StratoDataset(CSV_PATH, device)

    train_dataset, test_dataset = train_test_split(dataset, 0.8)

    train_dataloader = create_data_loader(
        data=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = create_data_loader(
        data=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    print(
        f"Tamaño Dataset Train: {len(train_dataloader.dataset)}\nTamaño Dataset Test: {len(test_dataloader.dataset)}"
    )

    model = CNN().to(device)
    print(model)

    # inicializar función de pérdidas y optimizador
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Métricas para analizar el comportamiento de la red durante train y test
    mse_test = []
    mse_train = []
    mse_train_withtest = []
    accuracy_test = []
    accuracy_train = []
    accuracy_train_withtest = []

    # train
    train(model, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)

    # test
    with torch.no_grad():
        test(model, test_dataloader, device, True)

    plot_metrics(mse_train, mse_train_withtest, accuracy_train, accuracy_train_withtest)

    torch.save(model.state_dict(), f"{OUTPUT_PATH}/model1d.pth")
    print("Trained convolutional network saved at cnn1d.pth")
