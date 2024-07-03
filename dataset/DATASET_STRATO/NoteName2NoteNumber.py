# -----------------------------------------------------------------------------
# Proyecto de Fin de Grado - Universidad Politécnica de Madrid (UPM)
# Sistema de conversión Audio-MIDI basado en redes neuronales
# Desarrollado por: Nicolás Uriz Roldán
# Fecha: Julio 2024
# -----------------------------------------------------------------------------
# Descripción:
# Función auxiliar la traducción de nombres de notas
# -----------------------------------------------------------------------------

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVE = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)


def name2number_str(name) -> str:
    # name_components = name.split()
    name_components = []
    for i in name:
        name_components.append(i)

    note = ""
    octave = 0

    # Contar el numero de componentes y si es 2 es nota natural y si es 3 es nota sostenida (natural +1)
    if len(name_components) == 2:
        # Es nota natural
        sostenido = 0
        note = name_components[0]
        octave = int(name_components[1])
        # print("no sostenido", sostenido)

    elif len(name_components) == 3:
        # Nota con sostenido
        sostenido = 1
        note = name_components[0]
        octave = int(name_components[2])
        # print("sostenido", sostenido)

    if note == "C":
        base_number = 24
    elif note == "D":
        base_number = 26
    elif note == "E":
        base_number = 28
    elif note == "F":
        base_number = 29
    elif note == "G":
        base_number = 31
    elif note == "A":
        base_number = 21
    elif note == "B":
        base_number = 23

    number = base_number + 12 * (octave - 1) + sostenido

    return str(number)


def name2number_int(name) -> int:
    # name_components = name.split()
    name_components = []
    for i in name:
        name_components.append(i)

    note = ""
    octave = 0

    # Contar el numero de componentes y si es 2 es nota natural y si es 3 es nota sostenida (natural +1)
    if len(name_components) == 2:
        # Es nota natural
        sostenido = 0
        note = name_components[0]
        octave = int(name_components[1])
        # print("no sostenido", sostenido)

    elif len(name_components) == 3:
        # Nota con sostenido
        sostenido = 1
        note = name_components[0]
        octave = int(name_components[2])
        # print("sostenido", sostenido)

    if note == "C":
        base_number = 24
    elif note == "D":
        base_number = 26
    elif note == "E":
        base_number = 28
    elif note == "F":
        base_number = 29
    elif note == "G":
        base_number = 31
    elif note == "A":
        base_number = 21
    elif note == "B":
        base_number = 23

    number = base_number + 12 * (octave - 1) + sostenido

    return number


def number2presentation(number):
    octave = number // NOTES_IN_OCTAVE
    note = NOTES[number % NOTES_IN_OCTAVE]
    name = note + octave
    return name


# if __name__ == "__main__":
#     nombre1 = "Cs1"
#     nombre2 = "A2"
#     numerito = NoteName2NoteName(nombre1)
#     print(numerito)
