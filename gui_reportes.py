import tkinter
from tkinter import *

from PIL import ImageTk, Image

accuracy_epoch = "./img/cnn/accuracy_epoch.png"
conf_mat = "./img/cnn/conf_mat.png"
pe_epoch = "./img/cnn/pe_epoch.png"


def mostrar_imagen(path, label, w, h):
    img = Image.open(path)
    img = img.resize((w, h), Image.ANTIALIAS)
    newimg = ImageTk.PhotoImage(img)
    label.configure(image=newimg)
    label.image = newimg


def crear_ventana_reportes(root):
    ventana_reportes = Toplevel(root)

    ventana_reportes.title("Reportes Red Neuronal Convolucional - MGC")
    ventana_reportes.geometry("900x850")

    # Precision por entrenamiento
    label_exactitud = tkinter.Label(ventana_reportes, image=None)
    label_exactitud.place(x=10, y=10)

    mostrar_imagen(accuracy_epoch, label_exactitud, 400, 400)

    # Matriz de confusion
    label_matriz = tkinter.Label(ventana_reportes, image=None)
    label_matriz.place(x=10, y=420)

    mostrar_imagen(conf_mat, label_matriz, 800, 400)

    # Pruebas y entrenamientos por epoch
    label_epocas = tkinter.Label(ventana_reportes, image=None)
    label_epocas.place(x=420, y=10)

    mostrar_imagen(pe_epoch, label_epocas, 400, 400)

    return ventana_reportes
