import tkinter

from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image

from gui_reportes import crear_ventana_reportes
from reportes import *
from clasificador import predecir

root = tkinter.Tk()

label_ds = tkinter.Label(root, image=None)
label_ds.place(x=10, y=50)

label_es = tkinter.Label(root, image=None)
label_es.place(x=10, y=150)

label_em = tkinter.Label(root, image=None)
label_em.place(x=310, y=150)

label_mf = tkinter.Label(root, image=None)
label_mf.place(x=10, y=400)

label_pred = tkinter.Label(root, image=None)
label_pred.place(x=600, y=100)


def mostrar_imagen(path, label, w, h):
    img = Image.open(path)
    img = img.resize((w, h), Image.ANTIALIAS)
    newimg = ImageTk.PhotoImage(img)
    label.configure(image=newimg)
    label.image = newimg


def diagrama_frecuencia(y):
    _, path = diagrama_senal(y)
    mostrar_imagen(path, label_ds, 600, 50)


def diagrama_es(y, sr):
    _, path = diagrama_espectrograma(y, sr)
    mostrar_imagen(path, label_es, 300, 250)


def diagrama_em(y, sr):
    _, path = diagrama_espectrograma_mel(y, sr)
    mostrar_imagen(path, label_em, 300, 250)


def diagrama_mfc(y, sr):
    _, path = diagrama_mfccs(y, sr)
    mostrar_imagen(path, label_mf, 300, 250)


def diagrama_pred(path):
    prediccion = predecir(path)
    _, path = diagrama_predicciones(prediccion)
    print(prediccion)
    mostrar_imagen(path, label_pred, 500, 400)


def analizar_cancion(path):
    y, sr = cargar_cancion(path)
    if y is None:
        return

    diagrama_frecuencia(y)
    diagrama_es(y, sr)
    diagrama_em(y, sr)
    diagrama_mfc(y, sr)
    diagrama_pred(path)


def seleccionar_cancion():
    filetypes = (("archivos wav", "*.wav"), ("archivos mp3", "*.mp3"))

    filename = fd.askopenfilename(title="Selecciona tu archivo", filetypes=filetypes)

    if filename == "":
        return

    txt_direccion_archivo_update(filename)
    analizar_cancion(filename)


def txt_direccion_archivo_update(text):
    direccion_archivo.delete(1.0, "end")
    direccion_archivo.insert(1.0, text)


def abrir_reportes():
    crear_ventana_reportes(root)
    return


def quit():
    root.quit()
    root.destroy()


root.resizable(False, False)
root.geometry("1200x700")

root.title("Clasificador Musical - Vista Principal")

btn_abrir = ttk.Button(root, text="Analizar canción", command=seleccionar_cancion)
btn_abrir.place(x=500, y=10)

btn_reportes = ttk.Button(root, text="Abrir reportes", command=abrir_reportes)
btn_reportes.place(x=700, y=10)


direccion_archivo = tkinter.Text(root, height=2, width=60)
direccion_archivo.bind("<Key>", lambda e: "break")
direccion_archivo.place(x=10, y=10)

analizar_cancion("./db/blues/blues.00000.wav")

root.protocol("WM_DELETE_WINDOW", quit)

tkinter.mainloop()
