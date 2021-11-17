import os
from keras.models import model_from_json
from procesamiento_audio import espectrograma_mel


def cargar_cnn_json(path="./modelos/cnn.json"):
    print(f">> Abriendo archivo json {path}")
    with open(path) as jf:
        modelo_cnn_json = jf.read()
        return model_from_json(modelo_cnn_json)


def cargar_cnn_h5(model, path="./modelos/cnn.h5"):
    print(f">> Abriendo archivo h5 {path}")
    model.load_weights(path)


def cargar_cnn():
    print(">> Cargando CNN")
    modelo = cargar_cnn_json()
    cargar_cnn_h5(modelo)
    return modelo


cnn = cargar_cnn()

cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(">> CNN Cargada y lista para utilizar")


def obtener_espectrograma(path):
    espectrograma = espectrograma_mel(path)
    espectrograma /= espectrograma.min()
    return espectrograma.reshape(espectrograma.shape[0], 128, 657, 1)


def predecir(path, opcion="cnn"):
    print(f">> Prediciendo genero con opcion en {opcion}")
    espectrograma = obtener_espectrograma(path)

    if opcion == "cnn":
        predicciones = cnn.predict(espectrograma, verbose=1)
        return predicciones
