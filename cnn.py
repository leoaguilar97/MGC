import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.keras.utils import to_categorical
from procesamiento_audio import espectrogramas_mel

print(">> Iniciando creacion y entrenamiento de CNN")

print(">> Obteniendo espectrogramas y etiquetas")
espectrogramas, etiquetas = espectrogramas_mel()

print(">> Recolectando datos de prueba")

entr_espect, prueba_espect, entr_etiquetas, prueba_etiquetas = train_test_split(
    espectrogramas, etiquetas, random_state=100, stratify=etiquetas, test_size=0.2
)

entr_espect /= entr_espect.min()
prueba_espect /= entr_espect.min()

# 657 es el maximo
entr_espect = entr_espect.reshape(entr_espect.shape[0], 128, 657, 1)
prueba_espect = prueba_espect.reshape(prueba_espect.shape[0], 128, 657, 1)

# 5 generos en labels
entr_etiquetas = to_categorical(entr_etiquetas, 5)
prueba_etiquetas = to_categorical(prueba_etiquetas, 5)

# Para replicar los ejemplos, hacer el random con un seed
np.random.seed(0)
tf.random.set_seed(0)

print(">> Informacion lista para el entrenamiento")

print(">> Creando CNN")

# Iniciar el modelo CNN secuencial
red_neuronal = Sequential(name="MGC_CNN")

print(">> Agregando red convolucional inicial")

# Agregar la primera capa, una convolucional
# 16 filtros, kernel de 3x3, activacion relu y el input shape definido anteriormente
red_neuronal.add(
    Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(128, 657, 1))
)

print(">> Agregando MaxPooling2D")
# Agregar una capa MaxPooling2D para obtener todos los mayores
# tamaño pool: 2x4
red_neuronal.add(MaxPooling2D(pool_size=(2, 4)))

print(">> Agregando siguiente red convolucional")
# Agregar otra capa convolucional
# 32 filtros, kernel de 3x3 y activacion relu
red_neuronal.add(Conv2D(filters=32, kernel_size=3, activation="relu"))

print(">> Agregando siguiente capa MaxPooling2D")
# De nuevo una capa MaxPooling2D
# tamaño pool: 2x4
red_neuronal.add(MaxPooling2D(pool_size=(2, 4)))

print(">> Agregando red Flatten")
# Agregar una capa de aplanamiento
red_neuronal.add(Flatten())

print(">> Agregando red Dense de 64 neuronas RELU")
# Agergar una capa Dense de 64 neuronas, con activacion relu
red_neuronal.add(Dense(64, activation="relu"))

print(">> Agregando red Dropout de 25%")
# Ignorar 25% de los nodos resultantes para prevenir overfitting
red_neuronal.add(Dropout(0.25))

print(">> Agregando red neuronal Dense de 5 neuronas y Softmax, de salida")
# Red neuronal final para obtener los porcentajes de cada género
red_neuronal.add(Dense(5, activation="softmax"))

print(">> Compilando red neuronal CNN")
# Compilar la red neuronal
red_neuronal.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

print(">> Guardando red CNN")

# Entrenar el modelo utilizando la informacion de prueba generada anteriormente
history = red_neuronal.fit(
    entr_espect,
    entr_etiquetas,
    batch_size=16,
    validation_data=(prueba_espect, prueba_etiquetas),
    epochs=15,
)

# Guardando modelo
print(">> Guardando modelo en ./modelo.json")
model_json = red_neuronal.to_json()

with open("./modelos/cnn.json", "w") as jf:
    jf.write(model_json)

# Guardando los pesos del modelo para cargar posteriormente
print(">> Guardando pesos en ./modelo.h5")
red_neuronal.save_weights("./modelos/cnn.h5")

# FIN
print(">> Red Neuronal creada, entrenada y almacenada correctamente")

print(">> Creando reportes de la CNN")

print(">> Creando grafica de entrenamiento vs validacion por Epoch")
train_loss = history.history["loss"]
test_loss = history.history["val_loss"]

plt.figure(figsize=(12, 8))

plt.plot(train_loss, label="Pérdida de entrenamiento", color="blue")
plt.plot(test_loss, label="Pérdida de pruebas", color="red")

plt.title("Pruebas y entrenamiento por Epoch", fontsize=25)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Crossentropy", fontsize=18)
plt.xticks(range(1, 16), range(1, 16))

plt.legend(fontsize=18)

plt.savefig("./img/cnn/pe_epoch.png")

print(">> Creando grafica de exactitud vs validacion")
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

plt.figure(figsize=(12, 8))

plt.plot(accuracy, label="Precision de entrenamiento", color="blue")
plt.plot(val_accuracy, label="Precision de pruebas", color="red")

plt.title("Presición por entrenamiento", fontsize=25)
plt.xlabel("Entrenamiento", fontsize=18)
plt.ylabel("Precisión", fontsize=18)
plt.xticks(range(1, 21), range(1, 21))

plt.savefig("./img/cnn/accuracy_epoch.png")

print(">> Creando matriz de confusion")

predicciones = red_neuronal.predict(prueba_espect, verbose=1)

for i in range(5):
    print(f"{i}: {sum([1 for aciertos in prueba_etiquetas if aciertos[i] == 1])}")

for i in range(5):
    print(
        f"{i}: {sum([1 for prediccion in predicciones if np.argmax(prediccion) == i])}"
    )

conf_matrix = confusion_matrix(
    np.argmax(prueba_etiquetas, 1), np.argmax(predicciones, 1)
)

confusion_df = pd.DataFrame(conf_matrix)

plt.figure(figsize=(20, 12))
sns.set(font_scale=2)
ax = sns.heatmap(confusion_df, annot=True, cmap=sns.cubehelix_palette(rot=-0.4))
ax.set(xlabel="Valores predecidos", ylabel="Valores correctos")
plt.savefig("./img/cnn/conf_mat.png")

print(">> Reportes creados correctamente")
