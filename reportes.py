import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def cargar_cancion(path):
    try:
        y, sr = librosa.load(path)
        return y, sr
    except:
        return None


def diagrama_senal(y, path="./img/dia_amplitud.png"):
    print(">> Generando plot de señal")

    plt.figure(figsize=(5, 1))
    plt.plot(y)
    plt.title("Diagrama de frecuencia")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.gcf().set_size_inches(5, 1)

    print(f">> Guardando archivo {path}")
    plt.savefig(path)

    return plt, path


def diagrama_espectrograma(y, sr, path="./img/dia_es.png"):
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma de frecuencias")
    plt.savefig(path)

    return plt, path


def diagrama_espectrograma_mel(y, sr, path="./img/dia_em.png"):
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(spect, y_axis="mel", fmax=8000, x_axis="time")
    plt.title("Espectrograma de mel")
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(path)

    return plt, path


def diagrama_mfccs(y, sr, path="./img/dia_mfccs.png"):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.title("MFCC")
    # mfccscaled = np.mean(mfcc.T, axis=0)
    plt.savefig(path)

    return plt, path


def diagrama_predicciones(predicciones, path="./img/dia_pred.png"):
    labels = ["Blues", "Clásica", "Country", "Rock", "Jazz"]
    lb = [0, 1, 2, 3, 4]

    explode = [0, 0, 0, 0, 0]
    index_max = np.argmax(predicciones)
    labels2 = ["%s, %1.1f %%" % (l, s) for l, s in zip(labels, predicciones[0] * 100)]
    explode[index_max] = 0.1
    exploce = tuple(explode)

    plt.figure(figsize=(8, 8))
    _, ax1 = plt.subplots()
    ax1.pie(
        predicciones[0],
        explode=explode,
        shadow=True,
        startangle=90,
    )
    plt.legend(bbox_to_anchor=(0.85, 1), loc="upper left", labels=labels2)
    ax1.axis("equal")
    plt.savefig(path)
    return plt, path
