import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("./db/blues/blues.00000.wav")
print(y.shape, sr)

plt.plot(y)
plt.title("Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()

spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)  # converting to decibals

plt.figure(figsize=(8, 5))
librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.show()

spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
spect = librosa.power_to_db(spect, ref=np.max)

plt.figure(figsize=(8, 5))
librosa.display.specshow(spect, y_axis="mel", fmax=8000, x_axis="time")
plt.title("Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")
plt.show()

mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
plt.figure(figsize=(8, 5))
librosa.display.specshow(mfcc, x_axis="time")
plt.title("MFCC")
# plt.show()
print(mfcc)
mfccscaled = np.mean(mfcc.T, axis=0)
print(mfccscaled)

# Creating an empty list to store sizes in
sizes = []

directorios = ['blues', 'classical', 'country', 'jazz', 'rock']

for directorio in directorios: 
    actual = f"./db/{directorio}"
    for file in os.scandir(actual):
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adding the size to the list
        sizes.append(spect.shape)

# Checking if all sizes are the same
print(
    f"The sizes of all the mel spectrograms in our data set are equal: {len(set(sizes)) == 1}"
)

# Checking the max size
print(f"The maximum size is: {max(sizes)}")
print("Revisados: ", len(sizes))

def extract_mel_spectrogram(directory):
    """
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the
    same size, and stores them in a numpy array.

    It also creates a list of genre labels and maps them to numeric values.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    X (array): array of mel spectrogram data from all audio files in the given
    directory
    y (array): array of the corresponding genre labels in numeric form
    """

    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []

    # Looping through each file in the directory
    for file in os.scandir(directory):

        if ".mp3" in file.name:
            continue

        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Extracting the label and adding it to the list
        label = str(file).split(".")[0][11:]
        labels.append(label)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adjusting the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128, 660, refcheck=False)

        # Adding the mel spectrogram to the list
        mel_specs.append(spect)

    # Converting the list or arrays to an array
    X = np.array(mel_specs)

    # Converting labels to numeric values
    labels = pd.Series(labels)
    label_dict = {
        "jazz": 1,
        "reggae": 2,
        "rock": 3,
        "blues": 4,
        "hiphop": 5,
        "country": 6,
        "metal": 7,
        "classical": 8,
        "disco": 9,
        "pop": 10,
    }
    y = labels.map(label_dict)

    # Returning the mel spectrograms and labels
    return X, y


def make_mel_spectrogram_df(directory):
    """
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the
    same size, flattens them, and stores them in a dataframe.

    Genre labels are also computed and added to the dataframe.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a dataframe of flattened mel spectrograms and their
    corresponding genre labels
    """

    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []

    # Looping through each file in the directory
    for file in os.scandir(directory):

        if ".mp3" in file.name:
            continue
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Extracting the label and adding it to the list
        label = str(file).split(".")[0][11:]
        labels.append(label)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adjusting the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128, 660, refcheck=False)

        # Flattening to fit into dataframe and adding to the list
        spect = spect.flatten()
        mel_specs.append(spect)

    # Converting the lists to arrays so we can stack them
    mel_specs = np.array(mel_specs)
    labels = np.array(labels).reshape(100, 1)

    # Create dataframe
    df = pd.DataFrame(np.hstack((mel_specs, labels)))

    # Returning the mel spectrograms and labels
    return df


X, y = extract_mel_spectrogram("./db/blues")
df = make_mel_spectrogram_df("./db/blues")

df.to_csv("./data/genre_mel_specs.csv", index=False)


def extract_audio_features(directory):
    """
    This function takes in a directory of .wav files and returns a
    DataFrame that includes several numeric features of the audio file
    as well as the corresponding genre labels.

    The numeric features incuded are the first 13 mfccs, zero-crossing rate,
    spectral centroid, and spectral rolloff.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a table of audio files that includes several numeric features
    and genre labels.
    """

    # Creating an empty list to store all file names
    files = []
    labels = []
    zcrs = []
    spec_centroids = []
    spec_rolloffs = []
    mfccs_1 = []
    mfccs_2 = []
    mfccs_3 = []
    mfccs_4 = []
    mfccs_5 = []
    mfccs_6 = []
    mfccs_7 = []
    mfccs_8 = []
    mfccs_9 = []
    mfccs_10 = []
    mfccs_11 = []
    mfccs_12 = []
    mfccs_13 = []

    # Looping through each file in the directory
    for file in os.scandir(directory):

        if ".mp3" in file.name:
            continue

        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Adding the file to our list of files
        files.append(file)

        # Adding the label to our list of labels
        label = str(file).split(".")[0]
        labels.append(label)

        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y)
        zcrs.append(np.mean(zcr))

        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y)
        spec_centroids.append(np.mean(spec_centroid))

        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y)
        spec_rolloffs.append(np.mean(spec_rolloff))

        # Calculating the first 13 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfccs_1.append(mfcc_scaled[0])
        mfccs_2.append(mfcc_scaled[1])
        mfccs_3.append(mfcc_scaled[2])
        mfccs_4.append(mfcc_scaled[3])
        mfccs_5.append(mfcc_scaled[4])
        mfccs_6.append(mfcc_scaled[5])
        mfccs_7.append(mfcc_scaled[6])
        mfccs_8.append(mfcc_scaled[7])
        mfccs_9.append(mfcc_scaled[8])
        mfccs_10.append(mfcc_scaled[9])
        mfccs_11.append(mfcc_scaled[10])
        mfccs_12.append(mfcc_scaled[11])
        mfccs_13.append(mfcc_scaled[12])

    # Creating a data frame with the values we collected
    df = pd.DataFrame(
        {
            "files": files,
            "zero_crossing_rate": zcrs,
            "spectral_centroid": spec_centroids,
            "spectral_rolloff": spec_rolloffs,
            "mfcc_1": mfccs_1,
            "mfcc_2": mfccs_2,
            "mfcc_3": mfccs_3,
            "mfcc_4": mfccs_4,
            "mfcc_5": mfccs_5,
            "mfcc_6": mfccs_6,
            "mfcc_7": mfccs_7,
            "mfcc_8": mfccs_8,
            "mfcc_9": mfccs_9,
            "mfcc_10": mfccs_10,
            "mfcc_11": mfccs_11,
            "mfcc_12": mfccs_12,
            "mfcc_13": mfccs_13,
            "labels": labels,
        }
    )

    # Returning the data frame
    return df


# Using the function to read and extract the audio files from the GTZAN Genre Dataset
df = extract_audio_features("./db/blues")
df.to_csv("./data/genre.csv", index=False)
