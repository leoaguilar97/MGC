# Imports
import os
import librosa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.keras.utils import to_categorical

def extract_mel_spectrogram(directory):
    '''
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
    '''
    
    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []
    
    
    # Looping through each file in the directory
    for file in os.scandir(directory):

        # Loading in the audio file
        y, sr = librosa.core.load(file)
        
        # Extracting the label and adding it to the list
        label = str(file).split('.')[0][11:]
        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        
        # Adjusting the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128,660, refcheck=False)
            
        # Adding the mel spectrogram to the list
        mel_specs.append(spect)
        
    # Converting the list or arrays to an array
    X = np.array(mel_specs)
    
    # Converting labels to numeric values
    labels = pd.Series(labels)
    label_dict = {
        'jazz': 0,
        'reggae': 1,
        'rock': 2,
        'blues': 3,
        'hiphop': 4,
        'country': 5,
        'metal': 6,
        'classical': 7,
        'disco': 8,
        'pop': 9
    }
    y = labels.map(label_dict).values
    
    # Returning the mel spectrograms and labels
    return X, y

X, y = extract_mel_spectrogram('./db/blues')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.2)

X_train.min()

X_train /= X_train.min()
X_test /= X_train.min()

X_train = X_train.reshape(X_train.shape[0], 128, 660, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 660, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

np.random.seed(100)
tf.random.set_seed(100)

cnn_model = Sequential(name='RedConvolucional')

cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(128,660,1)))

cnn_model.add(MaxPooling2D(pool_size=(2,4)))

cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2,4)))

cnn_model.add(Flatten())

cnn_model.add(Dense(64, activation='relu'))

cnn_model.add(Dropout(0.25))

cnn_model.add(Dense(10, activation='softmax'))

cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = cnn_model.fit(X_train,
                        y_train, 
                        batch_size=16,
                        validation_data=(X_test, y_test),
                        epochs=15)

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(12, 8))

plt.plot(train_loss, label='Pérdida de entrenamiento', color='blue')
plt.plot(test_loss, label='Pérdida de pruebas', color='red')

plt.title('Pruebas y entrenamiento por época', fontsize = 25)
plt.xlabel('Época', fontsize = 18)
plt.ylabel('Entropía cruzada', fontsize = 18)
plt.xticks(range(1,16), range(1,16))

plt.legend(fontsize = 18);


train_loss = history.history['accuracy']
test_loss = history.history['val_accuracy']

plt.figure(figsize=(12, 8))

plt.plot(train_loss, label='Precision de entrenamiento', color='blue')
plt.plot(test_loss, label='Precision de pruebas', color='red')

plt.title('Presición por entrenamiento', fontsize = 25)
plt.xlabel('Entrenamiento', fontsize = 18)
plt.ylabel('Precisión', fontsize = 18)
plt.xticks(range(1,21), range(1,21))

plt.legend(fontsize = 18);

predictions = cnn_model.predict(X_test, verbose=1)

for i in range(10): 
    print(f'{i}: {sum([1 for target in y_test if target[i] == 1])}')

for i in range(10): 
    print(f'{i}: {sum([1 for prediction in predictions if np.argmax(prediction) == i])}')

conf_matrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(predictions, 1))
conf_matrix

confusion_df = pd.DataFrame(conf_matrix)

labels_dict = {
    0: 'jazz',
    1: 'reggae',
    2: 'rock',
    3: 'blues',
    4: 'hiphop',
    5: 'country',
    6: 'metal',
    7: 'classical',
    8: 'disco',
    9: 'pop'
}

