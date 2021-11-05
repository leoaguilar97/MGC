import os
import librosa
import pandas as pd
import numpy as np

def espectrograma_mel():   
    labels = []
    mel_specs = []

    directorios = ['blues', 'classical', 'country', 'jazz', 'rock']

    for directorio in directorios: 
        actual = f"./db/{directorio}"

        for file in os.scandir(actual):
            y, sr = librosa.core.load(file)           
            labels.append(directorio)
            
            S = librosa.feature.melspectrogram(y=y, sr=sr) #hop_length=1024)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            if S_db.shape[1] != 657:
                S_db.resize(128, 657, refcheck=False)

            mel_specs.append(S_db)
    
    X = np.array(mel_specs)
    
    labels = pd.Series(labels)
    label_dict = {
        'blues': 0,
        'classical': 1,
        'country': 2,
        'jazz': 3,
        'rock': 4
    }
    y = labels.map(label_dict).values
    
    return X, y