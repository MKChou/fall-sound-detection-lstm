import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import librosa

SAMPLE_RATE = 22050
AUDIO_DURATION = 3
MFCC_FEATURES = 40
DATASET_PATH = r'C:\Users\user\Desktop\Sound\data'

def extract_label(file_name):
    label_str = file_name.split('-')[-1].split('.')[0]
    return 0 if label_str == '01' else 1

def preprocess_audio(file_path, sample_rate, duration, n_mfcc):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        max_length = sample_rate * duration
        if len(y) > max_length:
            y = y[:max_length]
        else:
            y = np.pad(y, (0, max_length - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T
    except:
        return None

X, y = [], []
for file_name in os.listdir(DATASET_PATH):
    if file_name.endswith(".wav"):
        label = extract_label(file_name)
        path = os.path.join(DATASET_PATH, file_name)
        mfcc = preprocess_audio(path, SAMPLE_RATE, AUDIO_DURATION, MFCC_FEATURES)
        if mfcc is not None:
            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

max_timesteps = max(m.shape[0] for m in X)
X_padded = np.zeros((len(X), max_timesteps, MFCC_FEATURES))
for i, m in enumerate(X):
    X_padded[i, :m.shape[0], :] = m

_, X_test, _, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saved X_test.npy and y_test.npy")

