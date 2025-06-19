import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

SAMPLE_RATE = 22050
AUDIO_DURATION = 3
MFCC_FEATURES = 40
BATCH_SIZE = 16
EPOCHS = 30
DATASET_PATH = r"C:\Users\user\Desktop\sss\archive"

def extract_label(file_name):
    return 0 if file_name.split('-')[-1].split('.')[0] == '01' else 1

def preprocess_audio(file_path, sr, duration, n_mfcc):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        max_len = sr * duration
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T
    except Exception as e:
        print(f"{file_path} error: {e}")
        return None

X, y = [], []
print("Loading data...")
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        label = extract_label(file)
        path = os.path.join(DATASET_PATH, file)
        mfcc = preprocess_audio(path, SAMPLE_RATE, AUDIO_DURATION, MFCC_FEATURES)
        if mfcc is not None:
            X.append(mfcc)
            y.append(label)

max_timesteps = max([m.shape[0] for m in X])
X_padded = np.zeros((len(X), max_timesteps, MFCC_FEATURES))
for i, mfcc in enumerate(X):
    X_padded[i, :mfcc.shape[0], :] = mfcc

X = np.array(X_padded)
y = np.array(y)
print(f"Feature shape: {X.shape}, Label count: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, input_shape=(max_timesteps, MFCC_FEATURES)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
print("Saved training_results.png")

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_labels)
ConfusionMatrixDisplay(cm, display_labels=["Fall", "Non-Fall"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

model.save("fall_detection_model.h5")
print("Saved fall_detection_model.h5")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open("fall_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("TFLite model saved as fall_detection_model.tflite")
except Exception as e:
    print(f"TFLite conversion error: {e}")

print("Running TFLite model test...")
try:
    interpreter = tf.lite.Interpreter(model_path="fall_detection_model.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    correct = 0
    for i in range(len(X_test)):
        input_tensor = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        pred = 1 if output[0][0] > 0.5 else 0
        if pred == y_test[i]:
            correct += 1
    tflite_acc = correct / len(X_test)
    print(f"TFLite test accuracy: {tflite_acc:.4f}")
except Exception as e:
    print(f"TFLite inference error: {e}")

try:
    import tf2onnx
    import onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model,
                                               input_signature=[tf.TensorSpec((1, max_timesteps, MFCC_FEATURES), tf.float32)],
                                               opset=13)
    onnx.save(onnx_model, "fall_detection_model.onnx")
    print("ONNX model saved as fall_detection_model.onnx")
except Exception as e:
    print(f"ONNX conversion failed: {e}")





