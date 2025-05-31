import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 參數設定
SAMPLE_RATE = 22050  # 統一使用 22050 Hz
AUDIO_DURATION = 3   # 每段音訊長度（秒）
MFCC_FEATURES = 40   # 提取多少 MFCC

DATASET_PATH = r'C:\Users\user\Desktop\Sound\data' # 修改成你的資料集路徑

def extract_label(file_name):
    """
    根據 SAFE 資料集的命名規則提取標籤
    0 = fall, 1 = non-fall
    """
    label_str = file_name.split('-')[-1].split('.')[0]
    return 0 if label_str == '01' else 1

def preprocess_audio(file_path, sample_rate, duration, n_mfcc):
    """
    讀取音訊檔，轉成 MFCC，並補齊或截斷到固定長度。
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        max_length = sample_rate * duration
        if len(y) > max_length:
            y = y[:max_length]
        else:
            y = np.pad(y, (0, max_length - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # 轉置以符合 LSTM 輸入
        return mfcc
    except Exception as e:
        print(f"⚠️ 無法處理 {file_path}: {e}")
        return None

# 讀取所有檔案
X = []
y = []

print("🔄 開始讀取資料...")

for file_name in os.listdir(DATASET_PATH):
    if file_name.endswith(".wav"):
        label = extract_label(file_name)
        file_path = os.path.join(DATASET_PATH, file_name)
        features = preprocess_audio(file_path, SAMPLE_RATE, AUDIO_DURATION, MFCC_FEATURES)
        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ 資料集大小: {X.shape}, 標籤: {y.shape}")

# 將資料填齊同樣的時間維度
max_timesteps = max([mfcc.shape[0] for mfcc in X])
X_padded = np.zeros((len(X), max_timesteps, MFCC_FEATURES))
for i, mfcc in enumerate(X):
    timesteps = mfcc.shape[0]
    X_padded[i, :timesteps, :] = mfcc

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 建立 LSTM 模型
model = Sequential([
    LSTM(64, input_shape=(max_timesteps, MFCC_FEATURES)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 訓練
try:
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
except Exception as e:
    print(f"⚠️ 模型訓練錯誤: {e}")

# 評估
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ 測試集準確率: {accuracy:.4f}")

# 視覺化訓練結果
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("✅ 已儲存訓練過程圖表：training_results.png")

# 儲存模型
model.save("fall_detection_model.h5")
print("✅ 模型已儲存為 fall_detection_model.h5")

# 轉換成 TFLite
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open("fall_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ TFLite 模型已儲存為 fall_detection_model.tflite")
except Exception as e:
    print(f"⚠️ TFLite 轉換失敗: {e}")
