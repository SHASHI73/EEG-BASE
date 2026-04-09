import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    AveragePooling1D, Dropout, Dense, LSTM, Bidirectional
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        score = K.tanh(K.dot(x, self.W) + self.b)
        score = K.softmax(score, axis=1)
        out = x * score
        return K.sum(out, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model():
    inputs = Input(shape=(256, 4))
    x = Conv1D(64, 5, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(64, 5, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = AttentionLayer()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(24, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def bandpass_filter(data, low=0.5, high=40.0, fs=128):
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    lo = np.clip(low / nyq, 1e-6, 0.999)
    hi = np.clip(high / nyq, 1e-6, 0.999)
    b, a = butter(4, [lo, hi], btype='band')
    return filtfilt(b, a, data, axis=0)

def normalize_signal(data):
    data = np.asarray(data, dtype=np.float64)
    mu = np.mean(data, axis=0, keepdims=True)
    sig = np.std(data, axis=0, keepdims=True) + 1e-8
    return (data - mu) / sig

print("[INFO] Building and loading model...")
model = build_model()
model.load_weights('best_model.weights.h5')

print("\n[INFO] Testing on dataset samples...\n")
print("="*80)
print(f"{'File':<25} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Result':<10}")
print("="*80)

data_folder = 'Filtered_Data'
csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

correct = 0
total = 0

# Test first file from each subject (1-24)
for subject_id in range(1, 25):
    # Find first file for this subject
    subject_files = [f for f in csv_files if f.startswith(f's{subject_id:02d}_')]

    if not subject_files:
        continue

    test_file = subject_files[0]
    filepath = os.path.join(data_folder, test_file)

    try:
        df = pd.read_csv(filepath, index_col=0)
        signal = df.values.astype(np.float64)[:256, :4]
        filtered = bandpass_filter(signal)
        normalized = normalize_signal(filtered)
        signal_input = normalized.reshape(1, 256, 4).astype(np.float32)

        predictions = model.predict(signal_input, verbose=0)
        predicted_subject = np.argmax(predictions[0])
        confidence = predictions[0][predicted_subject]

        expected_class = subject_id - 1
        is_correct = predicted_subject == expected_class
        status = "CORRECT" if is_correct else "WRONG"

        if is_correct:
            correct += 1
        total += 1

        print(f"{test_file:<25} S{subject_id:02d}        S{predicted_subject+1:02d}        {confidence*100:>9.2f}%     {status:<10}")

    except Exception as e:
        print(f"{test_file:<25} S{subject_id:02d}        ERROR              {str(e)[:20]:<10}")

print("="*80)
print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "No files tested")
