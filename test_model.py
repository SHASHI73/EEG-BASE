import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    AveragePooling1D, Dropout, Dense,
    LSTM, Bidirectional
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------
# CUSTOM ATTENTION LAYER (from notebook)
# -----------------------------------------------
class AttentionLayer(keras.layers.Layer):
    """Custom attention layer — compatible with all TF versions."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = K.tanh(K.dot(x, self.W) + self.b)
        score = K.softmax(score, axis=1)
        out = x * score
        out = K.sum(out, axis=1)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# -----------------------------------------------
# BUILD MODEL ARCHITECTURE
# -----------------------------------------------
def build_model(n_classes=24, n_samples=256, n_channels=4, dropout=0.4, lr=0.001):
    """CNN + Bidirectional LSTM + Attention"""
    inputs = Input(shape=(n_samples, n_channels), name='eeg_input')

    # CNN Block 1
    x = Conv1D(64, kernel_size=5, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(64, kernel_size=5, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # CNN Block 2
    x = Conv1D(128, kernel_size=3, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(128, kernel_size=3, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # CNN Block 3
    x = Conv1D(256, kernel_size=3, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.1))(x)
    x = Bidirectional(LSTM(64, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.1))(x)

    # Attention
    x = AttentionLayer(name='attention')(x)

    # Classification Head
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout * 0.5)(x)
    outputs = Dense(n_classes, activation='softmax',
                    name='output')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='CNN_BiLSTM_Attention')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -----------------------------------------------
# PREPROCESSING FUNCTIONS
# -----------------------------------------------
def bandpass_filter(data, low=0.5, high=40.0, fs=128, order=4):
    """Apply bandpass filter"""
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    lo = np.clip(low / nyq, 1e-6, 0.999)
    hi = np.clip(high / nyq, 1e-6, 0.999)
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, data, axis=0)

def normalize_signal(data):
    """Normalize signal"""
    data = np.asarray(data, dtype=np.float64)
    mu = np.mean(data, axis=0, keepdims=True)
    sig = np.std(data, axis=0, keepdims=True) + 1e-8
    return (data - mu) / sig


def load_and_preprocess(file_path, target_length=256):
    """Load CSV and preprocess"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        signal = df.values.astype(np.float64)

        # Keep first 4 channels only
        if signal.shape[1] > 4:
            signal = signal[:, :4]

        print(f"[INFO] Raw signal shape: {signal.shape}")

        # Filter
        filtered = bandpass_filter(signal, low=0.5, high=40.0, fs=128)
        print(f"[INFO] After filter: {filtered.shape}")

        # Normalize
        normalized = normalize_signal(filtered)
        print(f"[INFO] After normalization - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")

        # Reshape to match model input (1, target_length, 4)
        if normalized.shape[0] < target_length:
            # Pad if too short
            pad_amount = target_length - normalized.shape[0]
            normalized = np.vstack([normalized, np.zeros((pad_amount, 4))])
        else:
            # Take first target_length samples
            normalized = normalized[:target_length, :]

        signal_reshaped = normalized.reshape(1, target_length, 4).astype(np.float32)
        print(f"[INFO] Final shape for model: {signal_reshaped.shape}")

        return signal_reshaped

    except Exception as e:
        print(f"[ERROR] Failed to load/preprocess: {e}")
        return None


# -----------------------------------------------
# TEST MODEL
# -----------------------------------------------
def test_model(model_path, eeg_file_path):
    """Load model and test on EEG data"""

    print("\n" + "="*60)
    print("[INFO] Building model architecture...")
    model = build_model(n_classes=24, n_samples=256, n_channels=4)
    print("[SUCCESS] Model built!")

    print(f"\n[INFO] Loading weights from: {model_path}")
    try:
        model.load_weights(model_path)
        print("[SUCCESS] Weights loaded!")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return

    print(f"\n[INFO] Loading EEG data from: {eeg_file_path}")
    signal = load_and_preprocess(eeg_file_path, target_length=256)

    if signal is None:
        return

    print("\n[INFO] Making prediction...")
    predictions = model.predict(signal, verbose=0)
    predicted_subject = np.argmax(predictions[0])
    confidence = predictions[0][predicted_subject]

    print("\n" + "="*60)
    print("[RESULTS]")
    print("="*60)
    print(f"Predicted Subject ID: S{predicted_subject+1:02d}")
    print(f"Confidence Score: {confidence*100:.2f}%")
    print(f"\nTop 5 Predictions:")
    top_5_idx = np.argsort(predictions[0])[::-1][:5]
    for rank, idx in enumerate(top_5_idx, 1):
        print(f"  {rank}. S{idx+1:02d} - {predictions[0][idx]*100:.2f}%")
    print("="*60)

    return predictions


# -----------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------
if __name__ == "__main__":
    MODEL_PATH = "best_model.weights.h5"
    DATA_FOLDER = "Filtered_Data"

    # Auto-find first CSV file
    if os.path.exists(DATA_FOLDER):
        csv_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')])

        if csv_files:
            eeg_file = csv_files[0]
            EEG_FILE_PATH = os.path.join(DATA_FOLDER, eeg_file)

            print(f"\n[SUCCESS] Found dataset file: {eeg_file}")
            print(f"[INFO] Dataset has {len(csv_files)} total files")

            test_model(MODEL_PATH, EEG_FILE_PATH)
        else:
            print(f"[ERROR] No CSV files in {DATA_FOLDER}")
    else:
        print(f"[ERROR] Folder {DATA_FOLDER} not found")
