import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import os
import io

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EEG Brain Authentication",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .title-text {
        font-size: 3em;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    .subtitle-text {
        font-size: 1.2em;
        color: #ffffff;
        text-align: center;
        margin-top: 10px;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #00d4ff;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: rgba(0, 212, 255, 0.1);
        border: 2px solid #00d4ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .success-text {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1em;
    }
    .metric-card {
        background-color: rgba(0, 100, 150, 0.2);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CUSTOM ATTENTION LAYER
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def build_model(n_classes=24, n_samples=256, n_channels=4, dropout=0.4, lr=0.001):
    """Build CNN + BiLSTM + Attention model"""
    inputs = Input(shape=(n_samples, n_channels), name='eeg_input')
    x = Conv1D(64, kernel_size=5, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(64, kernel_size=5, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv1D(128, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)

    x = AttentionLayer(name='attention')(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout * 0.5)(x)
    outputs = Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL WEIGHTS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_weights():
    """Load trained model"""
    model = build_model()
    try:
        model.load_weights("best_model.weights.h5")
        return model
    except:
        try:
            # Try loading as full model if weights format fails
            model = keras.models.load_model("best_model.h5", custom_objects={'AttentionLayer': AttentionLayer})
            return model
        except:
            st.error("Model weights not found. Please ensure 'best_model.weights.h5' or 'best_model.h5' exists.")
            return None


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════
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

def load_and_preprocess(file_data, target_length=256):
    """Load CSV and preprocess"""
    try:
        df = pd.read_csv(file_data, index_col=0)
        signal = df.values.astype(np.float64)

        if signal.shape[1] > 4:
            signal = signal[:, :4]

        filtered = bandpass_filter(signal, low=0.5, high=40.0, fs=128)
        normalized = normalize_signal(filtered)

        if normalized.shape[0] < target_length:
            pad_amount = target_length - normalized.shape[0]
            normalized = np.vstack([normalized, np.zeros((pad_amount, 4))])
        else:
            normalized = normalized[:target_length, :]

        signal_reshaped = normalized.reshape(1, target_length, 4).astype(np.float32)
        return signal_reshaped, normalized
    except Exception as e:
        return None, None


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="title-text">🧠 EEG Brain Authentication</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Identify individuals using EEG brainwave patterns</div>', unsafe_allow_html=True)

st.markdown("---")

# Load model
model = load_model_weights()
if model is None:
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Upload CSV", "Sample Data"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: UPLOAD CSV
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📤 Upload Your EEG Data</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload EEG CSV file (4 channels: T7, F8, Cz, P4)"
        )

    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")

        # Process the file
        signal_input, normalized_signal = load_and_preprocess(uploaded_file)

        if signal_input is not None:
            # Make prediction
            with st.spinner("🔄 Analyzing EEG data..."):
                predictions = model.predict(signal_input, verbose=0)

            predicted_subject = np.argmax(predictions[0])
            confidence = predictions[0][predicted_subject]

            # Display results
            st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Predicted Subject", f"S{predicted_subject+1:02d}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                # Calculate authentication status
                AUTH_THRESHOLD = 0.5
                if confidence > AUTH_THRESHOLD:
                    st.metric("Status", "✅ AUTHENTIC", delta="VERIFIED")
                else:
                    st.metric("Status", "❌ SUSPICIOUS", delta="UNVERIFIED")
                st.markdown('</div>', unsafe_allow_html=True)

            # Top 5 predictions
            st.markdown('<div class="section-header">🎯 Top 5 Predictions</div>', unsafe_allow_html=True)

            top_5_idx = np.argsort(predictions[0])[::-1][:5]
            top_5_data = []

            for rank, idx in enumerate(top_5_idx, 1):
                top_5_data.append({
                    'Rank': rank,
                    'Subject': f'S{idx+1:02d}',
                    'Confidence': f"{predictions[0][idx]*100:.2f}%",
                    'Score': predictions[0][idx]
                })

            df_top5 = pd.DataFrame(top_5_data)
            st.dataframe(df_top5, hide_index=True, use_container_width=True)

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">📈 Confidence Distribution</div>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                top_10_idx = np.argsort(predictions[0])[::-1][:10]
                top_10_scores = predictions[0][top_10_idx]
                top_10_labels = [f'S{i+1:02d}' for i in top_10_idx]

                colors = ['#00ff88' if i == 0 else '#00d4ff' for i in range(len(top_10_idx))]
                ax.barh(top_10_labels, top_10_scores * 100, color=colors)
                ax.set_xlabel('Confidence (%)', fontsize=11)
                ax.set_title('Top 10 Subject Matches', fontsize=12, fontweight='bold', color='#00d4ff')
                ax.set_facecolor('#0f1419')
                fig.patch.set_facecolor('#0f1419')
                ax.spines['bottom'].set_color('#00d4ff')
                ax.spines['left'].set_color('#00d4ff')
                ax.tick_params(colors='#ffffff')

                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">🌊 EEG Waveform (First 256 samples)</div>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                time = np.linspace(0, 2, 256)
                channels = ['T7', 'F8', 'Cz', 'P4']
                colors_ch = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa07a']

                for i, (ch, color) in enumerate(zip(channels, colors_ch)):
                    ax.plot(time, normalized_signal[:256, i], label=ch, linewidth=2, color=color, alpha=0.8)

                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('Amplitude (normalized)', fontsize=11)
                ax.set_title('EEG Signal Channels', fontsize=12, fontweight='bold', color='#00d4ff')
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(alpha=0.3, color='#00d4ff')
                ax.set_facecolor('#0f1419')
                fig.patch.set_facecolor('#0f1419')
                ax.spines['bottom'].set_color('#00d4ff')
                ax.spines['left'].set_color('#00d4ff')
                ax.tick_params(colors='#ffffff')

                st.pyplot(fig, use_container_width=True)
        else:
            st.error("❌ Failed to process file. Ensure it's a valid EEG CSV with 4 channels.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: SAMPLE DATA
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎬 Sample EEG Files</div>', unsafe_allow_html=True)
    st.info("Click on any sample below to test the model instantly")

    # Get sample files
    data_folder = "Filtered_Data"
    sample_files = []

    if os.path.exists(data_folder):
        csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
        sample_files = csv_files[:5]  # Get first 5 samples

    if sample_files:
        cols = st.columns(5)

        for col_idx, (col, sample_file) in enumerate(zip(cols, sample_files)):
            with col:
                if st.button(
                    f"📊 {sample_file[:15]}...",
                    key=f"sample_{col_idx}",
                    use_container_width=True,
                    help=f"Test with {sample_file}"
                ):
                    # Process sample
                    sample_path = os.path.join(data_folder, sample_file)

                    with st.spinner(f"🔄 Analyzing {sample_file}..."):
                        signal_input, normalized_signal = load_and_preprocess(
                            open(sample_path, 'rb')
                        )

                    if signal_input is not None:
                        predictions = model.predict(signal_input, verbose=0)
                        predicted_subject = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_subject]

                        st.success("✅ Analysis Complete!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted", f"S{predicted_subject+1:02d}")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        with col3:
                            AUTH_THRESHOLD = 0.5
                            status = "✅ VERIFIED" if confidence > AUTH_THRESHOLD else "❌ UNVERIFIED"
                            st.metric("Status", status)

                        # Display top 5
                        st.markdown("**Top 5 Matches:**")
                        top_5_idx = np.argsort(predictions[0])[::-1][:5]
                        for rank, idx in enumerate(top_5_idx, 1):
                            st.write(f"{rank}. **S{idx+1:02d}** - {predictions[0][idx]*100:.2f}%")
    else:
        st.warning("⚠️ No sample files found. Please ensure the 'Filtered_Data' folder exists.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888888; padding: 20px;'>
    <p>🧠 EEG Brain Authentication System | CNN + BiLSTM + Attention Model</p>
    <p style='font-size: 0.9em;'>PhysioNet Auditory EEG Dataset | 24 Subjects | 88% Accuracy</p>
</div>
""", unsafe_allow_html=True)
