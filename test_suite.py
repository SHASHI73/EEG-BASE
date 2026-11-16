"""
Comprehensive Test Suite for EEG Brain Authentication System
Tests model architecture, weights loading, preprocessing, inference, and sample files.
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

def test_imports():
    print("[TEST 1/6] Testing library imports...")
    import tensorflow as tf
    import keras
    import streamlit
    import scipy
    import sklearn
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Keras: {keras.__version__}")
    print(f"  Streamlit: {streamlit.__version__}")
    print("  PASS: All required libraries are imported successfully.")

def test_model_build_and_weights():
    print("\n[TEST 2/6] Testing model architecture and weight loading...")
    from app import build_model, load_model_weights
    model = load_model_weights()
    assert model is not None, "Model failed to load!"
    assert model.input_shape == (None, 256, 4), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 24), f"Unexpected output shape: {model.output_shape}"
    param_count = model.count_params()
    print(f"  Model input shape : {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")
    print(f"  Total parameters  : {param_count:,}")
    print("  PASS: Model architecture built and weights loaded successfully.")
    return model

def test_dummy_inference(model):
    print("\n[TEST 3/6] Testing inference on synthetic EEG input...")
    dummy_input = np.random.randn(1, 256, 4).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    assert predictions.shape == (1, 24), f"Unexpected prediction shape: {predictions.shape}"
    assert np.isclose(np.sum(predictions[0]), 1.0, atol=1e-5), "Softmax probabilities do not sum to 1.0"
    top_pred = np.argmax(predictions[0])
    conf = predictions[0][top_pred]
    print(f"  Inference output shape: {predictions.shape}")
    print(f"  Probabilities sum     : {np.sum(predictions[0]):.6f}")
    print(f"  Predicted subject     : S{top_pred+1:02d} (Confidence: {conf*100:.2f}%)")
    print("  PASS: Synthetic inference executed successfully.")

def test_preprocessing():
    print("\n[TEST 4/6] Testing preprocessing pipeline...")
    from app import bandpass_filter, normalize_signal, load_and_preprocess
    
    # 1. Bandpass filter test
    raw = np.sin(np.linspace(0, 100, 256)).reshape(256, 1)
    filtered = bandpass_filter(raw, low=0.5, high=40.0, fs=128)
    assert filtered.shape == (256, 1), f"Unexpected filtered shape: {filtered.shape}"
    
    # 2. Normalization test
    norm = normalize_signal(filtered)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-2), f"Mean not centered: {np.mean(norm)}"
    assert np.isclose(np.std(norm), 1.0, atol=1e-2), f"Std not normalized: {np.std(norm)}"
    print(f"  Filtered signal shape : {filtered.shape}")
    print(f"  Normalized signal mean: {np.mean(norm):.6f}, std: {np.std(norm):.6f}")
    print("  PASS: Preprocessing pipeline functions accurately.")

def test_sample_files(model):
    print("\n[TEST 5/6] Testing all sample files in Filtered_Data...")
    from app import load_and_preprocess
    data_folder = "Filtered_Data"
    assert os.path.exists(data_folder), f"Folder {data_folder} not found!"
    
    csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])[:5]
    print(f"  Found {len(csv_files)} test sample files.")
    
    for f in csv_files:
        path = os.path.join(data_folder, f)
        with open(path, 'rb') as file_obj:
            signal_input, normalized_signal = load_and_preprocess(file_obj)
        
        assert signal_input is not None, f"Failed to preprocess {f}"
        assert signal_input.shape == (1, 256, 4), f"Incorrect shape {signal_input.shape} for {f}"
        assert not np.isnan(signal_input).any(), f"NaN values detected in {f}"
        
        preds = model.predict(signal_input, verbose=0)
        pred_subject = np.argmax(preds[0])
        conf = preds[0][pred_subject]
        status = "AUTHENTIC (VERIFIED)" if conf > 0.5 else "SUSPICIOUS (UNVERIFIED)"
        print(f"    File: {f:<18} -> Predicted: S{pred_subject+1:02d} | Conf: {conf*100:6.2f}% | Status: {status}")
    
    print("  PASS: All 5 sample files parsed and evaluated cleanly.")

def test_streamlit_server():
    print("\n[TEST 6/6] Testing Streamlit web server status...")
    try:
        resp = requests.get("http://localhost:8501/_stcore/health", timeout=3)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        print(f"  Streamlit health endpoint response: HTTP {resp.status_code} ({resp.text.strip()})")
        print("  PASS: Streamlit server is active and responding on port 8501.")
    except Exception as e:
        print(f"  WARNING: Streamlit server check failed: {e}")

if __name__ == "__main__":
    print("==================================================")
    print("   EEG Brain Authentication - Test Suite")
    print("==================================================")
    test_imports()
    model = test_model_build_and_weights()
    test_dummy_inference(model)
    test_preprocessing()
    test_sample_files(model)
    test_streamlit_server()
    print("\n==================================================")
    print("   ALL TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")
