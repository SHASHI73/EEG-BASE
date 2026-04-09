# 🧠 EEG Brain Authentication System

A Streamlit web application for identifying individuals using EEG brainwave patterns. This system uses a CNN + BiLSTM + Attention deep learning model trained on the PhysioNet Auditory EEG Biometric Dataset.

## Features

✅ **Upload CSV Files** - Test the model with your own EEG data
✅ **5 Sample Files** - Pre-loaded test samples for quick testing
✅ **Live Predictions** - Real-time subject identification with confidence scores
✅ **Visualizations** - EEG waveforms and confidence distribution charts
✅ **Authentication Status** - Verified/Unverified classification

## Model Performance

- **Accuracy**: 88.11%
- **EER**: 3.64%
- **Architecture**: CNN + Bidirectional LSTM + Attention
- **Dataset**: 24 subjects, 2232 epochs from PhysioNet Auditory EEG
- **Input Shape**: (256 samples, 4 channels)
- **Channels**: T7, F8, Cz, P4

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit tensorflow numpy pandas matplotlib scipy scikit-learn
```

### Step 2: Ensure Model File Exists

Make sure `best_model.weights.h5` is in the same directory as `app.py`

### Step 3: Ensure Data Folder Exists

The `Filtered_Data` folder should contain sample CSV files for testing

```
Brain_Auth/
├── app.py
├── best_model.weights.h5
├── requirements.txt
├── Filtered_Data/
│   ├── s01_ex01_s01.csv
│   ├── s01_ex01_s02.csv
│   ├── s01_ex01_s03.csv
│   ├── s01_ex02_s01.csv
│   └── ... (more CSV files)
```

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

## Usage

### Method 1: Upload CSV File
1. Go to the **"Upload CSV"** tab
2. Click the file uploader
3. Select an EEG CSV file from your computer
4. Results will display automatically

### Method 2: Use Sample Data
1. Go to the **"Sample Data"** tab
2. Click any of the 5 sample buttons
3. Results will display instantly

## CSV File Format

Your CSV file should have the following structure:

```
,T7,F8,Cz,P4
0,533.956726,-1017.742737,434.163116,345.218933
1,516.659729,-1034.703125,421.286652,339.302429
...
```

- **4 channels**: T7, F8, Cz, P4
- **Sampling Rate**: 128 Hz
- **Recommended Length**: 256+ samples (~2 seconds of data)

## Output Interpretation

### Prediction Results
- **Predicted Subject**: Which subject the EEG pattern matches (S01-S24)
- **Confidence**: How certain the model is (0-100%)
- **Status**: 
  - ✅ **VERIFIED** if confidence > 50%
  - ❌ **UNVERIFIED** if confidence ≤ 50%

### Visualizations
- **Confidence Distribution**: Top 10 matching subjects
- **EEG Waveform**: Raw signal across all 4 channels

## Model Architecture

```
Input (256, 4)
    ↓
CNN Block 1 (64 filters)
    ↓
CNN Block 2 (128 filters)
    ↓
CNN Block 3 (256 filters)
    ↓
BiLSTM (128 + 64 units)
    ↓
Attention Layer
    ↓
Dense Layers (256, 128)
    ↓
Output (24 classes)
```

## Troubleshooting

### Model weights not found
```
Error: Model weights not found. Please ensure 'best_model.weights.h5' exists.
```
**Solution**: Ensure `best_model.weights.h5` is in the same directory as `app.py`

### No sample files found
```
Warning: No sample files found. Please ensure the 'Filtered_Data' folder exists.
```
**Solution**: Create a `Filtered_Data` folder and add your CSV files to it

### CSV processing failed
```
Error: Failed to process file. Ensure it's a valid EEG CSV with 4 channels.
```
**Solution**: Check that your CSV has exactly 4 columns (T7, F8, Cz, P4) and numeric values

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository and main file (`app.py`)
5. Click "Deploy"

## File Structure

```
Brain_Auth/
├── app.py                      # Main Streamlit application
├── test_model.py              # Standalone testing script
├── main.py                    # Data preparation script
├── requirements.txt           # Python dependencies
├── best_model.weights.h5      # Trained model weights
├── Filtered_Data/             # Sample EEG files
│   ├── s01_ex01_s01.csv
│   ├── s01_ex01_s02.csv
│   ├── ... (more files)
└── results/                   # Training visualizations
    ├── training_history.png
    ├── confusion_matrix.png
    └── ... (more plots)
```

## Performance Metrics

| Metric | SVM Baseline | CNN + BiLSTM + Attention |
|--------|-------------|------------------------|
| Accuracy | 29.45% | **88.11%** |
| EER (%) | 19.65% | **3.64%** |
| FAR (%) | 23.61% | **3.29%** |
| FRR (%) | 13.86% | **3.99%** |

## Citation

If you use this project, please cite:

```
@dataset{physionet_auditory_eeg,
  title={Auditory Evoked Potential EEG Biometric Dataset},
  publisher={PhysioNet},
  url={https://physionet.org/}
}
```

## License

This project is for educational and research purposes.

## Contact & Support

For issues or questions, please check:
- Console output for error messages
- CSV file format requirements
- Model file existence and integrity

---

**Happy Brain Authentication! 🧠✨**
