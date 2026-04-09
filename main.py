import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
def load_data(data_path):
    data = []
    labels = []

    files = os.listdir(data_path)
    print(f"Found {len(files)} files")

    for file in files:
        if file.endswith(".csv"):
            try:
                # Extract subject ID (S01 → 0, S02 → 1, ... S24 → 23)
                subject_id = int(file.split('_')[0][1:]) - 1
            except:
                continue

            file_path = os.path.join(data_path, file)

            try:
                df = pd.read_csv(file_path, index_col=0)
                signal = df.values.astype(np.float32)
            except Exception:
                continue

            # Skip invalid signals
            if signal.ndim < 2:
                continue

            # Cap signal to 256 samples for consistency
            signal = signal[:256, :]

            data.append(signal)
            labels.append(subject_id)

    return data, np.array(labels)


# -------------------------------
# STEP 2: CREATE PAIRS
# -------------------------------
def create_pairs(data, labels):
    pairs = []
    pair_labels = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        idx = np.where(labels == label)[0]

        # -------- Genuine pairs --------
        for i in range(len(idx) - 1):
            for j in range(i + 1, len(idx)):
                pairs.append([data[idx[i]], data[idx[j]]])
                pair_labels.append(1)

        # -------- Imposter pairs --------
        other_labels = unique_labels[unique_labels != label]

        for i in idx:
            neg_label = np.random.choice(other_labels)
            neg_idx = np.random.choice(np.where(labels == neg_label)[0])

            pairs.append([data[i], data[neg_idx]])
            pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)


# -------------------------------
# STEP 3: MAIN EXECUTION
# -------------------------------
# 👉 Use this (since your folder is inside project)
DATA_PATH = "Filtered_Data"

print("\n[INFO] Loading data...")
data, labels = load_data(DATA_PATH)

print("Total samples:", len(data))
print("Total subjects:", len(np.unique(labels)))

# Safety check
if len(data) == 0:
    print("[ERROR] No data loaded. Check your path or files.")
    exit()

# -------------------------------
# STEP 4: FIX SIGNAL LENGTH
# -------------------------------
print("\n[INFO] Padding signals...")
data = pad_sequences(data, padding='post', dtype='float32')

print("Data shape after padding:", data.shape)

# -------------------------------
# STEP 5: TRAIN-TEST SPLIT
# -------------------------------
print("\n[INFO] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# -------------------------------
# STEP 6: CREATE PAIRS
# -------------------------------
print("\n[INFO] Creating training pairs...")
train_pairs, train_labels = create_pairs(X_train, y_train)

print("[INFO] Creating testing pairs...")
test_pairs, test_labels = create_pairs(X_test, y_test)

print("Train pairs:", len(train_pairs))
print("Test pairs:", len(test_pairs))

# -------------------------------
# STEP 7: SAVE DATA
# -------------------------------
print("\n[INFO] Saving pairs...")

os.makedirs("pairs", exist_ok=True)

np.save("pairs/train_pairs.npy", train_pairs)
np.save("pairs/train_labels.npy", train_labels)

np.save("pairs/test_pairs.npy", test_pairs)
np.save("pairs/test_labels.npy", test_labels)

print("\n[DONE] Dataset preparation COMPLETE!")
import os

for root, dirs, files in os.walk("Filtered_Data"):
    for file in files[:10]:
        print(file)
    break