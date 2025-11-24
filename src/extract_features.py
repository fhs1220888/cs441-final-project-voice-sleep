import librosa
import numpy as np
import pandas as pd
import os

RAW_DIR = "../data/raw/"
OUT_FILE = "../data/processed/features.csv"

def extract_features_from_file(path):
    y, sr = librosa.load(path, sr=16000)

    # MFCC (13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Pitch (fundamental frequency)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    features = {
        "mfcc_" + str(i): mfcc[i] for i in range(13)
    }
    features["pitch_mean"] = pitch_mean
    features["energy"] = energy

    return features

def main():
    rows = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".wav"):
            fullpath = os.path.join(RAW_DIR, filename)
            features = extract_features_from_file(fullpath)

            label = int(filename.split("_")[0].replace("sleep", ""))

            features["label"] = label
            features["file"] = filename
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print("Saved features:", OUT_FILE)

if __name__ == "__main__":
    main()
