import argparse
import os
import json
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import joblib


# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# -------------------------
# AUDIO LOADING
# -------------------------
def _load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    # Match training sample rate (16 kHz) so features align with saved models
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr


# -------------------------
# FEATURE EXTRACTION
# -------------------------
def extract_features(audio_path: str) -> np.ndarray:
    y, sr = _load_audio(audio_path)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)

    # Pitch
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size > 0 else 0.0

    # Energy (RMS mean to match training)
    energy = float(np.mean(librosa.feature.rms(y=y))) if len(y) > 0 else 0.0

    # Final feature vector
    features = np.concatenate([mfcc_means, [pitch_mean, energy]])
    return features.reshape(1, -1)


# -------------------------
# MODEL LOADING (LAZY)
# -------------------------
_scaler_reg = None
_scaler_clf = None
_regression_model = None
_classifier_model = None
_feature_names = None


def _load_model(path: Path):
    return joblib.load(path)


def _ensure_models_loaded():
    global _scaler_reg, _scaler_clf, _regression_model, _classifier_model, _feature_names

    if _scaler_reg is None:
        _scaler_reg = _load_model(MODELS_DIR / "scaler_reg.pkl")

    if _regression_model is None:
        _regression_model = _load_model(MODELS_DIR / "regression_model.pkl")

    if _scaler_clf is None:
        _scaler_clf = _load_model(MODELS_DIR / "scaler_clf.pkl")

    if _classifier_model is None:
        _classifier_model = _load_model(MODELS_DIR / "classifier_model.pkl")

    if _feature_names is None:
        _feature_names = json.load(open(MODELS_DIR / "feature_names.json"))


# -------------------------
# PREDICT FUNCTIONS
# -------------------------
def predict_sleep_score(audio_path: str) -> float:
    _ensure_models_loaded()
    feats = extract_features(audio_path)

    # Convert to DataFrame → removes warning
    df = pd.DataFrame(feats, columns=_feature_names)
    feats_scaled = _scaler_reg.transform(df)

    pred = _regression_model.predict(feats_scaled)
    return float(pred[0])


def predict_sleep_quality(audio_path: str) -> int:
    _ensure_models_loaded()
    feats = extract_features(audio_path)

    # Convert to DataFrame → removes warning
    df = pd.DataFrame(feats, columns=_feature_names)
    feats_scaled = _scaler_clf.transform(df)

    pred = _classifier_model.predict(feats_scaled)
    return int(pred[0])


# -------------------------
# MAIN (CLI)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict sleep score from audio.")
    parser.add_argument("audio_path", help="Path to audio file")
    args = parser.parse_args()

    audio_path = args.audio_path
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    score = predict_sleep_score(audio_path)
    quality = predict_sleep_quality(audio_path)

    print(f"Predicted sleep score (regression): {score:.3f}")
    print(f"Predicted sleep quality (0=poor, 1=good): {quality}")


if __name__ == "__main__":
    main()
