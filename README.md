# Voice-Sleep Analysis Project

A machine learning project for analyzing voice recordings and sleep patterns.

## Project Structure

```
voice-sleep/
│
├── data/
│   ├── raw/          # Original WAV audio files
│   ├── processed/    # Extracted features (CSV)
│
├── src/
│   ├── extract_features.py  # Feature extraction from audio
│   ├── train_models.py      # Model training
│   ├── evaluate.py          # Model evaluation
│
├── notebook/
│   └── analysis.ipynb       # Exploratory analysis and visualization
│
├── requirements.txt
└── README.md
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your raw WAV files in `data/raw/`
2. Extract features: `python src/extract_features.py`
3. Train models: `python src/train_models.py`
4. Evaluate results: `python src/evaluate.py`

## Requirements

- numpy
- scipy
- librosa
- pandas
- scikit-learn
- matplotlib
- soundfile

## License

TBD
