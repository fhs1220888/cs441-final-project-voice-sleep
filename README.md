# CS441 Final Project — Voice & Sleep Quality Prediction

A machine learning project for predicting sleep quality from short morning voice recordings.

This project extracts audio features (MFCC, pitch, energy), trains classical ML models (Linear Regression, Logistic Regression, kNN), and evaluates model performance with RMSE and accuracy metrics.

---

## Project Structure


## Project Structure

```text
voice-sleep/
│
├── data/
│   ├── raw/                        # Raw WAV audio files (NOT included in the repo)
│   └── processed/                  # Extracted features, evaluation results, plots
│
├── src/
│   ├── extract_features.py         # Convert audio → MFCC/pitch/energy features
│   ├── train_models.py             # Baseline model training
│   ├── evaluate.py                 # K-fold evaluation (RMSE & accuracy)
│   └── evaluate_with_plots.py      # Evaluation + automatic visualization
│
├── notebook/
│   └── analysis.ipynb              # Exploratory analysis & feature visualization
│
├── requirements.txt
└── README.md
```


---

## Setup & Installation

### **1. Clone the repository**

```
git clone https://github.com/fhs1220888/cs441-final-project-voice-sleep.git

cd cs441-final-project-voice-sleep
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

---

## Data Collection

This project requires **your own recorded audio data** to satisfy CS441's “self-collected dataset” requirement.

- Record a 3–5 second voice clip every morning
- Say a fixed sentence (e.g., *“Today is November 23rd, I slept okay last night.”*)
- Label your sleep quality on a **1–5 scale**
- Save files in:

```
data/raw/sleep<score>_<num>.wav
```

Example:

```
data/raw/sleep4_1.wav
```

---

## Running the Pipeline

### **1. Extract audio features**

Extract MFCC, pitch, energy, etc.:

```
python src/extract_features.py
```

This generates:

```
data/processed/features.csv
```

---

### **2. Train baseline models**

```
python src/train_models.py
```

Models included:

- Linear Regression
- Logistic Regression
- k-Nearest Neighbors (kNN)

---

### **3. Evaluate models (K-Fold CV)**

```
python src/evaluate.py
```

Outputs:

- RMSE (regression)
- Accuracy (classification)
- eval_results.csv

---

### **4. Generate plots**

```
python src/evaluate_with_plots.py
```

Produces:

```
data/processed/rmse_plot.png
data/processed/accuracy_plot.png
```

---

## Exploratory Analysis

Use the Jupyter notebook for visualization:

```
notebook/analysis.ipynb
```

Includes:

- MFCC inspection
- Sleep label distribution
- Spectrogram visualization
- Feature trends

---

## Requirements

```
numpy
scipy
librosa
pandas
scikit-learn
matplotlib
soundfile
```

---

## License

MIT

---

## Notes

This project fulfills CS441 Final Project requirements:

- Self-collected dataset
- ML task definition
- Training + validation + testing
- Quantitative evaluation
- Visualization + analysis
