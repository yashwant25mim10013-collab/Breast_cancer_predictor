
# Breast Cancer Prediction Dashboard

An interactive machine-learning web app that predicts whether a breast tumour is **benign** or **malignant** from 10 cell-nucleus measurements, with real-time probabilities, a risk meter, and an explanation of every prediction.

> **Technical Team Task Round: AI/ML Domain, Task 2** (Interactive Prediction Dashboard: model inference + deployment)

 **Live demo:** [Open the app](https://YOUR-APP-NAME.streamlit.app)

> **Educational demo only.** This is not a medical device and must not be used for diagnosis or treatment decisions.

---

## Features

- **Real-time inference:** the prediction updates instantly as you change any input
- **Sample buttons:** load a typical benign or malignant case with one click
- **Dynamic output visualisation:** probability bars, a malignancy risk meter, and a per-feature explanation chart ("why this prediction?")
- **Input validation:** missing, zero/negative and implausibly large values are blocked with clear messages; values outside the training range and inconsistent radius/area pairs trigger warnings
- **Batch mode:** upload a CSV, get predictions for every row, and download the results; handles empty files, unreadable files, missing columns and invalid rows gracefully
- **Model performance tab:** metrics, confusion matrix, ROC curve and model comparison
- **Saved model:** trained pipeline stored with `joblib` and loaded at start-up (auto-retrains if the file is missing or incompatible, so the app never crashes)

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** from the UCI Machine Learning Repository (Wolberg, Street and Mangasarian), also bundled with scikit-learn as `load_breast_cancer`.
569 cases (357 benign, 212 malignant). This project uses the 10 *mean* nucleus features: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.
The CSV used for training is included in `data/breast_cancer_wisconsin.csv`.

## Model and Performance

Pipeline: stratified 80/20 train/test split → `StandardScaler` + classifier inside one scikit-learn `Pipeline` (no data leakage) → 5-fold cross-validation on the training set to compare three models → best model saved with `joblib`.

| Model               | CV ROC-AUC | CV accuracy |
| ------------------- | ---------- | ----------- |
| Logistic Regression | 0.9852     | 0.9407      |
| Random Forest       | 0.9842     | 0.9451      |
| Gradient Boosting   | 0.9875     | 0.9407      |

The three models score almost the same, so **Logistic Regression** was selected: it is within 0.005 ROC-AUC of the best model, and it is simple and explainable.

**Held-out test set (114 cases):**

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
| -------- | --------- | ------ | -------- | ------- |
| 0.930    | 0.886     | 0.929  | 0.907    | 0.984   |

Confusion matrix (rows = actual, columns = predicted): benign → 67 correct, 5 wrongly flagged malignant; malignant → 39 correct, 3 missed.

**Limitations:** small dataset and only 10 features, so a few cases are still misclassified, including missed malignant cases. Results on 114 test cases can vary by a few percentage points.

## Tech Stack and Dependencies

Python · scikit-learn · pandas · NumPy · Matplotlib · joblib · Streamlit

All dependencies are listed in [`requirements.txt`](requirements.txt). `scikit-learn` is pinned to the version used to train the saved model.

## Project Structure

```
Breast_Cancer_Predictor/
├── app.py                          # Streamlit app (UI, validation messages, charts)
├── model_core.py                   # Data loading, training, joblib save/load, validation, prediction
├── train_model.py                  # Re-train and save the model
├── data/breast_cancer_wisconsin.csv
├── model/model.joblib              # Saved trained pipeline
├── model/metadata.json             # Metrics, feature ranges, ROC curve, comparison table
├── sample_batch.csv                # Example file for the Batch tab
├── requirements.txt
└── README.md
```

## Local Setup and Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/yashwant25mim10013-collab/Breast_Cancer_Predictor.git
cd Breast_Cancer_Predictor

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) re-train and re-save the model, then print the metrics
python train_model.py

# 5. Run the app
streamlit run app.py
```

If `streamlit` is not recognised, use `python -m streamlit run app.py`. The app opens at `http://localhost:8501`.

To try batch mode, upload `sample_batch.csv` in the **Batch (CSV)** tab.

## Deployment

Deployed for free on **Streamlit Community Cloud** from this public GitHub repository (main file: `app.py`).

## Author

**Yashwant Yadav**
GitHub: [yashwant25mim10013-collab](https://github.com/yashwant25mim10013-collab) · LinkedIn: [yashwant-yadav-533362383](https://linkedin.com/in/yashwant-yadav-533362383/)
