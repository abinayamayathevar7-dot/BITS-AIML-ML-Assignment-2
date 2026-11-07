# Machine Learning Assignment 2 – Breast Cancer Classification

## a. Problem statement
Implement five classification algorithms on one public dataset, compare their performance using six evaluation metrics, and demonstrate the saved models through an interactive Streamlit web application. The classification target is benign or malignant.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), accessed using `sklearn.datasets.load_breast_cancer`.
- **Number of instances:** 569.
- **Number of input features:** 30 numerical features.
- **Target:** Binary; `0 = malignant`, `1 = benign` (original Scikit-learn encoding).
- **Split:** 80% training and 20% testing; stratified; `random_state=42`.
- **Preprocessing:** StandardScaler is included in the Logistic Regression, KNN, Gaussian Naive Bayes and SVM pipelines. Decision Tree uses unscaled features. Scaling is fitted on training data only.
- **Source:** Scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset; original data from UCI Machine Learning Repository.

## c. GitHub Repository Link
**Replace before submission:** `[PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]`

Repository contents: `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `train_data.csv`, `test_data.csv`, `model_metrics.csv`, and five saved models in `model/`.

## d. Models used

### Evaluation metrics
All values below are calculated on the same held-out test set. Class `1` (benign) is treated as the positive class for precision, recall and F1. AUC uses the predicted probability of class `1`.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |\n|---|---:|---:|---:|---:|---:|---:|\n| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |\n| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |\n| Gaussian Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |\n| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n
### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9825 and AUC 0.9954. Strong precision and recall; highest AUC in this test split. |\n| Decision Tree | Accuracy 0.9211 and AUC 0.9163. Easier to interpret, but the lowest overall scores among these five settings. |\n| KNN | Accuracy 0.9561 and AUC 0.9788. Performs well; scaling is important because KNN relies on feature distances. |\n| Gaussian Naive Bayes | Accuracy 0.9298 and AUC 0.9868. Good ranking ability by AUC, but lower accuracy and F1 than the leading models. |\n| Support Vector Machine | Accuracy 0.9825 and AUC 0.9950. RBF kernel performs strongly and ties Logistic Regression on accuracy and other threshold metrics. |\n
### Overall winner for this dataset
Logistic Regression and Support Vector Machine tie on accuracy, precision, recall, F1 and MCC for this particular split. Logistic Regression has a slightly higher AUC (0.9954 compared with 0.9950), so it is selected as the winner using AUC as the tie-breaker. This conclusion applies only to this dataset, split and parameter choices.

## Streamlit application
**Replace before submission:** `[PASTE YOUR DEPLOYED STREAMLIT COMMUNITY CLOUD URL HERE]`

The application provides CSV upload, model selection, six evaluation metrics, a confusion matrix and a classification report. It loads saved trained models and evaluates the uploaded test data without retraining.

## How to run
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Notes
The reported metrics are from one stratified test split and may vary if the split or model parameters change. This is an educational ML exercise, not a clinical diagnostic tool. Before submission, run the project in BITS Virtual Lab, verify the outputs, personalize these observations in your own words, add your actual GitHub and deployed app URLs, and insert your BITS Lab screenshot in the final PDF.
