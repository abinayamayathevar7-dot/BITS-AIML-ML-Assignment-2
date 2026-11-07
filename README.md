# Breast Cancer Classification Using Machine Learning

**BITS Pilani – M.Tech Artificial Intelligence and Machine Learning**  
**Course:** Machine Learning  
**Assignment:** Assignment 2

## 1. Problem Statement

The objective of this project is to develop and compare five supervised machine learning classification algorithms to predict whether a breast tumour is malignant or benign.

The project evaluates the performance of the following algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer Dataset

**Source:** Scikit-learn built-in Breast Cancer Dataset.

Dataset characteristics:

- Number of instances: 569
- Number of input features: 30
- Number of target classes: 2
- Classification type: Binary classification
- Target classes:
  - 0 – Malignant
  - 1 – Benign

The dataset contains numerical features computed from digitized images of breast mass samples.

The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness and symmetry.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Loaded the Wisconsin Breast Cancer dataset.
- Separated input features and target labels.
- Divided the dataset into 80% training and 20% testing data.
- Used stratified sampling to maintain class distribution.
- Applied StandardScaler to Logistic Regression, KNN, Gaussian Naive Bayes and SVM.
- Decision Tree was trained without feature scaling.
- Used random_state = 42 for reproducibility.

## 4. Machine Learning Algorithms

### Logistic Regression
A supervised classification algorithm that estimates the probability of a binary outcome using a logistic function.

### Decision Tree
A tree-based classification algorithm that makes predictions using a sequence of feature-based decisions.

### K-Nearest Neighbours (KNN)
A distance-based algorithm that classifies a sample based on the classes of its nearest neighbours.

### Gaussian Naive Bayes
A probabilistic classifier based on Bayes' theorem, assuming that features follow Gaussian distributions within each class.

### Support Vector Machine (SVM)
A classification algorithm that identifies an optimal decision boundary separating different classes.

## 5. Model Performance Comparison

All five models were evaluated using the same held-out test dataset.

| Algorithm | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

**Note:** Metrics are calculated on the 20% held-out test dataset. Precision, Recall and F1-score use class 1 (Benign) as the positive class. AUC uses the predicted probability of class 1.

## 6. Observations and Analysis

### Logistic Regression
- Achieved 98.25% accuracy.
- Achieved the highest AUC of approximately 0.9954.
- Demonstrated excellent precision, recall and F1-score.
- Performed consistently well on this dataset.

### Decision Tree
- Achieved 92.11% accuracy.
- Recorded the lowest AUC among the five algorithms.
- Its performance was comparatively lower than the other models on this test split.

### K-Nearest Neighbours
- Achieved 95.61% accuracy.
- Demonstrated good recall and F1-score.
- Performed better than Decision Tree and Gaussian Naive Bayes in overall classification metrics.

### Gaussian Naive Bayes
- Achieved 92.98% accuracy.
- Recorded an AUC of approximately 0.9868.
- Demonstrated good probability-based class discrimination despite comparatively lower accuracy.

### Support Vector Machine
- Achieved 98.25% accuracy.
- Recorded an AUC of approximately 0.9950.
- Achieved the same precision, recall, F1-score and MCC as Logistic Regression on this test split.
- Demonstrated excellent classification performance.

## 7. Overall Model Comparison and Winner

Based on the evaluation results, Logistic Regression and Support Vector Machine achieved the highest accuracy of 98.25%.

Both algorithms also achieved identical precision, recall, F1-score and MCC values.

However, Logistic Regression recorded a slightly higher AUC of 0.9954 compared with SVM's 0.9950.

Therefore, **Logistic Regression is selected as the overall winner** based on the AUC tie-breaker.

The result applies to the selected dataset, train-test split and model configurations. Further validation would be useful to confirm generalisation performance.

## 8. Streamlit Application

A Streamlit web application was developed to demonstrate the trained classification models.

Application features:

- Upload test data through a CSV file.
- Select one of the five trained classification algorithms.
- Load saved machine learning models.
- Evaluate uploaded test data without retraining.
- Display Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Display confusion matrix and classification report.

**Live Streamlit Application:** [(https://bits-aiml-ml-assignment-2-dfnsem3uefqyv6suydagmn.streamlit.app/)]

## 9. GitHub Repository

**GitHub Repository:** [https://github.com/abinayamayathevar7-dot/BITS-AIML-ML-Assignment-2/edit/main/]

### Project Structure

```text
BITS-AIML-ML-Assignment-2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── train_data.csv
├── test_data.csv
├── model_metrics.csv
├── EXECUTION_GUIDE.txt
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## 10. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train and save the models

```bash
python train_models.py
```

### Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

## 11. Conclusion

This project implemented and compared five supervised machine learning algorithms for breast cancer classification.

The experimental results demonstrate that Logistic Regression and Support Vector Machine achieved the best overall performance on the selected test dataset.

Logistic Regression was selected as the overall winner based on its slightly higher AUC.

The project provided practical experience in data preprocessing, supervised learning, model evaluation, performance comparison, model persistence and deployment using Streamlit.

---

**BITS Pilani – M.Tech AIML | Machine Learning Assignment 2**
