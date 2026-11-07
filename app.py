import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="AIML Assignment 2", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification Model Explorer")
st.caption("BITS Pilani M.Tech AIML | Machine Learning Assignment 2")
st.write("Upload held-out test data containing all 30 features and the `target` column. Encoding: 0 = malignant, 1 = benign. Saved models are loaded; the app does not retrain.")
files={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","KNN":"knn.joblib","Gaussian Naive Bayes":"gaussian_naive_bayes.joblib","Support Vector Machine":"support_vector_machine.joblib"}
model_name=st.selectbox("Select classifier",list(files))
uploaded=st.file_uploader("Upload test CSV",type=["csv"])
if uploaded:
    try:
        data=pd.read_csv(uploaded)
        if "target" not in data.columns: st.error("CSV must include target column."); st.stop()
        model=joblib.load(Path(__file__).parent/"model"/files[model_name])
        X=data.drop(columns="target"); y=data["target"]
        expected=list(model.feature_names_in_) if hasattr(model,"feature_names_in_") else list(X.columns)
        missing=[c for c in expected if c not in X.columns]
        extra=[c for c in X.columns if c not in expected]
        if missing or extra: st.error(f"Feature mismatch. Missing={missing}; unexpected={extra}"); st.stop()
        X=X[expected]; pred=model.predict(X); prob=model.predict_proba(X)[:,1]
        vals={"Accuracy":accuracy_score(y,pred),"AUC":roc_auc_score(y,prob),"Precision":precision_score(y,pred,zero_division=0),"Recall":recall_score(y,pred,zero_division=0),"F1":f1_score(y,pred,zero_division=0),"MCC":matthews_corrcoef(y,pred)}
        st.subheader(model_name+" – Evaluation")
        cols=st.columns(6)
        for col,(k,v) in zip(cols,vals.items()): col.metric(k,f"{v:.4f}")
        c1,c2=st.columns(2)
        with c1:
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),index=["Actual malignant (0)","Actual benign (1)"],columns=["Predicted malignant (0)","Predicted benign (1)"]))
        with c2:
            st.subheader("Classification Report")
            rep=classification_report(y,pred,labels=[0,1],target_names=["malignant (0)","benign (1)"],output_dict=True,zero_division=0)
            st.dataframe(pd.DataFrame(rep).transpose().round(4))
        st.subheader("Test data preview"); st.dataframe(data.head(10))
    except Exception as e: st.error(f"Evaluation error: {e}")
else: st.info("Upload this repository's test_data.csv to evaluate the selected saved model.")
