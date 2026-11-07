"""Rebuild the 80:20 split and train/save five classifiers. Run: python train_models.py"""
from pathlib import Path
import pandas as pd, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
ROOT=Path(__file__).resolve().parent; (ROOT/"model").mkdir(exist_ok=True)
d=load_breast_cancer(as_frame=True); frame=d.data.copy(); frame["target"]=d.target.astype(int)
train,test=train_test_split(frame,test_size=.2,random_state=42,stratify=frame.target)
train.to_csv(ROOT/"train_data.csv",index=False); test.to_csv(ROOT/"test_data.csv",index=False)
X,y=train.drop(columns="target"),train.target; Xt,yt=test.drop(columns="target"),test.target
models={
"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,random_state=42))]),
"Decision Tree":DecisionTreeClassifier(max_depth=5,min_samples_split=4,random_state=42),
"KNN":Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))]),
"Gaussian Naive Bayes":Pipeline([("scale",StandardScaler()),("model",GaussianNB())]),
"Support Vector Machine":Pipeline([("scale",StandardScaler()),("model",SVC(kernel="rbf",probability=True,random_state=42))])}
rows=[]
for name,model in models.items():
 model.fit(X,y); joblib.dump(model,ROOT/"model"/(name.lower().replace(" ","_")+".joblib"))
 p=model.predict(Xt); proba=model.predict_proba(Xt)[:,1]
 rows.append({"ML Model Name":name,"Accuracy":accuracy_score(yt,p),"AUC":roc_auc_score(yt,proba),"Precision":precision_score(yt,p),"Recall":recall_score(yt,p),"F1":f1_score(yt,p),"MCC":matthews_corrcoef(yt,p)})
out=pd.DataFrame(rows); out.to_csv(ROOT/"model_metrics.csv",index=False); print(out.to_string(index=False))
