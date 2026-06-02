import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

dagshub.init(repo_owner='srimukha.sarma', repo_name='Mlflow-experiments', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/srimukha.sarma/Mlflow-experiments.mlflow")

#load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#Random Forest parameters
max_depth = 5
n_estimators = 8

#Mentioning Experiment
mlflow.autolog() #tracks all metrics and models automatically.
mlflow.set_experiment("mlflow-tutorial") #will create experiment in UI if it doesnt exist already

with mlflow.start_run(): #running the model in this context manager
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matr = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matr, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    #Doesnt log file automatically
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": "Srimukha", "Project": "Wine Classification"})

    print(accuracy)


