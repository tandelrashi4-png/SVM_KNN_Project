import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# Page title
st.title("Indian Flower Classification")
st.write("Flower Classification using KNN and SVM")


# Load dataset
data = pd.read_csv("data/flower.csv")

st.subheader("Indian Flower Dataset")
st.dataframe(data)


# Features and target
X = data[
    ["petal_length", "petal_width", "sepal_length", "sepal_width"]
]

y = data["species"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)


# SVM model
svm = SVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)


# Accuracy
st.subheader("Model Accuracy")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "KNN Accuracy",
        f"{knn_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "SVM Accuracy",
        f"{svm_accuracy * 100:.2f}%"
    )


# Confusion Matrix
st.subheader("Confusion Matrix")

model_choice = st.selectbox(
    "Select Model",
    ["KNN", "SVM"]
)

if model_choice == "KNN":
    cm = confusion_matrix(
        y_test,
        knn_pred,
        labels=sorted(y.unique())
    )
else:
    cm = confusion_matrix(
        y_test,
        svm_pred,
        labels=sorted(y.unique())
    )

cm_df = pd.DataFrame(
    cm,
    index=sorted(y.unique()),
    columns=sorted(y.unique())
)

st.dataframe(cm_df)


# Prediction
st.subheader("Predict Indian Flower")

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=4.8
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=1.8
)

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=6.2
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.1
)


# Prediction button
if st.button("Predict Flower"):

    input_data = pd.DataFrame(
        [[
            petal_length,
            petal_width,
            sepal_length,
            sepal_width
        ]],
        columns=[
            "petal_length",
            "petal_width",
            "sepal_length",
            "sepal_width"
        ]
    )

    knn_result = knn.predict(input_data)
    svm_result = svm.predict(input_data)

    st.success("KNN Prediction: " + knn_result[0])
    st.success("SVM Prediction: " + svm_result[0])