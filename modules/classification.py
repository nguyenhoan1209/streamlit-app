from cmath import nan
from sklearn import svm
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.iloc[-3][0] = nan
    df_classification_report.iloc[-3][1] = nan
    df_classification_report.iloc[-3][3] = df_classification_report.iloc[-2][3]
    return df_classification_report


class Classification:
    def knn_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)
        
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            # Split the data into training and testing sets
            X = data_copy[feature_columns]
            X = pd.get_dummies(X)
            y = data_copy[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Get the number of neighbors from the user
            k = st.slider("Chọn số lượng hàng xóm k", 1, 50)

            # Create and train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = knn.predict(X_test)


            # Display the results
            st.markdown("### K-Nearest Neighbors Classification Results ###")
            st.markdown("Number of Cases: {}".format(X_train.shape[0]))
            st.dataframe(get_classification_report(y_test, y_pred))
        
        
        
    def lgreg_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)
        
        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        X = pd.get_dummies(X)
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the number of neighbors from the user
        k = st.slider("Chọn số lần lặp tối đa k", 0, 500, step=50)

        # Create and train the KNN classifier
        knn = LogisticRegression(max_iter=k)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)


        # Display the results
        st.markdown("### Logistic Regression Classification Results ###")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        st.dataframe(get_classification_report(y_test, y_pred))
        
    def randomfor_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)
        
        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        X = pd.get_dummies(X)
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the number of neighbors from the user
        k = st.slider("Chọn số cây k", 1, 50, step=1)

        # Create and train the KNN classifier
        knn = RandomForestClassifier(n_estimators= k)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)


        # Display the results
        st.markdown("### Random Forest Classification Results ###")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        st.dataframe(get_classification_report(y_test, y_pred))
        
    def naivebayes_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)
        
        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        X = pd.get_dummies(X)
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the KNN classifier
        knn = GaussianNB()
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)


        # Display the results
        st.markdown("### Naive-Bayes Classification Results ###")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        st.dataframe(get_classification_report(y_test, y_pred))
        
    def svm_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)
        
        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        X = pd.get_dummies(X)
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the KNN classifier
        knn = SVC()
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)


        # Display the results
        st.markdown("### SVM Classification Results ###")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        st.dataframe(get_classification_report(y_test, y_pred))
    
    