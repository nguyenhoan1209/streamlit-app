import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import feature_selection
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import mean_squared_error


class Regression:
    def simple_linear_regresstion(data):
        # Create a copy of the data
        data_copy = data.copy()
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        
        # Select the target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_number_colum)

        # Select feature variables
        feature_column = st.selectbox("Chọn biến tính năng", data_number_colum)

        # Split the data into training and testing sets
        X = data_copy[[feature_column]]
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = ((y_pred - y_test) ** 2).mean()
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display the results
        st.markdown("### Regression Results ###")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        
        col1 , col2, col3 = st.columns(3)

        # Coefficients
        with col1:
            st.markdown("Coefficients")
            st.dataframe(pd.DataFrame({"Coefficients":[model.intercept_, model.coef_[0]]}))

        # Residuals Statistics\
        with col2:
            # ANOVA
            st.markdown("ANOVA")
            f_values, p_values = feature_selection.f_classif(X_train, y_train)
            st.dataframe(pd.DataFrame({"F_value":[np.round(f_values[0], 4)], "p_values":["{:.4e}".format(p_values[0])]}))
        with col3:
            st.markdown("Model Summary")
            st.dataframe(pd.DataFrame({"MSE":[mse], "RMSE":[rmse], "R-squared":[r2]}))
        
        # Plot the predicted values vs. the actual values
        fig = px.scatter(x=X_train.squeeze(), y=y_train, trendline="ols", labels={
            "x": feature_column,
            "y": target_column
        }, trendline_color_override="red")
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        # Print the equation of the linear regression model
        equation = r"y = {:.2f} + {:.2f}x".format(model.intercept_, model.coef_[0])
        st.markdown(f"The equation is : <span style='color: green'></span>", unsafe_allow_html=True)
        st.latex(equation)

        
    def multiple_linear_regression(data):
        # Create a copy of the data
        data_copy = data.copy()
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns

        # Select the target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_number_colum)

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum, default=[target_column])

        if len(feature_columns) == 0:
            st.error("Please select at least one feature variable.")
            return

        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        y = data_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = ((y_pred - y_test) ** 2).mean()
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display the results
        st.header("Multiple Linear Regression Results")
        st.markdown("Number of Cases: {}".format(X_train.shape[0]))
        
        col1 , col2, col3 = st.columns(3)

        # Coefficients
        with col1:
            st.markdown("Coefficients")
            coefi = np.insert(model.coef_,obj=0,values=model.intercept_)
            newColumns = ["intercept"]
            for i in feature_columns:
                newColumns.append(i)
            frame = pd.DataFrame({"Columns": newColumns, "Coefficients":coefi})
            st.dataframe(frame.set_index("Columns"))
        with col2:
            # ANOVA
            st.markdown("ANOVA")
            f_values, p_values = feature_selection.f_classif(X_train, y_train)
            p_values_round = []
            for i in p_values:
                p_values_round.append("{:.3e}".format(i))
            st.dataframe(pd.DataFrame({"Columns":feature_columns, "F_value":f_values, "p_values":p_values_round}).set_index("Columns"))
        with col3:
            st.markdown("Model Summary")
            st.dataframe(pd.DataFrame({"MSE":[mse], "RMSE":[rmse], "R-squared":[r2]}))
        
        # Print the equation of the linear regression model
        equation = r"y = {:.2f}".format(model.intercept_)
        for i, coef in enumerate(model.coef_):
            equation += r" + {:.2f}x_{}".format(coef,i)
        st.markdown(f"The equation is : <span style='color: green'></span>", unsafe_allow_html=True)
        st.latex(equation)