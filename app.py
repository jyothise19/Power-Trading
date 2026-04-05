# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:34:29 2026

@author: jyoth
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Electricity Price Forecasting", layout="wide")

# Add background color
st.markdown(
    """
    <style>
    .main {
        background-color: #e8f4f8;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Electricity Price Forecasting - Power Trading")

st.write("Upload the dataset to predict electricity prices")

# Load trained model
model_package = joblib.load("power_price_model.pkl")
if isinstance(model_package, dict) and "model" in model_package:
    model = model_package["model"]
    expected_features = model_package.get("features", None)
else:
    model = model_package
    expected_features = None

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(df.head())

    # Data summary
    st.subheader("Data Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # Visualizations
    st.subheader("Data Visualizations")

    # Histogram and Box plot side by side
    col1, col2 = st.columns(2)
    with col1:
        if 'MCP (Rs/MWh) *' in df.columns:
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.histplot(df['MCP (Rs/MWh) *'], bins=30, kde=True, ax=ax, color="#1f77b4")
            ax.set_title("Distribution of MCP")
            st.pyplot(fig)

    with col2:
        bid_cols = [col for col in df.columns if 'Bid' in col]
        if bid_cols:
            fig, ax = plt.subplots(figsize=(3, 2))
            df[bid_cols].boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor="#ff7f0e"))
            ax.set_title("Box Plot of Bid Volumes")
            st.pyplot(fig)

    # Time series full width
    if 'Date' in df.columns and 'MCP (Rs/MWh) *' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(df['Date'], df['MCP (Rs/MWh) *'], color="#2ca02c")
        ax.set_title("MCP Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("MCP (Rs/MWh) *")
        st.pyplot(fig)

    # Select features
    features = df.select_dtypes(include=np.number)

    if st.button("Predict Prices"):
        # remove non numeric columns
        df = df.drop(columns=["Date","Time Block","Hour","time",'MCP (Rs/MWh) *'], errors="ignore")

        features = df.select_dtypes(include=["int64","float64"])
        
        # Reindex to match expected features if available
        if expected_features is not None:
            features = features.reindex(columns=expected_features, fill_value=0)
        
        predictions = model.predict(features)

        df["Predicted MCP"] = predictions

        st.subheader("Predictions")
        st.write(df[["Predicted MCP"]].head(20))

        # Plot predictions side by side
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(predictions, color="#d62728")
            ax.set_title("Predicted Electricity Prices")
            ax.set_xlabel("Time Blocks")
            ax.set_ylabel("Price (Rs/MWh)")
            st.pyplot(fig)

        with col2:
            if 'MCP (Rs/MWh) *' in df.columns:
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.plot(df['MCP (Rs/MWh) *'][:len(predictions)], label='Actual', color="#1f77b4")
                ax.plot(predictions, label='Predicted', color="#ff7f0e")
                ax.set_title("Actual vs Predicted MCP")
                ax.legend()
                st.pyplot(fig)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )