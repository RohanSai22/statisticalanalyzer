import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

def main():
    st.title("Advanced Statistical Analysis Tool")

    st.sidebar.header("Data Input")
    data_source = st.sidebar.selectbox("Choose data source", ["Upload File", "Sample Dataset", "URL"])

    data = None

    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
    elif data_source == "Sample Dataset":
        sample_dataset = st.sidebar.selectbox("Choose a sample dataset", ["Iris", "Tips", "Titanic"])
        if sample_dataset == "Iris":
            data = sns.load_dataset("iris")
        elif sample_dataset == "Tips":
            data = sns.load_dataset("tips")
        elif sample_dataset == "Titanic":
            data = sns.load_dataset("titanic")
        st.success(f"Sample dataset '{sample_dataset}' loaded successfully!")
    elif data_source == "URL":
        url = st.sidebar.text_input("Enter the URL of the dataset")
        if url:
            try:
                data = pd.read_csv(url)
                st.success("Data loaded from URL successfully!")
            except Exception as e:
                st.error(f"Failed to load data from URL: {e}")

    if data is not None:
        st.header("Dataset Preview")
        st.dataframe(data.head())

        st.sidebar.header("Preprocessing")
        if st.sidebar.button("Preprocess Data"):
            st.subheader("Data Preprocessing")

            # Handle missing values
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found for preprocessing.")
            else:
                imputer = KNNImputer(n_neighbors=5)
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

                # Standardize numerical features
                scaler = StandardScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

                # Encode categorical features
                for col in data.select_dtypes(include=['object']).columns:
                    data[col] = LabelEncoder().fit_transform(data[col])

                st.write("Data preprocessed successfully!")
                st.dataframe(data.head())

        st.sidebar.header("Statistical Analysis")
        analysis_type = st.sidebar.selectbox("Choose analysis type", ["Descriptive Statistics", "Inferential Statistics", "Regression Analysis", "Clustering", "Time Series Analysis"])

        if analysis_type == "Descriptive Statistics":
            st.subheader("Descriptive Statistics")
            st.write(data.describe())
            for column in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(data[column], kde=True, ax=ax)
                ax.set_title(f"Distribution of {column}")
                st.pyplot(fig)

        elif analysis_type == "Inferential Statistics":
            st.subheader("Inferential Statistics")
            columns = st.multiselect("Select two columns for T-test", numeric_cols or [])
            if len(columns) == 2:
                t_stat, p_value = stats.ttest_ind(data[columns[0]], data[columns[1]])
                st.write(f"T-test Results: T-statistic={t_stat}, P-value={p_value}")

        elif analysis_type == "Regression Analysis":
            st.subheader("Regression Analysis")
            target_column = st.selectbox("Select target column", numeric_cols or [])
            predictor_columns = st.multiselect("Select predictor columns", [col for col in numeric_cols if col != target_column])
            if target_column and predictor_columns:
                X = data[predictor_columns]
                y = data[target_column]

                models = {
                    'Linear': LinearRegression(),
                    'Ridge': Ridge(),
                    'Lasso': Lasso(),
                    'ElasticNet': ElasticNet()
                }

                for name, model in models.items():
                    model.fit(X, y)
                    predictions = model.predict(X)
                    st.write(f"{name} Regression:")
                    st.write("R^2 Score:", r2_score(y, predictions))
                    st.write("MSE:", mean_squared_error(y, predictions))
                    st.write("MAE:", mean_absolute_error(y, predictions))

        elif analysis_type == "Clustering":
            st.subheader("Clustering")
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            if len(numeric_cols) >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                data['Cluster'] = kmeans.fit_predict(data[numeric_cols])

                st.write("Cluster Centers:", kmeans.cluster_centers_)
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[numeric_cols[0]], y=data[numeric_cols[1]], hue=data['Cluster'], palette='viridis', ax=ax)
                ax.set_title("Clustering Results")
                st.pyplot(fig)
            else:
                st.error("At least two numeric columns are required for clustering.")

        elif analysis_type == "Time Series Analysis":
            st.subheader("Time Series Analysis")
            column = st.selectbox("Select column for analysis", numeric_cols or [])
            order = st.text_input("Enter ARIMA order (p,d,q)", value="1,1,1")
            if column and order:
                try:
                    p, d, q = map(int, order.split(","))
                    ts_model = ARIMA(data[column], order=(p, d, q))
                    ts_fit = ts_model.fit()
                    st.write(ts_fit.summary())
                    fig = ts_fit.plot_diagnostics()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error in ARIMA modeling: {e}")

if __name__ == "__main__":
    main()
