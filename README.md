

# Advanced Statistical Analysis Tool

This Streamlit application provides a user-friendly interface for performing advanced statistical analysis on datasets. It supports a wide range of functionalities, including data preprocessing, statistical analysis, clustering, regression, and time series analysis.


Try it out at [Streamlit](https://statisticalanalyzer.streamlit.app/)
---

## Features

### 1. **Data Input**
   - **Upload File**: Upload a CSV or Excel file for analysis.
   - **Sample Dataset**: Use preloaded datasets like Iris, Tips, and Titanic.
   - **URL**: Enter a URL to load a dataset directly.

### 2. **Data Preprocessing**
   - Handle missing values using the **KNN Imputer**.
   - Standardize numerical features using **StandardScaler**.
   - Encode categorical features using **LabelEncoder**.

### 3. **Statistical Analysis**
   - **Descriptive Statistics**: View summary statistics and visualize distributions.
   - **Inferential Statistics**: Perform a T-test between two columns.
   - **Regression Analysis**:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - ElasticNet Regression
   - **Clustering**:
     - Apply **KMeans Clustering** and visualize cluster results.
   - **Time Series Analysis**:
     - Fit and analyze ARIMA models.
     - Plot diagnostics for time series models.

### 4. **Interactive Visualizations**
   - Histogram and KDE plots for data distributions.
   - Scatter plots for clustering results.
   - Time series diagnostics plots.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Streamlit
- Required Python libraries (see `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/statistical-analysis-tool.git
   cd statistical-analysis-tool
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## How to Use

### Step 1: Load Data
1. Choose a data source from the sidebar:
   - Upload a file.
   - Select a sample dataset.
   - Enter a dataset URL.

2. Preview the dataset in the **Dataset Preview** section.

### Step 2: Preprocess Data
- Click **Preprocess Data** in the sidebar to:
  - Impute missing values.
  - Standardize numerical columns.
  - Encode categorical variables.

### Step 3: Perform Analysis
1. Choose an analysis type from the sidebar:
   - Descriptive Statistics
   - Inferential Statistics
   - Regression Analysis
   - Clustering
   - Time Series Analysis

2. Follow the on-screen prompts to configure the analysis.

3. View results and visualizations in real time.

---

## Dependencies

The following Python libraries are required:

```plaintext
streamlit
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
statsmodels
plotly
```

---

## Example Use Cases

1. **Exploratory Data Analysis (EDA)**:
   - Upload a dataset, preprocess it, and explore using descriptive statistics and visualizations.

2. **Clustering**:
   - Segment customers or products using KMeans clustering.

3. **Regression Modeling**:
   - Predict outcomes using linear or regularized regression techniques.

4. **Time Series Analysis**:
   - Analyze and forecast trends using ARIMA models.


---

## Contributing

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and create a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- Developed using [Streamlit](https://streamlit.io/).
- Sample datasets provided by [Seaborn](https://seaborn.pydata.org/).

