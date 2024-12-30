# Wine Quality Prediction

This project aims to predict the quality of white wines from the Vinho Verde variety based on various physicochemical properties. Wine certification involves tests like determining density, pH, alcohol content, and acidity levels. The quality of the wine is rated on a scale of 1 to 10, which is then transformed into three categories: **0 - Very Bad**, **1 - Good**, and **2 - Excellent**.

The goal is to use machine learning to predict wine quality based on the results of these tests. This can benefit certification bodies, wine producers, and consumers.

## Dataset

The dataset used in this project is available from the UCI Machine Learning Repository (also available on Kaggle). The data consists of 12 physicochemical features of white wines along with their quality ratings:

### Features:
- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**

### Target Variable:
- **Quality**: A score between 0, 1, and 2, representing wine quality:
  - 0 - Very Bad
  - 1 - Good
  - 2 - Excellent

## Objective

The main objective is to explore and analyze the dataset to extract important features, and build a classification model to predict wine quality.

## Tasks Completed

1. **Initial Visual Analysis**: Conducted visualizations to understand the data and its distributions.
2. **Data Preprocessing**: Cleaned the dataset by handling missing values, normalizing features, and encoding categorical variables.
3. **Gathering Training and Testing Data**: Split the dataset into training and testing sets for model evaluation.
4. **Exploratory Data Analysis**: Explored relationships between features and quality using statistical and visual methods.
5. **Quality Prediction**: Built machine learning models to predict wine quality based on physicochemical properties.

## Technologies Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib**: For creating static visualizations.
- **Seaborn**: For enhanced data visualizations.
- **Scikit-learn**: For implementing machine learning algorithms and model evaluation.

## Machine Learning Algorithms Used

1. **K-Nearest Neighbors (KNN)**: A simple, yet effective algorithm for classification based on proximity.
2. **Random Forest Classifier**: An ensemble method using multiple decision trees to improve accuracy and reduce overfitting.
3. **Decision Tree**: A tree-like model for decision-making, useful for visualizing decision rules.
4. **Stochastic Gradient Descent (SGD)**: A gradient descent-based optimization method for training machine learning models.

## Final Algorithm

After evaluating different algorithms, **Random Forest Classifier** was selected as the final model for wine quality prediction, as it provided the best accuracy and performance in classifying wine quality.
