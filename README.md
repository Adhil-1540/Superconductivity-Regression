# Predicting Superconducting Critical Temperature: A Machine Learning Approach

## Name: ADHIL KAPPAN

## Date: 09/06/2025

## Overview of Problem Statement

Superconductors exhibit zero electrical resistance below a certain temperature, known as the critical temperature (Tₕ). Identifying new superconducting materials with high Tₕ is crucial for advancements in energy transmission, magnetic levitation, and quantum computing. However, traditional experimental methods for discovering new superconductors are time-consuming and costly.

This project leverages machine learning (ML) techniques to predict the critical temperature of superconductors based on their features. The dataset includes 82 features extracted from material properties. Feature selection is performed using **SelectKBest**, enhancing model performance. This research accelerates the discovery of new superconducting materials and reduces reliance on costly laboratory experiments.

## Objective

To develop a machine learning model that accurately predicts the critical temperature (Tₕ) of superconductors using material properties.

* **Enhancing Material Discovery**: Use data-driven techniques for efficient identification of superconducting materials.
* **Reducing Experimental Costs & Time**: Provide reliable predictions to minimize laboratory experiments.
* **Improving Model Performance**: Use **SelectKBest** for feature selection, ensuring optimal and interpretable models.
* **Advancing Materials Informatics**: Integrate physics and ML for advanced superconductor research.

## Data Description

* **Source**: Superconductivity Dataset - UCI Machine Learning Repository
* **Dataset Link**: [Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data)

## Features

The dataset consists of 82 features derived from the chemical composition of superconducting materials:

* **Atomic Properties**: Atomic weight, electronegativity, atomic radius, valency.
* **Electronic Structure**: Density of states, Fermi energy, ionization energy.
* **Material Composition**: Elemental presence ratios.
* **Target Variable**: `critical_temp` (Critical Temperature)

## Initial Insights

* **Dataset Size**: 21,263 rows, 82 features
* **Target Variable**: Continuous numerical variable (`critical_temp`)

## Model & Methodology

### 1. Data Preprocessing

* Removed outliers using **IQR method**.
* Split data:

```python
X = df3.drop('critical_temp', axis=1); y = df3['critical_temp']
```

* Selected top features using **SelectKBest (Univariate Feature Selection)** to retain the most statistically significant features.
* Scaled data using **StandardScaler**:

```python
X_scaled = StandardScaler().fit_transform(X_selected)
```

* Split data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 2. Model Building

Implemented multiple regression algorithms:

* **Linear Regression**: Simple baseline for comparison.
* **Decision Tree Regressor**: Captures both linear and nonlinear patterns.
* **Random Forest Regressor**: Ensemble of decision trees, high accuracy.
* **Gradient Boosting Regressor**: Boosted trees for better prediction.
* **Support Vector Regressor (SVR)**: For complex, small data structures.

### 3. Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Best-Performing Model

* The model with the **lowest MSE, MAE, RMSE** and **highest R² Score** was considered optimal.
* Provided high accuracy and robustness in prediction.

## Hyperparameter Tuning and Pipeline

* Fine-tuned model performance through **GridSearchCV**.
* Used a **Pipeline** to streamline preprocessing and model training.

## Saving the Model

* Trained model saved using `joblib` for reuse without retraining.

## Testing with Unseen Data

* Model was tested on a separate validation dataset.
* Applied inverse transformation using **PowerTransformer** to convert predicted values back to original scale.

## Conclusion

* Machine learning successfully predicted the critical temperature of superconductors.
* **Feature selection and scaling** improved model accuracy.
* Best-performing model offers a **data-driven approach** to superconductor discovery, reducing cost and time.
* Predictions were inverse-transformed to obtain values in real-world temperature scale.

## Future Scope

* Apply deep learning techniques (e.g., neural networks) for improved prediction.
* Enhance feature engineering for better interpretability.
* Use trained model in real-world material discovery applications.
