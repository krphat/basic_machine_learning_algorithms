# Basic Machine Learning Algorithms for binary classification problems.

Welcome to the **Basic Machine Learning Algorithms for binary classification problems Project**! This repository demonstrates the implementation of various machine learning algorithms to solve a binary classification problem using the famous Titanic dataset.

## 📌 Project Overview

The goal of this project is to predict the survival status of passengers on the Titanic based on their features. The dataset has been analyzed and preprocessed, and several classification algorithms have been implemented and optimized using GridSearchCV to find the best hyperparameters.

### 🤖 Algorithms Implemented

1. **Decision Tree**  
2. **Random Forest**  
3. **Naïve Bayes**  
4. **Logistic Regression**  
5. **Support Vector Machine (SVM)**  
6. **Adaptive Boosting (AdaBoost)**  
7. **Gradient Boosting**

## 📂 Repository Structure

This repository is organized as follows:

### 🔍 Data Analysis and Preprocessing
The Titanic dataset was analyzed and processed to ensure it is suitable for training and evaluation. The preprocessing steps include handling missing values, feature engineering, Exploratory Data Analysis (EDA), feature encoding, and feature scaling. All preprocessing steps are contained in the following file:

- [**Data Preprocessing**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/data_preprocessing.ipynb)

### 🤖 Machine Learning Algorithms
Each algorithm has been implemented in a separate file. The files include the model training, hyperparameter tuning using GridSearchCV, and evaluation.

- [**Decision Tree**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/TFDF_CART.ipynb)
- [**Random Forest**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/TFDF_RandomForest.ipynb)
- [**Naïve Bayes**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/NBC.ipynb)
- [**Logistic Regression**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/LogisticRegression.ipynb)
- [**Support Vector Machine (SVM)**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/SVM.ipynb)
- [**Adaptive Boosting (AdaBoost)**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/AdaBoost.ipynb)
- [**Gradient Boosting**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/TFDF_GradientBoostedTree.ipynb)

### 📊 Model Performance & Comparisons

- Comparison of model performance is available in [**compare_models.ipynb**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/compare_models.ipynb).
- K-Fold Cross-Validation comparison is performed in [**baseline_models_comparison.ipynb**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/baseline_models_comparison.ipynb)

**Result:** The best performing model is **Logistic Regression**, achieving an accuracy of **0.77033** on **Kaggle test set**.

### ⚖️ Handling Imbalanced Data
This project also explores techniques for handling imbalanced data, including:
- Class Weighting
- Undersampling & Oversampling
- Bagging & Boosting Methods

📌 Find details in: [**handling_imbalanced_data.ipynb**](https://github.com/krphat/basic_machine_learning_algorithms/blob/main/source%20codes/baseline_models_comparison.ipynb)

#### 🔍 Insights
- **ADASYN** and **Class Weight** methods improved the model’s ability to identify the minority class but slightly reduced overall accuracy.
- **Random Forest** and **Gradient Boosting** models performed well in classifying the minority class and achieved higher accuracy on **Kaggle** compared to **Logistic Regression**.

## 🔑 Key Features

- **Preprocessing:** Comprehensive data cleaning and feature engineering.
- **Model Variety:** Implementation of multiple machine learning models for comparison.
- **Hyperparameter Tuning:** Utilization of GridSearchCV for optimal parameter selection.
- **K-Fold Cross Validation:** Helps evaluate the performance of models in the most objective way.
- **Imbalanced Data Handling:** Techniques like oversampling, undersampling, class weighting, and ensemble methods help balance datasets and improve model performance.
- **Scikit-learn and TensorFlow Decision Forests:** Leveraging two powerful libraries for model implementation.

## 🎯 Results

Each script outputs the evaluation metrics of the respective model, including accuracy, precision, recall, F1-score, ROC curve and AUC scores.

### 🏆 Best Models on Kaggle Test Set

| Model        | Accuracy           |
| ------------- |:-------------:|
| Logistic Regression      | 0.77033|
| Random Forest     | 0.77272      |
| Gradient Boosting | 0.77751      |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- The project is built on [Google Colab](https://colab.google/): Python 3.10+

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/krphat/basic_machine_learning_algorithms.git
   ```

2. Navigate to the project directory:
   ```bash
   cd basic_machine_learning_algorithms
   ```

3. Run the preprocessing script to prepare the dataset:
   ```bash
   data_preprocessing.ipynb
   ```

4. Execute any algorithm script to train and evaluate the respective model. For example:
   ```bash
   TFDF_CART.ipynb
   ```

## Contributing

Contributions are welcome! If you have ideas to improve the project or add more algorithms, feel free to submit a pull request.

## Acknowledgments

- The Titanic dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- Libraries used: [Scikit-learn](https://scikit-learn.org/), [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)

---

Feel free to explore, modify, and learn from this project. Happy coding!
