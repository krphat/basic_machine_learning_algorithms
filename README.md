# Basic Machine Learning Algorithms for binary classification problems.

Welcome to the **Basic Machine Learning Algorithms for binary classification problems Project**! This repository demonstrates the implementation of various machine learning algorithms to solve a binary classification problem using the famous Titanic dataset.

## Project Overview

The goal of this project is to predict the survival status of passengers on the Titanic based on their features. The dataset has been analyzed and preprocessed, and several classification algorithms have been implemented and optimized using GridSearchCV to find the best hyperparameters.

### Algorithms Implemented

1. **Decision Tree**  
2. **Random Forest**  
3. **Naïve Bayes**  
4. **Logistic Regression**  
5. **Support Vector Machine (SVM)**  
6. **Adaptive Boosting (AdaBoost)**  
7. **Gradient Boosting**

## Repository Structure

This repository is organized as follows:

### Data Analysis and Preprocessing
The Titanic dataset was analyzed and processed to ensure it is suitable for training and evaluation. The preprocessing steps include handling missing values, feature engineering, feature encoding, and scaling. All preprocessing steps are contained in the following file:

- [**Data Preprocessing**]()

### Machine Learning Algorithms
Each algorithm has been implemented in a separate file. The files include the model training, hyperparameter tuning using GridSearchCV, and evaluation.

- [**Decision Tree**]()
- [**Random Forest**]()
- [**Naïve Bayes**]()
- [**Logistic Regression**]()
- [**Support Vector Machine (SVM)**]()
- [**Adaptive Boosting (AdaBoost)**]()
- [**Gradient Boosting**]()

## Key Features

- **Preprocessing:** Comprehensive data cleaning and feature engineering.
- **Model Variety:** Implementation of multiple machine learning models for comparison.
- **Hyperparameter Tuning:** Utilization of GridSearchCV for optimal parameter selection.
- **Scikit-learn and TensorFlow Decision Forests:** Leveraging two powerful libraries for model implementation.

## Getting Started

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

## Results

Each script outputs the evaluation metrics of the respective model, including accuracy, precision, recall, F1-score, ROC curve and AUC scores. You can compare the performance of the algorithms to determine the most effective one for the dataset.

## Contributing

Contributions are welcome! If you have ideas to improve the project or add more algorithms, feel free to submit a pull request.

## Acknowledgments

- The Titanic dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- Libraries used: [Scikit-learn](https://scikit-learn.org/), [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)

---

Feel free to explore, modify, and learn from this project. Happy coding!
