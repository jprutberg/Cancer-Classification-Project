# Breast Cancer Classification with Neural Networks

## Introduction
Breast cancer ranks among the most widespread cancers globally, with early detection being vital for successful treatment outcomes. This project explores the use of fully connected feedforward neural networks using Python's `TensorFlow` library to classify breast cancer as benign or malignant based on various features derived from digitized images of a fine needle aspirate (FNA) of a breast mass. The dataset used in this project contains features such as mean radius, texture, perimeter, area, and several other measurements that describe the characteristics of the cell nuclei present in the image.

The project aims to build, evaluate, and optimize a neural network model that can accurately classify breast cancer cases. Various techniques such as L2 regularization, dropout, and hyperparameter tuning are employed to enhance the model's performance and robustness.

## Purpose
The primary objective of this project is to develop a machine learning model capable of accurately classifying breast cancer as benign or malignant. By leveraging neural networks, the project seeks to create a model that can generalize well to unseen data, potentially aiding in the early detection of breast cancer.

## Significance
Early and accurate classification of breast cancer is vital for determining the appropriate course of treatment. This project not only demonstrates the effectiveness of neural networks in medical diagnostics but also provides insights into various model optimization techniques. The methods used in this project, such as feature selection, regularization, and hyperparameter tuning, can be applied to other classification tasks in medical diagnostics and beyond.

## Project Overview
### Data Exploration and Preprocessing:
1. Data Loading and Summary: The dataset was loaded and inspected for basic statistics, data types, and missing values.
2. Feature Engineering: A correlation matrix was used to identify and remove highly correlated features, reducing the dataset's dimensionality.
3. Data Normalization: The feature data was normalized using the `TensorFlow` normalization layer to ensure that the model could converge faster and more effectively.

### Model Development:
1. Baseline Model: A simple neural network was built and trained to establish a baseline for model performance. The model was evaluated using metrics such as accuracy and loss on the validation set.
2. Model Improvement Techniques:
   a. L2 Regularization: Introduced to prevent overfitting by penalizing large weights.
   b. Dropout: Applied to reduce the risk of overfitting by randomly dropping units during training.
   c. Hyperparameter Tuning: A grid search over various dropout rates, L2 regularization strengths, and learning rates was conducted to find the best combination of hyperparameters.
    
### Model Evaluation:
1. Learning Curves: Generated to assess the model's learning behavior over different training set sizes.
2. Confusion Matrix: Used to evaluate the classification performance in terms of true positives, true negatives, false positives, and false negatives.
3. ROC and Precision-Recall Curves: Plotted to provide insights into the model's performance across different decision thresholds.

### Final Model:
The best-performing model, selected based on test accuracy and loss, was saved for future use. This model was fine-tuned with the optimal hyperparameters discovered during the grid search.

## Findings and Conclusions
The final neural network model achieved a high level of accuracy in classifying breast cancer, demonstrating the effectiveness of neural networks in this domain. The combination of L2 regularization and dropout was particularly effective in preventing overfitting, resulting in a model that generalizes well to new data. The project showcases the importance of model optimization techniques in achieving high performance in machine learning tasks.

The success of this project highlights the potential of neural networks in medical diagnostics and encourages further exploration into more complex architectures and techniques to push the boundaries of what is possible in automated cancer detection.

## References
The dataset is available at https://www.kaggle.com/datasets/sahilnbajaj/cancer-classification
