# Parkinson's Disease Prediction

### Overview
This repository contains the code and models developed for the prediction of Parkinson's disease. The project focuses on early detection using machine learning algorithms applied to different datasets related to gait, speech, and handwriting. The accompanying Flask app provides a user-friendly interface to interact with the prediction model.

### Repo Structure

python notebooks: 
This directory contains Jupyter notebooks with the code for developing the prediction model along with the datasets. Different notebooks are dedicated to each dataset, showcasing the analysis and implementation of machine learning algorithms.

app: 
The Flask app directory includes the code for the user interface. Users can input relevant data, and the app will predict the likelihood of Parkinson's disease based on the trained models.

pdf file: 
The research paper detailing the methodology and findings of the prediction model. The paper covers the analysis of Freezing of Gait dataset, Clinical Parkinson’s Dataset, and Waves and Spiral Dataset.

## Prediction Models

### Abstract
The research focuses on the early detection of Parkinson's disease using machine learning algorithms applied to diverse datasets. The datasets include Freezing of Gait (FOG) for gait analysis, Clinical Parkinson’s for speech abnormalities, and Waves and Spiral for handwriting impairments. A Convolutional Neural Network (CNN) using Transfer Learning is employed for image analysis.

### Methodology

1. Freezing of Gait (FOG) Model <br>
Dataset: Daphnet FOG dataset from UCI ML Repository.
Algorithms: Logistic Regression, KNN, Random Forest Classifier, Naive Bayes, Decision Tree Classifier.
Best Accuracy: Decision Tree Classifier with 94.98%.

2. Speech Model <br>
Dataset: Clinical Parkinson’s dataset with speech recordings.
Algorithms: Extra Tree Classifier, Logistic Regression, KNN, Random Forest Classifier, SVM, Decision Tree Classifier.
Best Accuracy: KNN with 97%.

3. Wave and Spiral CNN Model <br>
Dataset: Handwritten waves and spirals images.
Techniques: Data Augmentation, Transfer Learning (ResNet V1 50 model).
Evaluation Metrics: Accuracy, Precision, Recall, AUC score.
Results: Wave dataset outperformed the spiral dataset.

### Results
1. FOG Dataset: <br>
Decision Tree Classifier: 94.98% accuracy.

2. Speech Dataset: <br>
KNN: 97% accuracy.

3. Spiral and Wave Dataset: <br>
Transfer Learning CNN model:
Spiral: 80% accuracy on validation set.
Wave: 93.33% accuracy on validation set.

### Conclusion
The study successfully identified symptoms of Parkinson's disease using machine learning on different datasets. While FOG and Speech datasets performed well, the ease of obtaining data makes the Spiral and Wave dataset a more convenient and cost-effective option for early detection. The Flask app provides a practical interface for users to access the prediction model.
 
## Flask App

This Flask app provides a user-friendly interface to interact with the trained prediction model for Parkinson's disease. Follow the steps below to run the app and make predictions.

### Prerequisites
Python-3.6

### Usage

1. Navigate to app directory:
 ```cd app```
2. Install requirements.txt:
 ```pip install -r requirements.txt```
3. Run the app:
   ```python app.py```
