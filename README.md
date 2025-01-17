# Credit Card Fraud Detection
This python project show how to use classifier to detect fraudulent credit card transactions

## Research about the card transactions fraud detection

If you request the Google Scholar web site [G1]. We can see the increase of the publication about card transactions fraud detection.
The number of publications have doubled in 10 years.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD00A-googleScholarStat.png)

## Data source
### Kaggle example [PR3]
Even fraud detection in credit card transactions has a lot of publications. Data example are very rare.
A first source data is the example in Kaggle [PR3]. Even it is very often used for publication. It is not very satisfying.

### Simulated data
A second solution is to simulate data set using for example the Python library imbalanced learn [PL2].
The need od example data for fraud detection is described in the site [PR2]. 
This site explained how to create a data set simulating credit card transaction with some fraudulent transactions.

The following figures are just an approximation
* Number transactions per day
	- 1 000 000
* Number fraudulent transactions per day
 	- 1 000 
	- Fraud rate 0,1% 
* 2/3 Mastercard transactions, 1/3 Visa transactions

## Fraud rate 
Fraud rate
* in Kaggle example [PR3] 0,173%
* in credit card transactions in Europe (Belgium, France) in 2024 0.1%

About debit card transactions, the fraud rate can be considered ten times lower (0.01%).
This is due to the fact that debit cards have less possibilities about e-commerce, about booking.
For this example the imbalanced is higher.

This work is based on credit card transactions because it was more simple to achieve some significant results.


# How to manage imbalanced data
## Definition
Imbalanced data refers to scenarios where the classes in the dataset are not reprsented equally.
Also the target class is often the one underrepresented. For fraud on card transactions, I often see *severe* imbalanced data. 

## Metrics choice issue


The confusion matrix is an important tool to visualize the force or the weakness of a classifier. 
But it is quite difficult to use it for automation. A simple indicator is always easier for automation.

Among the single figure metrics the first one is Accurancy. it measures the proportion of correctly classified transactions among all transactions.
For accuracy the scale is from 0 (poor result) to 1 (excelent result)

$Accurancy = (TP+TN)/(Total Samples)$

But it is not a good metric for severe imbalanced datasets. We can see that we the example of the dummy classifier.

* Precision
	- it measures the proportion of actual fraudulent transactions correctly identified by the model.
	- $Recall = TP / (TP+ FN)$
* Recall
	- it quantifies the proportion of transactions predicted as fraudulent that are indeed fraudulent.
	- $Precision = TP / (TP+FP)$
* F1-score
	- it is the harmonic mean of Precision and Recall. This value is useful when there is a significant imbalanced. F1 score has been used during previous projects in my company.
	- $F1-score = 2 x (Precision x Recall) / (Precision + Recall)$

All these metrics (Accuracy, Recall, Precision, F1 score) follow the same scale : 1 is a perfect result, 0 is the worst result. 
F1 following the formula could be undefined but in this case the implementation returns 0 with a warning.  
F1 score is a first important metrics for managing severe imbalanced data.

* ROC-AUC
	- it means the Area Under the Receiver Operation Characteristic Curve.  It provides a  measure of the ability of the model to discriminate between classes across different threshold levels. A higher AUC score means a better performing model. The scale of this score is different 0.5 is a total random classification and 1.0 represents a perfect classification. It is also possible to draw the ROC curve.
* Matthews correlation coefficient (MCC)
	- it is adapted for the problems with two classes, which is the case of the fraud detection. A random classification returns 0. A perfect classification gives 1. The following the formula a complete misclassification should return -1.
* Cohen's kappa

# Python and Python libraries

* [PL1] scikit-learn
	- https://scikit-learn.org
* [PL2] imbalanced learn
	- https://imbalanced-learn.org
  
## Resampling techniques
### Oversampling the minority class
### Undersampling the majority class
### Combining oversampling and undersampling

# Main Classifiers
## Dummy Classifier
The Dummy classifier does not learn anything from the data. 
In fact it is used as a baseline for comparing the performance of more complex and more realistic models. 
There are several possible strategies, here are the most known :
* most frequent :
	- This classifier always predicts the most frequent class in the training data.
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixMostFrequent.png)
* uniform :
	- This classifier generates random predictions with uniform probabilities for all classes
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixUniform.png) 
* stratified :
  	- This classifier generates random predictions based on the class distribution in the training data
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixStratified.png)

### Results
Using strategy="most_frequent"
* accuracy score: 0.9989
* f1 score: 0.0000
* mcc score: 0.0000
* roc auc score: 0.5000

Using stategy="uniform"
* accuracy score: 0.4990
* f1 score: 0.0021
* mcc score: 0.0007
* roc auc score: 0.5051

Using stategy="stratified"
* accuracy score: 0.9978
* f1 score: 0.0000
* mcc score: -0.0011
* roc auc score: 0.4994

the following result are the minimum that the learning algorithm should exceed:

maximal result:
* ~~accuracy score: 0.9989~~
* f1 score: 0.0021
* mcc score: 0.0007
* roc auc score: 0.5051

# Hyperparameter tuning
GridSearchCV

# References
## General web site
* [G1] Google Scholar
	- https://scholar.google.com/

## Payment reference  and credit card fraud detection web sites
* [PR1] High risk merchant category code (MCC)
  - https://www.commercegate.com/ultimate-guide-high-risk-mcc-codes
* [PR2] Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook
  - https://fraud-detection-handbook.github.io/fraud-detection-handbook
* [PR3] Kaggle credit card transactions
  - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
 
## Python and Python libraries
* [PL1] scikit-learn
	- https://scikit-learn.org
* [PL2] imbalanced learn
	-  https://imbalanced-learn.org 
