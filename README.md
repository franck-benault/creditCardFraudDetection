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
Accurancy is not a good metric for severe imbalanced datasets.

The confusion matrix is quite powerfull but it does not give you on figure to compare to other result. So I need also a single figure metric.

* Precision & Recall
* F1-score
* ROC-AUC
* Matthews correlation coefficient (MCC)
* Cohen's kappa

# Python and Python libraries
* imbalanced learning

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
* uniform :
	- This classifier generates random predictions with uniform probabilities for all classes  
* stratified :
  	- This classifier generates random predictions based on the class distribution in the training data 

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
* accuracy score: 0.9989
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
