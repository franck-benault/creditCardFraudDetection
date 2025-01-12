# Credit Card Fraud Detection
This python project show how to use classifier to detect fraudulent credit card transactions

## Research about the card transactions fraud detection

If you request the Google Scholar web site [G1]. We can see the increase of the publication about card transactions fraud detection.
The number of publications have doubled in 10 years.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD00A-googleScholarStat.png)

## Data source

Even fraud detection in credit card transactions has a lot of publication. Data example are very rare.
A first source data is the example in Kaggle [PR3] but it is not very satisfying.


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


## Resampling techniques
### Oversampling the minority class
### Undersampling the majority class
### Combining oversampling and undersampling


# Hyperparameter tuning
GridSearchCV

# References
## General web site
* [G1] Google Scholar
	- https://scholar.google.com/

## Payment reference  and credit card fraud detection web sites
* [PR1] High risk merchant category code (MCC)
  - https://www.commercegate.com/ultimate-guide-high-risk-mcc-codes/
* [PR2] Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook
  - https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html
* [PR3] Kaggle credit card transactions
  - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
