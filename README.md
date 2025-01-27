# Credit Card Fraud Detection
This python project show how to use classifier to detect fraudulent credit card transactions

## Research about the card transactions fraud detection

If you request the Google Scholar web site [G1]. We can see the increase of the publication about card transactions fraud detection.
The number of publications have doubled in 10 years.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD00A-googleScholarStat.png)

## Data source
### Kaggle example [PR5]
Even fraud detection in credit card transactions has a lot of publications. Data example are very rare.
A first source data is the example in Kaggle [PR5]. Even it is very often used for publication. It is not very satisfying.

### Simulated data
A second solution is to simulate data set using for example the Python library imbalanced learn [PL2].
The need od example data for fraud detection is described in the site [PR4]. 
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
* in Kaggle example [PR5] 0,173%

The Bank of France publishes each year a report about the payment [PR6].
The fraud rate about card payment in value is between 0,07% and 0,05% following the years.
Be careful the market in France is different than in Belgium. The split between credit card and debit card is less clear in France.
For a country like Belgium where there is a clear split between credit card and debit card. I think that the fraud rate is around 0,1%.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD00B-fraudRateFrance.png)

About debit card transactions, the fraud rate can be considered ten times lower (0.01%).
This is due to the fact that debit cards have less possibilities about e-commerce, about booking.
For this example the imbalanced is higher.

This work is based on credit card transactions because it was more simple to achieve some significant results.


## How to manage imbalanced data
### Definition
Imbalanced data refers to scenarios where the classes in the dataset are not reprsented equally.
Also the target class is often the one underrepresented. For fraud on card transactions, I often see *severe* imbalanced data. 

### Metrics choice issue

The confusion matrix is an important tool to visualize the force or the weakness of a classifier. 
But it is quite difficult to use it for automation. A simple indicator is always easier for automation.

Among the single figure metrics the first one is Accurancy. it measures the proportion of correctly classified transactions among all transactions.
For accuracy the scale is from 0 (poor result) to 1 (excelent result)

$Accurancy = (TP+TN)/(Total Samples)$

But it is not a good metric for severe imbalanced datasets. We can see that we the example of the dummy classifier.
With imbalanced dataset, we can speak about **misleading** accuracy.

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

## Python and Python libraries

* [PL1] scikit-learn
	- https://scikit-learn.org
* [PL2] imbalanced learn
	- https://imbalanced-learn.org

## Transactions data detail
### Numerical data
#### Transaction amount
The transaction amount if it is not in Euro is converting to Euro.
When the transaction amount is put in box plot compairing the Geniune and the Fraudulent transaction, the outliers makes the graphic difficult to read.
It is clear that the very high amount are more present for the Geniune transactions thant the fraudulent transaction.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01B-Amount-boxplot.png)

If we do a transformation of the amount using log10, the diagram is clearer.
The explanation is usually that the fraudsters try to not be too visible so they do often low amount or medium amount transactions.
This finding is explained in the article [A1].
The Amount column from the Kaggle example has exactly the same distribution.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01B-Amount-log10-boxplot.png)


### Categorical data
One issue with the transactions data is that there are a lot of categorical data.
Also some of these categorical data may have a lot of possible values (more than 100).
So the classical encoding may create a lot of column and the algorithm will be slow and will have poor results.

#### MCC (Merchant Category Code)
The MCC code defined the type of activity the merchant does. Even it is not always fully thrustable.
It seems an important information about the fraud detection. The fraudsters are targeted a type of activities.
The information value calculated on this field is 1.404 which shows the importance of this field.

If we consider the case of 1 million transactions per day. We can evaluate that 500 differents mcc categories are used per day.
The idea is to group the mcc code in around 10 groups without loosing to much information.

Issuing the grouping proposed in [PR1] does not give a good IV result the information value calculated on this grouping is 0.00008.

If we just do a simple grouping using the code 6011 (AUTOMATED CASH DISBURSEMENTS) or ATM representing around 2% of the transactions.
This simple grouping with on one side ATM and on the other side all other codes has an information value of 0.023.

#### Terminal Country
The terminal country is stored using the norm ISO-3166-1 the code alpha3, meaning that Belgium is stored using "BEL".

During one day, there are sometimes 200 values which is huge because the full table has aurond 250 codes.
We could take only the most present values.

the IV (information value) calculated on this field is 1.907.

#### Trx_reversal
The normal financial flux for a payment transaction is that the card holder pays the merchant (no reversal).
But sometimes the merchant has to reimbourses the card holder this is the reversal transactions.
There are more "no reserval" transactions than "reversal" transactions.
But the reversal transactions are used sometimes by the fraudsters. The fraud rate is around 10 times higher for reversal transactions.
The information value calculated on this column in 0.013, quite low but not without any information.

Because this column contains only 3 possible values the encoding of this column is not really a problem.


#### How to encode the data
The idea is the find a way to group several categories to a new one without lossing to much information.
This grouping must have a business meaning.
About the country, the idea is define three group:
* Belgium (BELGIUM)
* All the other european countries (EUROPE)
* All the other countries (WORLD)

The information value calculated on this new column is 1.117 (compared to the 1.907).
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01D-piediag-country-group.png)

### Card holder profil

### Merchant profil
  
## Resampling techniques
### Data-based technique
#### Undersampling the majority class
* RandomUnderSampling
* Tomek's link
* Edited Nearest Neighbour (ENN)

#### Oversampling the minority class
* Random over sampling
* SMOTE (Synthetic Minority Over-sampling Technique)
* ADASYN (Adaptive Synthetic Sampling)

#### Combining oversampling and undersampling

### Algorith-based solutions
Ensemble methods:
* BalancedRandomForestClassifier
* EasyEnsembleClassifier
* BalancedBaggingClassifier

### Tuning-based solutions
class_weight Parameter for Scikit-Learn.

## Data filtering
The idea behind the filtering is to simplify the work of the classifier by filtering some transactions are a considered as very probably geniune.

The metrics are in fact 
* the proportion of total transactions filtered
* the number of transactions fraudulent that are filtered

### Filter on the amount
The transaction with high amount are very often genuine.
The result is not very convincig because only a few transactions are filtered (0.02%). 
The good point is that the number of fraudulent transactions filtered are low : only one

### Filter on the partial reversal
The transaction with partial reversal are very often genuine but are very rare.
So again the result is not very convincig because only a few transactions are filtered.
The good point is that the number of fraudulent transactions filtered are low.

## Main Classifiers
### Dummy Classifier
The Dummy classifier does not learn anything from the data. 
In fact it is used as a baseline for comparing the performance of more complex and more realistic models. 
There are several possible strategies, here are the most known :
* most frequent :
	- This classifier always predicts the most frequent class in the training data.
* uniform :
	- This classifier generates random predictions with uniform probabilities for all classes
* stratified :
  	- This classifier generates random predictions based on the class distribution in the training data


#### Results
Using strategy="most_frequent"
* accuracy score: 0.9989
* f1 score: 0.0000
* mcc score: 0.0000
* roc auc score: 0.5000
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixMostFrequent.png)

Using stategy="uniform"
* accuracy score: 0.4990
* f1 score: 0.0021
* mcc score: 0.0007
* roc auc score: 0.5051
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixUniform.png) 

Using stategy="stratified"
* accuracy score: 0.9978
* f1 score: 0.0000
* mcc score: -0.0011
* roc auc score: 0.4994
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD03A-dummyClassifierMatrixStratified.png)

the following result are the minimum that the learning algorithm should exceed:

maximal result:
* ~~accuracy score: 0.9989~~
* f1 score: 0.0021
* mcc score: 0.0007
* roc auc score: 0.5051

### Naive Bayes Classifier
Even if it is a real machine learning algorithm, the expected result is low.
The reason is that it algorithm takes each variable independently which is cleary wrong for the transactions.
This is also the reason that there is no scaling or normalisation applied on the input data.
The result are better than the dummy classifier. So the algorithm learns something from the data.

On test data:
* f1 score: 0.0431
* mcc score: 0.10741
* roc auc score: 0.7631

This is a sort of second starting point from more complex algorithms.

## Hyperparameters tuning
The following techniques () are coming from the package sklearn.model_selection of scikit-learn.

### RandomizedSearchCV
With this method, the hyperparamters are chosen randomly and not all the combinaison are tested.
But it is a good approach when there are a lot of hyperparameters.

### GridSearchCV
GridSearchCV (Grid Search Cross-Validation) is a technique used in machine learning for hyperparameter tuning.
It searchs the optimal hyperparameters by trying all the possible combinaisons giving by the parameter param_grid (a dictionary).

The disavantage of this technique is that it may be computationally expensive, and take a huge time to process all the possibilities.
So the idea is to first use the RandomizedSearchCV to have a first approach of the hyperparameters and then use the GridSearchCV to refine the result.

## Main classifiers comparaison
### Skitlearn library
#### Ensemble package
There 3 approaches in this package :
* Bagging
* Boosting
* Stacking

#### Adaboost classifier
This classifier is present in skitlearn library in the package ensemble.
It uses a boosting approach.
The scaling has no influence to the performance.

## Final results
### Time learning
The figures (here in secondes) do not mean anything alone. But it allows to compare the time learning between algorithm.
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD99A-Summary-timeLearning.png)

### Metrics

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD99A-Summary-metrics.png)

## References
### General web site
* [G1] Google Scholar
	- https://scholar.google.com/

### Articles about credit card fraud detection
* [A1] Impact of sampling techniques and data leakage on XGBoost performance in credit card fraud detection
  - https://arxiv.org/pdf/2412.07437

### Payment reference  and credit card fraud detection web sites
* [PR1] High risk merchant category code (MCC)
  - https://www.commercegate.com/ultimate-guide-high-risk-mcc-codes
* [PR2] MCC group resource center
  - https://resourcecenter.comdata.com/wp-content/uploads/sites/4/2019/10/Merchant_Groups_and_Category_Codes_MCCs.pdf
* [PR3] MCC group citybank
  - https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf
* [PR4] Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook
  - https://fraud-detection-handbook.github.io/fraud-detection-handbook
* [PR5] Kaggle credit card transactions
  - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* [PR6] bank of France (OBSERVATORY FOR THE SECURITY OF PAYMENT MEANS) annual report 3023
    - https://www.banque-france.fr/system/files/2025-01/OSMP_2023_EN.pdf
 
### Python and Python libraries
* [PL1] scikit-learn
	- https://scikit-learn.org
* [PL2] imbalanced learn
	-  https://imbalanced-learn.org 
