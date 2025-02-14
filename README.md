# Credit Card Fraud Detection
This project shows how to use classifier to detect fraudulent credit card transactions.
It is in a context of a back office and the goal is to do a classification based on supervised learning.
the goal is not do to a kind of scoring.

## Research about the card transactions fraud detection

If you request the Google Scholar web site [G1]. We can see the increase of the publication about card transactions fraud detection.
The number of publications have doubled in 10 years.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD00A-googleScholarStat.png)

## Fraud detection in Cloud as a service
During the last years some articles have been published about the credit card fraud detection. The article [CL1] explains how to build a solution of fraud detection using Big Query.

Amazon has a solution "Fraud Detector" which is a complete fraud detection tool in its Cloud as a Service [CL2].   


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
	- it means the Area Under the Receiver Operation Characteristic Curve.  It provides a  measure of the ability of the model to discriminate between classes across different threshold levels. A higher AUC score means a better performing model. The scale of this score is different 0.5 is a total random classification and 1.0 represents a perfect classification. It is also possible to draw the ROC curve. The ROC value is less sensible than the F1 score about the false positive. So if the goal is to find the maximum of the fraudulent transactions even there are false positives then ROC score is a better choice than the f1 score.
* Matthews correlation coefficient (MCC)
	- it is adapted for the problems with two classes, which is the case of the fraud detection. A random classification returns 0. A perfect classification gives 1. The following the formula a complete misclassification should return -1.
* Cohen's kappa

## Python and Python libraries

* [PL1] scikit-learn
	- https://scikit-learn.org

This is really the main library for Machine Learning in Python.
Also now a lot of other libraries like imbalanced learn follows the same coding interfaces.
So it is quite easy to move from scikit-learn to another library. 

* [PL2] imbalanced learn
	- https://imbalanced-learn.org

This library addresses the issue of imbalanced data set.

## Transaction data detail
In fact the transaction data can be split in three part
* The transaction data itself
  	- Amount
  	- Currency
  	- Date time of the transaction
* The merchant information
  	- MCC (merchant code activity)
  	- terminal id
  	- terminal country
* The card and card holder information
  	- Card id
  	- Card holder name
  	- issuer id (code banque)
 
The question is what are the information really relevant to detect the Fraud.
The issuer id, the card holder name are probably less or not relevant.
The MCC, amount are relevant may be but not enough to detect the fraud.

### Numerical data
#### Transaction amount
The transaction amount if it is not in Euro is converting to Euro.
When the transaction amount is put in box plot compairing the Geniune and the Fraudulent transaction, the outliers makes the graphic difficult to read.
It is clear that the very high amount are more present for the Geniune transactions thant the fraudulent transaction.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01B-Amount-boxplot.png)

If we draw the histogram of the transaction, we can see the extreme heavy tail. The diagram is quite difficult to understand.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01B-Amount-histogram.png)

In the book [BF1] topics 2.5.3, Aurélien Géron advices to transform the value by using the log.
After this transformation, the histogram shows something closer to the normal distribution.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01B-Amount-log10-histogram.png)

If we do a transformation of the amount using log10, the boxplot diagram is clearer too.
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
There are more "no reserval" transactions than "reversal" transactions (no_resersal are around 98% of the transactions).
But the reversal transactions are used sometimes by the fraudsters. The fraud rate is around 10 times higher for reversal transactions.
The information value calculated on this column in 0.013, quite low but not without any information.

Because this column contains only 3 possible values the encoding of this column is not really a problem.


#### How to encode the data
##### Grouping and one hot encoding
The idea is the find a way to group several categories to a new one without lossing to much information.
This grouping must have a business meaning.
About the country, the idea is define three group:
* Belgium (BELGIUM)
* All the other european countries (EUROPE)
* All the other countries (WORLD)

The information value calculated on this new column is 1.117 (compared to the 1.907).
![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD01D-piediag-country-group.png)

##### Use algorithm accepting categorical data
There are two algorithms accepting directly the categorical data
* Autogluon [PL3]
* CatBoost [PL4]

I haven't studied in this way further and I have used these algorithms with the categorical data already processed by grouping and one hot encoding. 

#### Transaction information enrichment 
The transaction information are not enough to detect correctly the fraudulent transactions. 
In the article [A2] this issue is clearly set : 
> a single transaction information is typically not sufficient to detect a fraud occurrence and the analysis has to consider aggregate measures like total spent per day, transaction number per week or average amount of a transaction.

The idea is to create card holder profile and merchant profile by using unsupervised learning and then enrich the transactions with these profiles. 

Also I would like to add some information about the previous transaction. So the global enrichment of the transaction information looks like

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/transactionsPreparationProcess.png)


### Card holder profile
The idea is to create using agregate of the transactions during at least last 30 days a profile of the card holder.
The last 90 days is usually what a bank provides by default to the client in the application.
If a card holder who uses very rarely his card, suddenly uses his card a lot is a possible fraud.

if we take an agregate of the 
- last 30 days, there are 3 600 000 cards
- last 60 days, there are 4 100 000 cards
- last 90 days, there are 4 500 000 cards


#### Clustering RFM (Recency, Frequency, Monetary)
This is an approach very often used for customer segmentation for the analysis of the churn.
We can use it also for the card holder even it seems probably too simple.

This approach can be apply for the card holder.
* the frequency is how often the card is used.
* the recency is the duration since the card has been used last time
* Monetary is the sum of amount for the payment done with this card.

About the frequency and the monetary field a quit explorary data analysis shows that it is better to use the log of these values.
Without this operation the clusters are not correctly balanced.

For the fraud detection, I want to have a least 4 clusters and with almost the same size.
The silouhette score is quite low (around 0,38 for 4 clusters for 90 days aggregate).

### Merchant profile
  
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
The idea here instead of modifying the data (oversampling or undersampling).
Here we use a hyperparameter scale_pos_weight to ask the classifier to increase the focus to the minority class.

class_weight Parameter for Scikit-Learn ?
#### XGBoost


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
The good point is that the number of fraudulent transactions filtered are low (0.05%). 

## Main Classifiers
### First Classifiers
The goal of these first classifiers (DummyClassifier, Naive Bayes, Auto gluon) are not to be used in a "real production context" but to set up some figures to evaluate the performance of more advanced classifiers.

#### Dummy Classifier
The Dummy classifier does not learn anything from the data. 
In fact it is used as a baseline for comparing the performance of more complex and more realistic models. 
There are several possible strategies, here are the most known :
* most frequent :
	- This classifier always predicts the most frequent class in the training data.
* uniform :
	- This classifier generates random predictions with uniform probabilities for all classes
* stratified :
  	- This classifier generates random predictions based on the class distribution in the training data


##### Results
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
* 
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

#### Naive Bayes Classifier
Even if it is a real machine learning algorithm, the expected result is low.
The reason is that it algorithm takes each variable independently which is cleary wrong for the transactions.
This is also the reason that there is no scaling or normalisation applied on the input data.
The result are better than the dummy classifier. So the algorithm learns something from the data.

The results are at the end of this document, but they are quite poor 
and following the time of this project, some features are added and the results then decrease.
It could be possible to do a feature selection but because it won't give good result.
The goal is here to have a kind of starting point to evaluate more complex algorithms.

#### AutoGluon [PL3]
AutoGluon is a project coming for AWS (Amazon) but now available as Python library.
The goal is to do Machine Learning without a deep knowlegde of this subjet. 
AutoGluon will choose the best algorithm and the best hyperparameters associated to the data in input.

Even it is quite easy to start with. It does not follow completly the API defined in scikit-learn.
The result done is that CatBoost would be the most performant algorithm. 
Last point because AutoGluon tries a lot of algorithms and hyperparameters, 
it is very computing expensive and the learning time is around 15 times longer than the worst algorithm in term of time computing.

The first result shows that CatBoost [PL4] is probably the most performant algorithm.

### More Advanced classifiers
#### XGBoost
XGBoast is a project outside scikit-learn.
It is known a very effective for imbalanced data.
The first results show that it is among the most performant classifiers.

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


#### Imbalanced learning library
##### imblearn.ensemble EasyEnsembleClassifier

## Final results
### Time learning
The figures (here in secondes) do not mean anything alone. But it allows to compare the time learning between algorithms.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD99A-Summary-timeLearning.png)

### Metrics on test data

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD99A-Summary-metrics.png)

### Metrics on next days
Here we check after tuning a model on the day1 if this model perform correctly to the next days (day2 day3 and day8).
We want to see if the performance stays stable or if the performance drops quickly.
if it drops it means that the model overfits and has just saved some noise.

![image](https://github.com/franck-benault/creditCardFraudDetection/blob/main/imgs/FD99A-Summary-nextdays.png)

## References
### General web site
* [G1] Google Scholar
	- https://scholar.google.com/

### Articles about credit card fraud detection
* [A1] Impact of sampling techniques and data leakage on XGBoost performance in credit card fraud detection
  - https://arxiv.org/pdf/2412.07437
* [A2] Learned lessons in credit card fraud detection from a practitioner perspective (2014 Andrea Del Pozzolo, Olivier Caelen)
  - http://euro.ecom.cmu.edu/resources/elibrary/epay/1-s2.0-S095741741400089X-main.pdf

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

### Cloud articles and solutions
* [CL1] How to build a serverless real-time credit card fraud detection solution (Google Big Query)
  - https://cloud.google.com/blog/products/data-analytics/how-to-build-a-fraud-detection-solution?hl=en
* [CL2] Amazon Fraud detector
  - https://aws.amazon.com/fr/fraud-detector/
 
### Python and Python libraries
* [PL1] scikit-learn
	- https://scikit-learn.org
* [PL2] imbalanced learn
	-  https://imbalanced-learn.org
* [PL3] AutoGluon documentation
  	- https://auto.gluon.ai/stable/index.html
* [PL4] CatBoost documentation
	- https://catboost.ai/

### Books over Python and Machine learning
#### Book in French
These books exist in English (Translated in English or the original version is in English)
* [BF1] Machine Learning avec Scikit-learn (Aurélien Géron) Dunod 2023 third edition
