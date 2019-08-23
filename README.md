# 1. INTRODUCTION

<br>

In Sentiment Analysis, there are two approaches.


* Machine Learning Algorithm Based Sentiment Analysis

* Dictionary(Rules) Based Sentiment Analysis

<br>

# 2. IMPLEMENTATION

<br>

In this project, I implemented the second approach which is Dictionary(Rules) Based Sentiment Analysis. This 

project is constructed on the combination of three sub projects.

* Constructing Turkish Dictionary(Rules)

* Creating Your Own Sklearn Classifier

* Implementing Turkish Dictionary(Rules) Based Sentiment Analyzing

<br>

## 2.1. CONSTRUCTING TURKISH DICTIONARY(RULES)

<br>

* At first, I examined English Dictionary Based Sentiment Analyzing tools. Then, I tried to create Dictionary of 

Turkish Rules. Because of inadequate Turkish resources, I examined the English resources that have words with their 

polarities. Finally, I found the SentiWordNet dataset. In this dataset, there are lots of words with their polarities. After 

that, I started to process the dataset. At first, I smoothed the data and then translated the words into Turkish equivalents 

using Google's Translation tool. Finally, I had sample [rules](https://github.com/slmttndrk/Turkish_Dictionary-Rules-_Based_Sentiment_Analysis/blob/master/classsifier/dictionary_of_rules_stemmed_with_verius_nlp_tools.csv) (words with their polarities). For more 

rules, please contact with [me](https://www.linkedin.com/in/selamettin-dirik/) After setting rules, I stepped into the second phase of my project.

<br>

## 2.2. CREATING YOUR OWN SKLEARN CLASSIFIER

<br>

* In this part, I searched for Sklearn Classifiers in order to understand the concept of them and create my own 

Sklearn Classifier. 

<br>

## 2.2.1. BUILDING AN OBJECT

<br>

* I created an Sklearn Text Classifier object by inheriting from Sklearn [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) and  [ClassifierMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html) 

classes. After that, I created some functions to implement my own Sklearn Classifier.

<br>

## 2.2.2.  ABIDING SKLEARN RULES

<br>

* All arguments of __init__() must have default value. It helps us to initialize the Sklearn Classifier just by typing 

DictionaryBasedSentimentAnalyzer()

<br>

## 2.2.3.  FIT AND PREDICT METHODS

<br>

* Every Sklearn Classsifier requires fit() and predict() method for classification

<br>

## 2.2.4. EXPLANATION OF SKLEARN CLASSIFIER'S METHODS
<br>

#### __init__()

<br>
	
* Initially read the rules from the dictionary

#### fit()

<br>

* Method for fitting the data
  
#### _generate_ngrams()

<br>

* Method that generates unigrams and bigrams of the sentences

#### _data_prepare()

<br>

* Method that preprocesses the unlabeled data

#### predictor()

<br>

* Method that calculates the polarity of the data

#### predict()

<br>

* Method for analyzing the sentiment of the data

#### score()

<br>

* Method for calculating the accuracy score of predicted data

#### predict_proba()

<br>

* Method for calculating the prediction probabilities of predicted data

<br>

Eventually, our new Sklearn Classifier is built. Then, I stepped into the last phase of my project.

<br>

## 2.3.  IMPLEMENTING TURKISH DICTIONARY(RULES) BASED SENTIMENT ANALYZING

<br>

## 2.3.1. DICTIONARY(RULES) BASED SENTIMENT ANALYZING STEPS

<br>

## 2.3.1.1. DATA FETCHING

<br>

*   The first rule is to get adequate dataset to train your model efficiently. Here, I have sample movie 

critics from Beyazperde. You can find it from this [link](http://www.beyazperde.com/filmler/elestiriler-beyazperde/).

<br>

## 2.3.1.2. DATA PREPROCESSING

<br>

*   This step is the crucial step for any kind of Machine Learning model training. Real life data is 

not always clean. So, you must process your dataset as possible as. In Machine Learning, there 

is a ratio that is, data preprocessing/cleaning is 80% and modelling is 20% of overall work. So, I 

also splitted data preprocessing into sub steps. 

<br>

## 2.3.1.2.1. LOAD DATASET

<br>
		
* [Dataset](https://github.com/slmttndrk/Turkish_Dictionary-Rules-_Based_Sentiment_Analysis/blob/master/sample_beyazperde_dataset.csv) is in the form of csv file. For more dataset, please contact with [me](https://www.linkedin.com/in/selamettin-dirik/)

<br>

## 2.3.1.2.2. ELIMINATE NAN VALUES

<br>
		
*   Nan values is not useful for training model

<br>

## 2.3.1.2.3. ELIMINATE PUNCTUATIONS

<br>

*   Punctuations are unnecessary for training model

<br>

## 2.3.1.2.4. NORMALIZATION

<br>

*   This corrects the miswritten words and throws meaningless words away

<br>

## 2.3.1.2.5. STEMMING/LEMMATIZATION

<br>

*   This removes the suffixes and gives us the root of each word

<br>

## 2.3.1.3. DATA CLASSIFICATION

<br>
	
*   In this step, I use my own Sklearn Classifier which is [DictionaryBasedSentimentAnalyzer](https://github.com/slmttndrk/Turkish_Dictionary-Rules-_Based_Sentiment_Analysis/blob/master/classsifier/dbsa.py) for Turkish 

Dictionary Based Sentiment Analyzing. This classifier checks ngrams(unigram and bigram) of the sentences and 

captures whether it matches our Sentiment Rules or not. If it matches, then it returns the polarity value 

of that token. I also, splitted data classification into some sub steps.

<br>

## 2.3.1.3.1. FIT AND PREDICT

<br>

*   The model learns rules by fitting and analyzes the sentiment of the data by predicting

<br>
    
## 2.3.1.3.2. OBSERVING [ACCURACY](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [F1, PRECISION AND RECALL](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) SCORES

<br>

*   This scores are useful for comparing model’s success

<br>
    
## 2.3.1.3.3. OBSERVING [CONFUSION MATRIX](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) AND PREDICTION PROBABILITIES

<br>

*   This gives us an intuition of how confidently the model makes the predictions

<br>
    
## 2.3.1.3.4. [TEN-FOLD CROSS VALIDATION](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)

<br>

*   This shows us whether our model performs correctly or not

<br>

## 2.3.1.4. MODEL PIPELINING AND PICKLING

<br>

*   In this step, I create a pipeline for the model. [Pipelining](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) prevents us from repeating all steps again 

and again. With the help of pipelining, when I give any raw unlabeled data, at first, the model preprocess

it and then, makes prediction. So, it makes our model reusable.

<br>

*   [Pickling](https://scikit-learn.org/stable/modules/model_persistence.html) a model means transforming it into binary form. It makes our model portable. When you want to

use the model in different projects, by just loading this pickled file, you can use the model and get

predictions wherever you want. 

<br>

## 4. IMPROVEMENTS

<br>

*   The model score can be improved by increasing the number of "Turkish Dictionary(Rules)".

<br>

## 5. CONCLUSION

<br>

*   As I mentioned before, this project constructed on three sub projects which are different branches of Sentiment 

Analyzing. In the first step, I learned how to prepare Dictionary(Rules) for Sentiment Analysis. And, this 

Dictionary(Rules) will be a good basis for Turkish Dictionary(Rules) Based Sentiment Analyzing field. I'm glad 

to prepare this Dictionary(Rules). In the second step, I learned how to create a Text Classifier 

for Sentiment Analysis. This gained me a good understanding of Sklearn Classifiers and helped me to create my 

own Classifier. Creation of this kind of Classifier is a good chance for me to dive deep into the Machine Learning 

Algorithms and their working principles. In the last step, I learned to apply Dictionary(Rules) Based Sentiment 

Analyzing/Text Classification. It also provided me very useful knowledge about Natural Language Processing. 

Since, fetching and preprocessing the dataset is the crucial part of any Machine Learning model training. In 

conclusion, this project gained me lots of NLP concepts that is very crucial for a Data Scientist. I hope this 

study will be useful for everyone. 

<br>

## 6. RESOURCES/THANKS

<br>

*   I completed this project in cooperation with [Verius Technology Company](https://verius.com.tr/).The training dataset (Beyazperde) 

and data preprocessing tools (normalization, stemming) are provided me by them. I also used python libraries 

such as: Sklearn, Pandas, Numpy, Nltk.


<br>
