#  import required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin


#  class definition of the Dictionary Based Sentiment Analyzer which inherits from BaseEstimator, ClassifierMixin
class DictionaryBasedSentimentAnalyzer(BaseEstimator, ClassifierMixin):
	
	#  initially read the rules in the dictionary
    def __init__(self):
        
        self = self
        self.ruleframe_ = pd.read_csv("./classifier/dictionary_of_rules_stemmed_with_verius_nlp_tools.csv")

    #  method for fitting the data
    def fit(self, X=None, y=None):
        
        self.X_ = X
        self.y_ = y 

    #  function that generates unigrams and bigrams of the sentences
    def _generate_ngrams(self, text, number_of_gram):
        
        text = str(text).lower()
        tokens = [token for token in text.split(" ") if token != ""]

        ngrams = zip(*[tokens[i:] for i in range(number_of_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    #  function that preprocesses the unlabeled data
    def _data_prepare(self, list_of_strings_that_will_be_predicted):
        
        unigram = [self._generate_ngrams(text,1) for text in list_of_strings_that_will_be_predicted]
        bigram = [self._generate_ngrams(text,2) for text in list_of_strings_that_will_be_predicted]
        
        combined = []
        for i in range(len(bigram)):
            combined.append(bigram[i]) 
        for i in range(len(unigram)):
            combined[i].extend(unigram[i])

        #  returns the combination of the unigrams and bigrams of each sentences
        return combined


    #  function that calculates the polarity of a sentence
    def _predictor(self, rule, data, rf):
        
        outer_score = []
        outer_pos_score = []
        outer_neg_score = []
        for text in data:
            
            inner_score = []
            pos_score = []
            neg_score = []
            for token in text:
                
                if token in rule:
                    
                    pos_value = rf.at[rf.index[rf['stemmed_version'] == token][0], 'PosScore']
                    neg_value = rf.at[rf.index[rf['stemmed_version'] == token][0], 'NegScore']
                    
                    #  collects each token's polarity if it is in the dictionary
                    inner_score.append(float(pos_value) - float(neg_value))
                    pos_score.append(float(pos_value))
                    neg_score.append(float(neg_value))
            
            #  collects each text's polarity
            outer_score.append(sum(inner_score))
            outer_pos_score.append(sum(pos_score))
            outer_neg_score.append(sum(neg_score))
        
        converted_outer_score = []
        converted_outer_conf_score = []
        for i in range(len(outer_score)):
            
            #  encodes each "Positive" or "Negative" polarities into "0" or "1" labels
            if outer_score[i] > 0:
                
                scr = 1
                conf = str(outer_pos_score[i]/(outer_pos_score[i] + outer_neg_score[i]))
            
            elif outer_score[i] < 0:
                
                scr = 0
                conf = str(outer_neg_score[i]/(outer_pos_score[i] + outer_neg_score[i]))
            
            #  encodes the text as label "1" if it has no polarity and gives it default "0.50" polarity
            else:
                
                scr = 1
                conf = str(0.50)
            
            converted_outer_score.append(scr)
            converted_outer_conf_score.append(conf)

        return converted_outer_score, converted_outer_conf_score

    #  method for analyzing the sentiment of the given dataset
    def predict(self, list_of_strings_that_will_be_predicted):
        
        try:
            
            getattr(self, "X_")
        
        except AttributeError:
            
            raise RuntimeError("You must train classifer before predicting data!")

        #  get the rules from dictionary
        rf = self.ruleframe_
        rules = [r for r in rf["stemmed_version"]]
        
        #  preprocess the data
        combined_data = self._data_prepare(list_of_strings_that_will_be_predicted)
        
        #  get the predictions and confidences(confidence level ol predictions)
        predictions, confidences = self._predictor(rules, combined_data, rf)
        self.predictions_ = predictions
        self.confidences_ = confidences
        
        return predictions

    #  method for calculating the accuracy score of predicted data
    def score(self, X, y=None):
        
        try:
            
            getattr(self, "predictions_")
        
        except AttributeError:
            
            raise RuntimeError("You must predict before checking score!")

        numpy_float = accuracy_score(X, self.predictions_)
        normal_float = float(numpy_float)
        
        return normal_float

    #  method for calculating the prediction probabilities of predicted data
    def predict_proba(self, X=None, y=None):
        
        try:
            
            getattr(self, "predictions_")
        
        except AttributeError:
            
            raise RuntimeError("You must predict before checking predict_proba!")

        return [float(item) for item in self.confidences_]