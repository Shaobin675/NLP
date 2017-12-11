#NLTK-LoadPickle.py

#File: sentiment_mod.py

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


load_documents = open("pickles/documentsMovieReview.pickle", "rb")
documents = pickle.load(load_documents)
load_documents.close()
load_word_features = open("pickles/word_features39k.pickle", "rb")
word_features = pickle.load(load_word_features)
load_word_features.close()

# def find_features(document):
# 	words = word_tokenize(document)
# 	features = {}
# 	for w in word_features:
# 		features[w] = (w in words) #if w is in words, then features[word] = 1, otherwise set to 0

# 	return features

# featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets_f = open("pickles/featureset39K.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

#movie review dataset
testing_set = featuresets[1800:]
training_set = featuresets[:1800]
#pos-neg dataset
# testing_set = featuresets[10000:]
# training_set = featuresets[:10000]


open_file = open("pickles/NBC39K.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)


open_file = open("pickles/MNB_classifier39K.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



open_file = open("pickles/BernoulliNB_classifier39K.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


open_file = open("pickles/LogisticRegression_classifier39K.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


open_file = open("pickles/SGDClassifier_classifier39K.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)


open_file = open("pickles/LinearSVC_classifier39K.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


open_file = open("pickles/NuSVC_classifier39K.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  NuSVC_classifier,
                                  SGDC_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


def sentiment(document):

	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words) #if w is in words, then features[word] = 1, otherwise set to 0

	return voted_classifier.classify(features),voted_classifier.confidence(features)
