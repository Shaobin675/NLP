#NLTK-Pickle.py
import nltk, random, time
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews, stopwords


import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

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


posFile = open('pos.txt', encoding='latin-1').read()
negFile = open('neg.txt', encoding='latin-1').read()

documents = [] #sentences
all_words = [] #break into words

#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]
#allowed_word_types = ["J"]

for p in posFile.split('\n'):
	documents.append( (p, "pos") )
	words = word_tokenize(p)
	
	pos = nltk.pos_tag(words)
	for w in pos:
		#filtering words. adj, adv and verb only
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())
    
'''
('going', 'VBG')
V going
'''    
for p in negFile.split('\n'):
	documents.append( (p, "neg") )
	words = word_tokenize(p)
	
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types: 
			all_words.append(w[0].lower())



save_documents = open("pickles/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

posWords = word_tokenize(posFile)
negWords = word_tokenize(negFile)

for w in posWords:
	all_words.append(w.lower())
for w in negWords:
	all_words.append(w.lower())

stop_words = set(stopwords.words('english'))
all_words = [w for w in all_words if not w in stop_words]
all_words = nltk.FreqDist(all_words) #45889 in total, 20173 after stopwords

# print(all_words.most_common()[20:30])
# print(all_words['september'])
#print(len(all_words.keys()), list(all_words.keys())[4900:5000])

word_features = list(all_words.keys())[:20000]
save_word_features = open("pickles/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_feature(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words) #if w is in words, then features[word] = 1, otherwise set to 0

	return features

featuresets = [(find_feature(rev), category) for (rev, category) in documents]
# fileDatasets = open('featureset.pickle', 'rb')
# datasets = pickle.load(fileDatasets)
# fileDatasets.close()
random.shuffle(featuresets)
#print(len(featuresets)) #10664


# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

##
### negative data example:      
##training_set = featuresets[100:]
##testing_set =  featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
save_NBC = open('./pickles/NBC20K.pickle', 'wb')
pickle.dump(classifier, save_NBC)
save_NBC.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_NBC = open('./pickles/MNB_classifier20K.pickle', 'wb')
pickle.dump(MNB_classifier, save_NBC)
save_NBC.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_NBC = open('./pickles/BernoulliNB_classifier20K.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_NBC)
save_NBC.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_NBC = open('./pickles/LogisticRegression_classifier20K.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_NBC)
save_NBC.close()


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
save_NBC = open('./pickles/SGDClassifier_classifier20K.pickle', 'wb')
pickle.dump(SGDClassifier_classifier, save_NBC)
save_NBC.close()


##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_NBC = open('./pickles/LinearSVC_classifier20K.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_NBC)
save_NBC.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
save_NBC = open('./pickles/NuSVC_classifier20K.pickle', 'wb')
pickle.dump(NuSVC_classifier, save_NBC)
save_NBC.close()

voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
save_NBC = open('./pickles/voted_classifier20K.pickle', 'wb')
pickle.dump(voted_classifier, save_NBC)
save_NBC.close()
