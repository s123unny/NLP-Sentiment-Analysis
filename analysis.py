import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import gensim
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer

class MeanEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		# if a text is empty we should return a vector of zeros
		# with the same dimensionality as all the other vectors
		self.dim = len(word2vec.itervalues().next())
	def fit(self, X):
		return self
	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec]
					or [np.zeros(self.dim)], axis=0)
			for words in X
		])

def train_predict(train_tweet, train_target, train_sentiment, test_tweet, test_target, test_sentiment, wordembedding, train_snippet, test_snippet):
	Y_train = np.array([float(x) for x in train_sentiment])
	Y_test = np.array([float(x) for x in test_sentiment])
	X_train_boe = []
	X_test_boe = []
	X_train_bos = []
	X_test_bos = []
	X_train = []
	X_test = []
	print 'train: ', len(Y_train), 'test:', len(Y_test)

        vec = CountVectorizer(ngram_range=(1,1))
	vec.fit(train_tweet)
	X_train_bos = vec.transform(train_snippet).toarray()
	X_test_bos = vec.transform(test_snippet).toarray()
	print 'len x_Train_bos, X_test_bos', X_train_bos.shape , X_test_bos.shape

	# let X be a list of tokenized texts (i.e. list of lists of tokens)
	model = gensim.models.Word2Vec.load(wordembedding)
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	
	vec = MeanEmbeddingVectorizer(w2v)
	vec.fit(train_tweet)
	X_train_boe = vec.transform(train_tweet)
	X_test_boe = vec.transform(test_tweet)
	print "len x_Train_boe, X_test_boe", X_train_boe.shape , X_test_boe.shape
	
	X_train = np.concatenate((X_train_bos, X_train_boe), axis=1)
	X_test = np.concatenate((X_test_bos, X_test_boe), axis=1)
	#X_train = X_train_boe
	#X_test = X_test_boe

	print "len x_Train_, X_test_", X_train.shape , X_test.shape
	clf = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
	print 'Extra trees regression...'
	clf.fit(X_train, Y_train)
	print "testing..."
	Y_predict = clf.predict(X_test)
	mse = mean_squared_error(Y_test, Y_predict)
	Y_test_f1 = []
	for i in range(len(Y_test)):
		if Y_test[i] < 0.:
			Y_test_f1.append('-1')
		elif Y_test[i] == 0:
			Y_test_f1.append('0')
		else:
			Y_test_f1.append('1')
	Y_predict_f1 = []
	for i in range(len(Y_predict)):
		if Y_predict[i] < 0.:
			Y_predict_f1.append('-1')
		elif Y_predict[i] == 0:
			Y_predict_f1.append('0')
		else:
			Y_predict_f1.append('1')
	f1_ma = f1_score(Y_test_f1, Y_predict_f1, average='macro')
	f1_mi = f1_score(Y_test_f1, Y_predict_f1, average='micro')
	print 'MSE: ', mse
	print 'f1 macro: ', f1_ma, '\tf1 micro: ', f1_mi
	return mse, f1_ma, f1_mi

def readfile(path):
	tweet = []
	target = []
	sentiment = []
	snippets = []
	with open(path, "r") as fid:
		for l in fid:
			item = l.split("\t")
			tweet.append(item[0])
			target.append(item[1])
			sentiment.append(item[2].replace('$', ''))
			snippets.append(item[3])
	return tweet, target, sentiment, snippets

def main():
	trainfile = "train.csv"
	testfile = "test.csv"
	wordembedding = "word2vec_model"
	train_tweet, train_target, train_sentiment, train_snippet = readfile(trainfile)
	test_tweet, test_target, test_sentiment, test_snippet = readfile(testfile)
	train_predict(train_tweet, train_target, train_sentiment, test_tweet, test_target, test_sentiment, wordembedding, train_snippet, test_snippet)

if __name__ == "__main__":
	main()