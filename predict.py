import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import gensim
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import argparse

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def get_arg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('test', help='test file')
    parser.add_argument('bos_feature', help='bos_feature')
    parser.add_argument('boe_feature', help='boe_feature')
    parser.add_argument('ETR_model', help='ETR_model')
    args = parser.parse_args()
    return args

def train_predict(test_tweet, test_target, test_sentiment, test_snippet, args):
	Y_test = np.array([float(x) for x in test_sentiment])
	X_test_boe = []
	X_test_bos = []
	X_test = []
	print 'test:', len(Y_test)

	with open(args.bos_feature, 'rb') as f:
            vec = pickle.load(f)
	X_test_bos = vec.transform(test_snippet).toarray()
	print 'len X_test_bos' , X_test_bos.shape

	with open(args.boe_feature, 'rb') as f:
            vec = pickle.load(f)
	X_test_boe = vec.transform(test_tweet)
	print "len X_test_boe", X_test_boe.shape
	
	X_test = np.concatenate((X_test_bos, X_test_boe), axis=1)

	with open(args.ETR_model, 'rb') as f:
            clf = pickle.load(f)
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
	args = get_arg()
	testfile = args.test
	test_tweet, test_target, test_sentiment, test_snippet = readfile(testfile)
	train_predict(test_tweet, test_target, test_sentiment, test_snippet, args)

if __name__ == "__main__":
	main()
