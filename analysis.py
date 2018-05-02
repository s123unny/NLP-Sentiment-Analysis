from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import gensim
import argparse
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesRegressor
import operator
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from itertools import izip

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


def  train_predict(labels_train, msgs_train, labels_test, msgs_test, cashtags_train, cashtags_test, embeddings):
    print "loading features..."
    Y_train = np.array([float(x) for x in labels_train])
    Y_test = np.array([float(x) for x in labels_test])
    X_train_bow = []
    X_train_boe = []
    X_test_bow = []
    X_test_boe = []
    X_train = []
    X_test = []
    X_train_bos = []
    X_test_bos = []
    print 'train: ', len(Y_train), 'test:', len(Y_test)
    vec = CountVectorizer(ngram_range=(1,1))
    vec.fit(msgs_train)
    X_train_bow = vec.transform(msgs_train).toarray()
    X_test_bow = vec.transform(msgs_test).toarray()
    model = gensim.models.Word2Vec.load(embeddings) 
    w2v = dict(zip(model.wv.index2word, model.wv.syn0)) 
    vec = MeanEmbeddingVectorizer(w2v)
    vec.fit(msgs_train)
    X_train_boe = vec.transform(msgs_train)#.toarray()
    X_test_boe = vec.transform(msgs_test)#.toarray()
    print 'len x_Train_bow, X_test_bow', X_train_bow.shape , X_test_bow.shape
    print 'len x_Train_boe, X_test_boe', X_train_boe.shape , X_test_boe.shape
    X_train = np.concatenate((X_train_bow,X_train_boe), axis=1)
    X_test = np.concatenate((X_test_bow,X_test_boe), axis=1)
    print 'len x_Train_, X_test_', X_train.shape , X_test.shape
    clf = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
    #clf = MLPRegressor(hidden_layer_sizes=(5,) ,verbose=True)
    #clf = SVR(kernel='linear',C=1.0, epsilon=0.2)
    print "fiting.. extra trees."
    print Y_train
    clf.fit(X_train, Y_train)
    print "testing..."
    y_hat = clf.predict(X_test)
    np.savetxt("predict.csv", y_hat, delimiter='\n')
    res_dict = {}
    mse = mean_squared_error(Y_test, y_hat)
    #f1_ma = f1_score(Y_test, y_hat, average='macro')
    #f1_mi = f1_score(Y_test, y_hat, average='micro')
    print 'MSE: ', mse
    #print 'f1 macro: ', f1_ma, '\tf1 micro: ', f1_mi
    return mse


def readfile(path):
    labels = []
    msgs = []
    cashtags = []
    with open(path,"r") as fid:
        for l in fid:
            splt = l.strip('\n').lower().split("\t")
            labels.append(splt[2])
            msgs.append(splt[0])
            cashtags.append(splt[1].replace('$',''))
    return labels, msgs, cashtags


def main():
    train_file = "train_new.csv"
    test_file = "test_new.csv"
    embeddings = "word2vec_model_new"
    labels_train, msgs_train, cashtags_train = readfile(train_file)
    labels_test, msgs_test, cashtags_test = readfile(test_file)
    train_predict(labels_train, msgs_train, labels_test, msgs_test, cashtags_train, cashtags_test, embeddings)


if __name__ == "__main__":
    main()
