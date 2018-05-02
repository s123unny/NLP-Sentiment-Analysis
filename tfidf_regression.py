import nltk
import numpy as np
import json
import os
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from numpy import array

wordlist = list()
train = list()
test = list()
y = list()
ans = list()

elim = ['$','@',':']

def tweet2wordlist(tweet):
	words = tweet.lower().split()
	stops = set(stopwords.words("english"))
	words = [w for w in words if not w in stops]
	
	for item in elim:
		words = [w for w in words if not item in w]
	
	for w in words:
		if bool(re.search(r'#',w)):
			w = re.sub(r'#',"",w)
	for w in words:
		if bool(re.search(r'\d',w)):
			if bool(re.search(r'%',w)):
				if bool(re.search(r'-',w)):
					w = 'MP'
				else :
					w = 'P'
			elif bool(re.search(r'[a-zA-Z]',w)):
					w = 'NW'
			else:
				if bool(re.search(r'-',w)):
					w = 'MN'
				else :
					w = 'N' 		
		
	return(words) 

my_file = open('training_set.json', 'r')
line = my_file.readline()
my_file.close

for i in range(0,1396):
	text = json.loads(line)[i]
	train.append(text)

test_file = open('test_set.json', 'r')
line2 = test_file.readline()
test_file.close
for i in range(0,634):
	text = json.loads(line2)[i]
	test.append(text)

for i in range(0,1396):
	y.append(train[i]['sentiment'])

for i in range(0,634):
	ans.append(test[i]['sentiment'])
	
traindata = []
for i in xrange(0,1396):
	traindata.append(" ".join(tweet2wordlist(train[i]['tweet'])))

testdata = []
for i in xrange(0,634):
	testdata.append(" ".join(tweet2wordlist(test[i]['tweet'])))

tfv = TfidfVectorizer(max_features=None,
                      analyzer='word',
                      max_df = 0.8,
                      ngram_range=(1,3),
                      norm = 'l2',
                      use_idf = False,
                      stop_words = 'english')

tfv.fit(traindata)
X_train = tfv.transform(traindata)
X_test = tfv.transform(testdata)

model = LogisticRegression(penalty='l2',
                           tol=0.0001,
                           C=1,
                           fit_intercept=True,
                           intercept_scaling=1.0,
                           class_weight='balanced',
                           random_state=None)

model.fit(X_train,y)

result = model.predict(X_test)
yy = array(ans)

result = result.astype('float64')
yy = yy.astype('float64')

yp = []
yt = []
for i in xrange(0,634):
	if result[i] < 0.:
		yp.append('-1')
	elif result[i] > 0.:
		yp.append('1')
	else :
		yp.append('0')

y_pred = array(yp)
y_pred = y_pred.astype('int32')
y_pred.reshape((634,1))

for i in xrange(0,634):
	if yy[i] < 0.:
		yt.append('-1')
	elif yy[i] > 0.:
		yt.append('1')
	else :
		yt.append('0')

y_true = array(yt)
y_true = y_true.astype('int32')
y_true.reshape((634,1))

print result

print f1_score(y_true,y_pred,average='macro')
print f1_score(y_true,y_pred,average='micro')

print mean_squared_error(result,yy)

