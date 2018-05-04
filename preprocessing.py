'''do preprocess including 
    1. remove URL, cashtag, stopwords, replace number
    2. do word2vec
    3. dump new data file: [tweet]\t[target]\t[sentiment]
'''
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re
from nltk.corpus import stopwords
import argparse

def get_arg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', help='train w2v model')
    parser.add_argument("load")
    parser.add_argument("csv")
    args = parser.parse_args()
    return args

args = get_arg()
data = json.loads(open(args.load).read())
sentences = []
new_tweet = open(args.csv, "w")
cashtag = ["$"+chr(asci) for asci in range(97, 123)]
stop_words = set(stopwords.words('english'))
for item in data:
    tweet = item["tweet"].lower()
    tweet = re.sub("&#39;|\\u2019", "'", tweet)
    tweet = re.sub("&gt;|&quot;|&lt;|u[\da-f]{4}", " ", tweet)
    tweet = re.sub("\$[\d\.,\-\+]+", "DD", tweet)
    tweet = re.sub("\d+/\d+|\d+%", "RR", tweet)
    tweet = re.sub(">?\d+\-*\d*[mb]*", "NN", tweet)
    tweet = re.sub("http\S+|www\S+", " ", tweet)
    tweet = re.sub("&amp;|&", " & ", tweet)
    tweet = re.sub("\?", " Qmark ", tweet)
    tweet = re.sub("!", " Emark ", tweet)
    word_tokens = re.split("[<>\s.,(:)~\-\+\=]", tweet)
    word_tokens = [word for word in word_tokens if not any(i in word for i in cashtag) and len(word) > 0]
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    snippet = item['snippet']
    if type(snippet) == list:
        snippet = " ".join(snippet)
    snippet = snippet.lower()
    snippet = re.sub("&#39;|\\u2019", "'", snippet)
    snippet = re.sub("&gt;|&quot;|&lt;|u[\da-f]{4}", " ", snippet)
    snippet = re.sub("\$[\d\.,\-\+]+", "DD", snippet)
    snippet = re.sub("\d+/\d+|\d+%", "RR", snippet)
    snippet = re.sub(">?\d+\-*\d*[mb]*", "NN", snippet)
    snippet = re.sub("http\S+|www\S+", " ", snippet)
    snippet = re.sub("&amp;|&", " & ", snippet)
    snippet = re.sub("\?", " Qmark ", snippet)
    snippet = re.sub("!", " Emark ", snippet)
    word_token = re.split("[<>\s.,(:)~\-\+\=]", snippet)
    word_token = [word for word in word_token if not any(i in word for i in cashtag) and len(word) > 0]
    filtered_snippet = [w for w in word_token if not w in stop_words]
    sentences.append(filtered_sentence)
    new_tweet.write(' '.join(filtered_sentence)+'\t'+item["target"]+'\t'+str(item["sentiment"])+'\t'+' '.join(filtered_snippet)+'\n')

if args.model != None:
    model = Word2Vec(sentences, size=10, min_count=1, workers=4)
    model.save(args.model)
    #model.wv.save_word2vec_format("vectors", binary=False)




