# NLP project - Sentiment Analysis

**A project on sentiment analysis in financial microblogs**

## Compile

### Features
```bash
$ python preprocessing.py [rawdata/train_set.json] [train.csv] --model [model/word2vec_model]
$ python preprocessing.py [rawdata/test_set.json] [test.csv]
```

### Analysis and Save the models 
```bash
$ python2 analysis.py [train.csv] [test.csv] [model/word2vec_model]
```

If wish to save the model and features, then run the line below
```bash
$ python2 analysis.py [train.csv] [test.csv] [model/word2vec_model] --model [model/bos_feature] [model/boe_feature] [model/ETR_model] 
```

### Prediction
```bash
$ python2 predict.py [test.csv] [model/bos_feature] [model/boe_feature] [model/ETR_model]
```
