# NLP project - Sentiment Analysis

**A project on sentiment analysis in financial microblogs**

## Compile

### Features
```bash
$ python preprocessing.py [train_set.json] [train.csv] --model [vectors]
$ python preprocessing.py [test_set.json] [test.csv]
```

### Analysis and Save the models 
```bash
$ python2 analysis.py [train.csv] [test.csv] [vectors]
```

If wish to save the model and features, then run the line below
```bash
$ python2 analysis.py [train.csv] [test.csv] [vectors] --model [bos_feature] [boe_feature] [ETR_model] 
```

### Prediction
```bash
$ python2 predict.py [test.csv] [bos_feature] [boe_feature] [ETR_model]
```
