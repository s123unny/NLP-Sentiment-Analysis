
# coding: utf-8

# In[1]:


# Import the necessary modules
import pandas as pd
import numpy as np
import re
from nltk.tokenize import TweetTokenizer, WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#training Modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#Evaluation Modules
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#reading the dataset files
train_df = pd.read_json("training_set.json")
test_df = pd.read_json("test_set.json")
# hashtag_corpus = pd.read_json("NTUSDFinCorpus/NTUSD_Fin_hashtag_v1.0.json")
# word_corpus = pd.read_json("NTUSDFinCorpus/NTUSD_Fin_word_v1.0.json")
emoji_corpus = pd.read_json("NTUSD_Fin_emoji_v1.0.json")


# # Pre-Processing

# In[3]:


#placeholders 
rep = 0  #index for placeholders
p_mentions = [" @mentions ", " "]
p_cashtag = [" @cashtag ", " "]
p_url = [" @url ", " "]
stopwords = set(stopwords.words("english"))
qmark = " qmark "  
emark = " emark "


# In[4]:


w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# #### processing steps
# - lowercase conversion
# - replace mentions
# - replace cashtag
# - replace urls
# - replace special unicode characters (&, > , < ,' )
# - removing stopwords
# - lemmatization

# In[5]:


train_df['tweet'] = train_df['tweet'].str.lower()
train_df['tweet'] = train_df['tweet'].str.replace('([@][\w_-]+)', p_mentions[rep], case=False)
train_df['tweet'] = train_df['tweet'].str.replace('([$][a-z]+)', p_cashtag[rep], case=False)
train_df['tweet'] = train_df['tweet'].str.replace('http\S+|www.\S+', p_url[rep], case=False)
train_df['tweet'] = train_df['tweet'].str.replace('&amp', " & ", case=False)
train_df['tweet'] = train_df['tweet'].str.replace('&#39;', "'", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&gt;', " ", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&lt;', " ", case=False)
train_df['tweet'] = train_df['tweet'].str.replace('\?', qmark, case=False)
train_df['tweet'] = train_df['tweet'].str.replace('!', emark, case=False)
train_df['tweet'] = train_df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df['tweet'] = train_df['tweet'].apply(lemmatize_text)


# preprocessing steps taken for test dataset is the same for training dataset

# In[6]:


test_df['tweet'] = test_df['tweet'].str.lower()
test_df['tweet'] = test_df['tweet'].str.replace('([@][\w_-]+)', p_mentions[rep], case=False)
test_df['tweet'] = test_df['tweet'].str.replace('([$][a-z]+)', p_cashtag[rep], case=False)
test_df['tweet'] = test_df['tweet'].str.replace('http\S+|www.\S+',  p_url[rep], case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&amp', " & ", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&#39;', "'", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&gt;', " ", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('&lt;', " ", case=False)
test_df['tweet'] = test_df['tweet'].str.replace('\?', qmark, case=False)
test_df['tweet'] = test_df['tweet'].str.replace('!', emark, case=False)
test_df['tweet'] = test_df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['tweet'] = test_df['tweet'].apply(lemmatize_text)


# # Training the Model (Linear Regression) 

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( analyzer='word',
                      ngram_range=(1,3),
                      stop_words = 'english')


# In[10]:


model = cv.fit_transform(list(train_df["tweet"]))


# X = list(train_df["tweet"])
# y = train_df["sentiment"]
# X = cv.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.33, random_state=42 )

# In[11]:


cv.fit(train_df["tweet"])
X_train = cv.transform(train_df["tweet"])
X_test = cv.transform(test_df["tweet"])
y_train = train_df["sentiment"]
y_test = test_df["sentiment"]


# In[12]:


#linear regression model
from sklearn.linear_model import LinearRegression
log_model = LinearRegression(fit_intercept=True)
log_model = log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)


# # Evaluation

# In[13]:


def assignClasses(data): 
    value = list()
    for i in data: 
        if i > 0:
            value.append("bullish")
        elif i < 0:
            value.append("bearish")
        else:
            value.append("neutral")
    
    return value 


# In[14]:


n_y_test = assignClasses(y_test)
n_y_pred = assignClasses(y_pred)


# In[15]:


#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print("MSE: ", mean_squared_error(y_test, y_pred))
print('\n')
print("F1 Macro Avg: ", f1_score(n_y_test, n_y_pred, average='macro'))
print("F1 Micro Avg: ", f1_score(n_y_test, n_y_pred, average='micro'))
print('\n')
print("Classification Report  \n", classification_report(n_y_test, n_y_pred))

