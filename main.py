#import the necessary modules
#global modules
import pandas as pd
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#training modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#evaluation modules
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error


#Global Variables
train_df = pd.read_json("training_set.json")
test_df = pd.read_json("test_set.json")



# Preprocessing Methods
def preprocessing():
    #placeholders
    mentions = " @mentions "
    cashtag = " @cashtag "
    url = " @url "
    qmark = " qmark "  
    emark = " emark "
    percent = "percent "
    dollar = "dollar "
    stopwords = set(stopwords.words("english"))


    train_df['tweet'] = train_df['tweet'].str.lower()
    train_df['tweet'] = train_df['tweet'].str.replace('([@][\w_-]+)',  mentions , case=False)
    train_df['tweet'] = train_df['tweet'].str.replace('([$][a-z]+)', cashtag, case=False)
    train_df['tweet'] = train_df['tweet'].str.replace('http\S+|www.\S+', url, case=False)

    #number processing +25% , -13% , 23% ==>  percentageincrease percentagedecrease
    #stopwords 
    #dollars  +$13.12 -$12.112 ==> dollarincease dollardecrease
    

    #print(train_df.tweet)

#Training the Models
def LinearRegressionModel():
    cv = CountVectorizer()

    X = list(train_df["tweet"])
    y = train_df["sentiment"]

    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.33, random_state=42 )
    log_model = LinearRegression()
    log_model = log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)

    print("=== Evaluating Model: Training Data ===")
    MSEevaluation(y_test, y_pred)
    F1evaluation(y_test,y_pred)
    print("/n")
    print("=== Evaluating Model: Test Data ===")
    print("** Upcoming ***")


#Evaluation Methods
def MSEevaluation(y_true, y_pred):
    print(" === MSE Score ===")
    print(mean_squared_error(y_true, y_pred))

def F1evaluation(y_true, y_pred):
    #Conversion of data type to "bullish, bearish and neutral"
    y_true = assignClasses(y_true)
    y_pred = assignClasses(y_pred)

    print(" === F1 Score ===")
    print(f1_score(y_true,y_pred,average='macro'))
    print(f1_score(y_true,y_pred,average='micro'))


def assignClasses(data):
    conditions = [
        (data['sentiment'] == 0) , 
        (data['sentiment'] <  0) ,
        (data['sentiment'] >  0)]
    choices = ['neutral', 'bullish', 'bearish']
    data['classes'] = np.select(conditions, choices, default='neutral')
    return data['classes']


def main():
    preprocessing()
    LinearRegressionModel()


if __name__ == "__main__":
    main()