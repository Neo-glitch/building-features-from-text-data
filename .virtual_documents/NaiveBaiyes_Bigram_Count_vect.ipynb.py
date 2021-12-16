import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.naive_bayes import GaussianNB


# clf prob is to clf text data into topics(14 classes)
dbpedia_df = pd.read_csv("./datasets/dbpedia_csv/train.csv", skiprows = 1, names = ["Label", "Name", "Text"])

dbpedia_df.sample(6)


dbpedia_df.shape


# dbpedia_df["Label"].unique()


# sample just 8,000 rows of 10, close to 600k is to big to work with at once
dbpedia_df = dbpedia_df.sample(10000, replace = False)


dbpedia_df.shape


X = dbpedia_df["Text"]
Y = dbpedia_df["Label"]


X.head()


# helper to check model performance
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize = True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)
    prec = precision_score(y_test, y_pred, average = "weighted")
    recall = recall_score(y_test, y_pred, average = "weighted")
    
    print("Len test data: ", len(y_test))
    print("accuracy count: ", num_acc)
    print("accuracy_score: ", acc)
    print("precision_score: ", prec)
    print("recall_score: ", recall)


count_vectorizer = CountVectorizer(ngram_range=(2,2))

feature_vector = count_vectorizer.fit_transform(X)

feature_vector.shape


# conv sparse matrix of festure vect to dense vector needed by naiveBaiyers clf
X_dense = feature_vector.todense()

X_dense


# not working due to low ram on ma pc at moment of testing
x_train, x_test, y_train, y_test = train_test_split(X_dense, Y, test_size = 0.2)


clf = GaussianNB().fit(x_train, y_train)


y_pred = clf.predict(x_test)

y_pred


summarize_classification(y_test, y_pred)


# compare side by side pred vs actual

y_test = np.array(y_test)

pred_results = pd.DataFrame({"y_pred": y_pred,"y_test": y_test})

pred_results.sample(10)



















































































































































