import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("/kaggle/input/images/Youtube01-Psy.csv")
print(data.sample(5))

data = data[["CONTENT","CLASS"]]
print(data.sample(5))

data["CLASS"] = data["CLASS"].map({0: "Not Spam",
                                   1: "Spam Comment"})
print(data.sample(5))

#Traning of Classification Model using Bernaouli Naive Bayes algorithm
x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])
cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(x,y, 
                                             test_size=0.2,
                                             random_state=42)
model = BernoulliNB()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

sample = "Lack of information"
data = cv.transform([sample]).toarray()
print(model.predict(data))
['Not Spam'] # <-- this is the predicted output 

sample = "https://www.geeksforgeeks.org/"
data = cv.transform([sample]).toarray()
print(model.predict(data))
#['Spam Content'] <--- this is the predicted output 
