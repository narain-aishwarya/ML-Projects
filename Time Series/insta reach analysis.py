#The data set used in this project is from Kaggle and the project is live on Kaggle.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from wordcloud import WordCloud ,STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("/kaggle/input/instagram-reach-analysis-case-study/Instagram_data_by_Bhanu.csv",encoding = 'latin1')
print(data.head())

data = data.dropna()
data.info()

plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions from Home")
sb.distplot(data['From Home'])
plt.show()

plt.figure(figsize=(10,8))
plt.title("Distribution of Impression from Hashtags")
sb.distplot(data['From Hashtags'])
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impression from Explore")
sb.distplot(data ['From Explore'])
plt.show()

#Now to look at the percentage of impression from various source
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
comments = data["Comments"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags', 'Comments','From Other']
values = [home, hashtags,comments,other]

fig = px.pie(data, values=values, names=labels, 
             title= 'Impressions on Instagram Posts From Various Sources',hole=0.5)
fig.show()
#the above code will display a "pie-chart" showing result of various impression.

#Analyzing the Content.
#Creating a wordColud of the most used words for Caption:
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords, background_color = 'white').generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()
#the above code create a whitebord with the most used words in caption of an insta post.

#Now for the Hashtag column:
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords, background_color = 'white').generate(text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#the above code create a whitebord with the most hashtags word.

#For analyzing Relationships:
figure = px.scatter(data_frame = data , x='Impressions',y='Likes', size='Likes', trendline='ols', 
                    title = "Relationship betweem Likes and Impression")
figure.show()
#this will print a scattering plot chart , which shows the relationship between Likes and Impression.

figure = px.scatter(data_frame = data , x='Impressions', y='Comments', size='Comments', trendline='ols',
                   title = "Relationship between Impression and Comments")
figure.show()

figure = px.scatter(data_frame = data, x='Impressions', y='Shares', size='Shares', trendline='ols',
                   title = "Relationship between Shares and Total Impressions")
figure.show()

figure = px.scatter(data_frame = data, x='Impressions', y='Saves', size='Saves', trendline='ols',
                   title="Relationship between Total Saves and Total Likes")
figure.show()

#Correlation of all the columns with the Impressions columns:
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))
#the ouput showa that the more like , hashtags and saves wil help in getting more reach on instagram.

#Analysis convertion rate:
#In instagram , conversion rate means how many follower you are getting from the no.of profile visit from a post.
#formula :- (Follows/Profile Visit)*100
conversion_rate = (data['Follows'].sum()/data['Profile Visits'].sum())*100
print(conversion_rate)

figure = px.scatter(data_frame = data, x="Profile Visits",
                   y="Follows", size="Follows", trendline="ols",
                   title="Relation between the no.of Profile Visits and the Followers gained")
figure.show()

#Training the Model to predict the reach of an instagram post:
#Spliting the data into training and test set.
x = np.array(data[["Likes","Follows","Saves","Comments","Shares","Profile Visits"]])
y = np.array(data["Impressions"])
xtrain,xtest,ytrain,ytest = train_test_split(x,y,
                                            test_size = 0.2,
                                            random_state = 42)
#training the model:
model = PassiveAggressiveRegressor()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)

#Features = ["Likes","Shares","Saves","Comments","Profile Visits","Follow"]
result = np.array([[282, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(result)

"""This is the predicted impression on an instagram post which was predicted by using various features which are present in Instagram.
For creator's who want to learn how to make there post reach a wider audience , it is important to analyze their data.
This model will help in the prediction of reach analysis of an Instagram post"""
