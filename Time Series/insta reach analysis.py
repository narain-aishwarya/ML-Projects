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

# To Analyse content of Insta post:
