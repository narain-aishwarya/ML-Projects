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