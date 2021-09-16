##################################################################################################################
############################################# RFM SEGMENTATION FOR IMDB TOP 250 ##################################
##################################################################################################################

##################################################################################################################
# Libraries
##################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################################################################################
# Reaching and Preparing Data
##################################################################################################################

url = "https://www.imdb.com/chart/top/"
response = requests.get(url)
print(response)

html_content = response.content

soup = BeautifulSoup(html_content, "html.parser")

print(soup.find_all("a"))

titles = soup.find_all("td", {"class": "titleColumn"})
ratings = soup.find_all("td", {"class": "ratingColumn imdbRating"})

title = []
rating = []
for i, y in zip(titles, ratings):
    i = i.text
    i = i.strip()
    i = i.replace("\n", "")
    i = i[2:]
    i = i.strip(" ")

    y = y.text
    y = y.strip()
    y = y.replace("\n", "")
    y = y.strip(" ")

    title.append(i)
    rating.append(y)

title = pd.DataFrame(title)
rating = pd.DataFrame(rating).astype(float)

data = pd.concat([title, rating], axis = 1, ignore_index = True)

year = [i[0][-5:-1] for i in data.values]
year = pd.DataFrame(year).astype(int)


data = pd.concat([year, data], axis = 1, ignore_index = True)
data.columns = ["Year", "Titles", "Ratings"]

data.head()
df = data.copy()

##################################################################################################################
# Classification by Using Pareto
##################################################################################################################

df["Pareto"] = ["High Rate" if i > df["Ratings"].quantile(0.80) else "Low Rate" for i in df["Ratings"]]

df["Pareto"].value_counts().plot(kind = "bar", rot = 0)

df.groupby("Pareto").agg(["mean", "count", "sum"])

##################################################################################################################
# Identfying RFM Metrics
##################################################################################################################

rfm = df.groupby("Titles").agg({"Year": lambda x: 2021 - x,
                                "Ratings": lambda x: x})

rfm.head()
rfm.columns = ["Recency", "Monetary"]

rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4, 3, 2, 1])
rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = rfm["Recency_Score"].astype(str) + rfm["Monetary_Score"].astype(str)


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm["RFM_SCORE"] = rfm["RFM_SCORE"].astype(int)
rfm = rfm.reset_index()

rfm.groupby("segment").agg(["mean", "count"])

