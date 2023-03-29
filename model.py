import pandas as pd
import numpy as np


DF=pd.read_csv("raw_data.csv")


DF.head()

pd.set_option("display.max_column", 50)


DF.info()

DF.shape

df=DF.copy()

features_list=['movie_title','genres','director_name','actor_1_name','actor_2_name','actor_3_name']


features=list(df.columns)

drop_features=[]

for i in features:
    if i not in features_list:
        drop_features.append(i)
    else:
        pass

df.drop(axis=1, columns=drop_features, inplace=True)


df.isnull().sum()

df.fillna(value='unknown',axis=1,inplace=True)

df.head()

df['genres']=df['genres'].str.replace('|',' ')

df.info()

df['total']=df['director_name']+df['actor_2_name']+df['genres']+df['actor_1_name']+df['actor_3_name']


df.head()





df.drop(axis=1,columns=['director_name','actor_2_name','genres','actor_1_name','actor_3_name'],inplace=True)

df.head()

df['movie_title']=df['movie_title'].str[:-1]

df['movie_title']=df['movie_title'].str.lower()

df.head()

df.to_csv('data.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv=CountVectorizer()

fv_matrix=cv.fit_transform(df['total'])

fv_matrix

print(fv_matrix)

similarity=cosine_similarity(fv_matrix)

similarity

similarity.dtype

pd.set_option('display.max_column',50000)

movie='John carter'

movie=movie.lower()


index=int()

if movie in df['movie_title'].unique():
    index=df.loc[df['movie_title']==movie].index[0]


List=list(enumerate(similarity[index]))


List=sorted(List,key=lambda x :x[1],reverse=True)


List=List[:11]


List_index=[]

for i,j in List:
    List_index.append(i)


movie_recomm_list=[]

for i in List_index:
    movie_recomm_list.append(df['movie_title'][i])
