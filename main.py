from flask import Flask,request,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def Similarity():
    df=pd.read_csv('data.csv')
    cv=CountVectorizer()

    fv_matrix=cv.fit_transform(df['total'])
    
    sim=cosine_similarity(fv_matrix)

    return df,sim

def recomm(movie):

    try:
        df.head()
        sim.shape
    except:
        df,sim=Similarity()
    
    movie=movie.lower()
 


    index=int()

    if movie in df['movie_title'].unique():
        index=df.loc[df['movie_title']==movie].index[0]


    List=list(enumerate(sim[index]))


    List=sorted(List,key=lambda x :x[1],reverse=True)


    List=List[:11]


    List_index=[]

    for i,j in List:
        List_index.append(i)


    movie_recomm_list=[]

    for i in List_index:
        movie_recomm_list.append(df['movie_title'][i])
    
    return movie_recomm_list


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    movie = request.args.get('movie')
    r = recomm(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__=='__main__':
    app.run()

           
           




