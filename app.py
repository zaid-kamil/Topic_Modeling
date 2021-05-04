from json import load
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import pickle
import os
from pycaret.datasets import get_data
from pycaret.nlp import *

def save_var(obj, path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_var(path):
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)

st.title("Topic Modeling",)
st.success('Our Topic modeling tool is a frequently used text-mining tool for discovery of hidden semantic structures in a text body')


o = st.radio("select option",("About the Project","Upload","Genrate Topics"))

if o == "About the Project":
    st.markdown('''
                Topic modeling discovers abstract topics that occur in a collection of documents (corpus) using 
                a probabilistic model. It's frequently used as a text mining tool to reveal semantic structures 
                with in a body of text. A document about a specific topic will have certain words appearing more 
                frequently than others.
                ''')
    st.markdown('''
                 In machine learning and natural language processing, a topic model is a type of statistical model 
                for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a 
                frequently used text-mining tool for discovery of hidden semantic structures in a text body.
                ''')

data= None
sample =500
if o =="Genrate Topics":
    
    a = st.sidebar.selectbox('Select a dataset',("wikipedia","kiva"))
    sample = st.sidebar.slider("select sample size",min_value=500,max_value=2000)
    if a and st.sidebar.button("fetch"):
        data = get_data(a)
        if sample > data.shape[0]:
            sample=data.shape[0]
        data = data.sample(sample, random_state=786).reset_index(drop=True)
        st.write(data)
        save_var(data,"data.pk")
if isinstance(load_var("data.pk"),pd.DataFrame):
    try:
        data=load_var("data.pk")
    except:
        pass
    b = st.sidebar.selectbox('Select a column',data.columns.tolist())
    c = st.sidebar.number_input("Num of topic categories to Find", 1,5,value=3)
    if st.sidebar.button("Analyse"):
        exp_nlp101 = setup(data = data, target = b, session_id = 123)
        lda2 = create_model('lda', num_topics = c, multi_core = True)
        st.write(lda2)
        save_var(lda2,"LDA.pk")

if load_var("LDA.pk"):
    lda=load_var("LDA.pk")
    lda_results = assign_model(lda)
    st.write(lda_results.head())
    cols = lda_results.columns.tolist()
    topic = st.selectbox("select col",[col for col in lda_results.columns.tolist() if "topic" in col.lower()])
    plot_model(lda, plot = 'frequency', display_format='streamlit')

    
            
if o =="Upload":
    file = st.file_uploader('upload data (less than 25mb)',type=['txt','docx','pdf'])
    if file and st.button('Upload'):
        path = os.path.join("uploads",f'{file.name}')
        with open(path,'wb') as f:
            with st.spinner("saving and processing data"):
                st.info("Upload Sucessfully now you can analyse")
    st.button("Analyse", key=1)


    

# exp_nlp101 = setup(data = data, target = 'Title', session_id = 123)

# lda = create_model('lda')

# print(lda)

# lda2 = create_model('lda', num_topics = 6, multi_core = True)

# print(lda2)

# lda_results = assign_model(lda)
# lda_results.head()

# 

# plot_model(plot = 'bigram')

# plot_model(lda, plot = 'frequency', topic_num = 'Topic 1')

# plot_model(lda, plot = 'topic_distribution')

# plot_model(lda, plot = 'tsne')

# plot_model(lda, plot = 'umap')

# save_model(lda,'Final LDA Model 08Feb2020')

# saved_lda = load_model('Final LDA Model 08Feb2020')