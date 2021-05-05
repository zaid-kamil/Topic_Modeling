import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import joblib
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import gensim
from itertools import combinations
from db import TextFile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import time
import plotly.express as px


plt.style.use("seaborn")
matplotlib.rcParams.update({"font.size": 14})

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_file(file,path):
    try:
        db = opendb()
        ext = file.type.split('/')[1] # second piece
        text = TextFile(filename=file.name,extension=ext,filepath=path)
        db.add(text)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

def load_file_raw_data(path):
    raw_documents = []
    snippets = []
    pb = st.sidebar.progress(0)
    with open(path,errors='ignore') as fin:
        lines = fin.readlines()
        total_lines = len(lines)
        for idx,line in enumerate(lines):
            text = line.strip()
            raw_documents.append(text)
            # keep a short snippet of up to 100 characters as a title for each article
            snippets.append( text[0:min(len(text),100)] )
            pb.progress((idx+1)*100//total_lines)
            

    # print("Read %d raw text documents" % len(raw_documents))
    return raw_documents,snippets

def load_stopwords():
    custom_stop_words = []
    with open( "stopwords.txt") as fin:
        for line in fin.readlines():
            custom_stop_words.append( line.strip() )
    print("Stopword list has %d entries" % len(custom_stop_words) )
    return custom_stop_words

def vectorize_data(raw_documents, custom_stop_words ):
    # use a custom stopwords list, set the minimum term-document frequency to 20
    vectorizer = CountVectorizer(stop_words = custom_stop_words, min_df = 20)
    A = vectorizer.fit_transform(raw_documents)
    print( "Created %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
    return vectorizer, A

def get_unique_terms(vectorizer):
    terms = vectorizer.get_feature_names()
    print("Vocabulary has %d distinct terms" % len(terms))
    return terms

def save_raw_article(snippets,A, terms, path = "articles-raw.pkl"):
    joblib.dump((A,terms,snippets), path)
    return True

def save_tfidf_article(snippets,A, terms, path = "articles-tfidf.pkl"):
    joblib.dump((A,terms,snippets), path)
    return True

def save_model(snippets,W, H, terms, datapath, path = "articles-model-nmf.pkl"):
    joblib.dump((W,H,terms,snippets,datapath),path)
    return True

def load_raw_article(path = "articles-raw.pkl"):
    (A,terms,snippets) = joblib.load(path)
    print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
    return A, terms, snippets

def load_NMF_model(path = "articles-model-nmf.pkl"):
    (W,H,terms,snippets,datapath) = joblib.load(path)
    print( "Loaded NMF model" )
    return W,H,terms,snippets,datapath

def load_tfidf_article(path="articles-tfidf.pkl"):
    (A,terms,snippets) = joblib.load(path)
    print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
    return A, terms, snippets

def get_Tfidf_document(raw_documents,custom_stop_words):
    # we can pass in the same preprocessing parameters
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, min_df = 20)
    A = vectorizer.fit_transform(raw_documents)
    print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )
    # extract the resulting vocabulary
    terms = vectorizer.get_feature_names()
    print("Vocabulary has %d distinct terms" % len(terms))
    return A,vectorizer,terms

def rank_terms( A, terms ):
    # get the sums over each column
    sums = A.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

def create_model(tfidf_matrix, k=10):
    model = decomposition.NMF( init="nndsvd", n_components=k ) 
    # apply the model and extract the two factor matrices
    W = model.fit_transform( tfidf_matrix )
    H = model.components_
    return model, W, H

def get_descriptor( terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
    return top_terms

def plot_top_term_weights(terms,H,topic_index,top):
    '''
    ### get the top terms and their weights
    plot_top_term_weights( terms, H, 3, 10 )
    '''
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    # note we reverse the ordering for the plot
    top_terms.reverse()
    top_weights.reverse()
    # create the plot
    fig,ax = plt.subplots(figsize=(13,8))
    # add the horizontal bar chart
    ypos = np.arange(top)
    plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)
    plt.xlabel("Term Weight",fontsize=14)
    plt.title(f'the top terms and their weights for {topic_index}')
    plt.tight_layout()
    return fig

def get_top_snippets( all_snippets, W, topic_index, top ):
    '''
    ### reverse sort the values to sort the indices
    ```
    topic_snippets = get_top_snippets( snippets, W, 0, 10 )
    for i, snippet in enumerate(topic_snippets):
        print("%02d. %s" % ( (i+1), snippet ) )
    ```
    '''
    top_indices = np.argsort( W[:,topic_index] )[::-1]
    # now get the snippets corresponding to the top-ranked indices
    top_snippets = []
    for doc_index in top_indices[0:top]:
        top_snippets.append( all_snippets[doc_index] )
    return top_snippets

def get_best_topic_number(A, kmin=1, kmax=15):
    topic_models = []
    # try each value of k
    for k in range(kmin,kmax+1):
        print("Applying NMF for k=%d ..."%k)
        model = decomposition.NMF( init="nndsvd", n_components=k ) 
        W = model.fit_transform( A )
        H = model.components_    
        # store for later
        topic_models.append((k,W,H))
    return topic_models

class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 2:
                    tokens.append( tok )
            yield tokens

def get_word2vec_model(raw_documents, custom_stop_words):
    docgen = TokenGenerator( raw_documents, custom_stop_words )
    # the model has 500 dimensions, the minimum document-term frequency is 20
    
    w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=20, sg=1)
    print( "Model has %d terms" % len(w2v_model.wv.vocab) )
    w2v_model.save("w2v-model.bin")
    return w2v_model

def load_w2v_model(path="w2v-model.bin"):
    return gensim.models.Word2Vec.load(path)

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            try:
                sc = w2v_model.similarity(pair[0], pair[1])
            except:
                sc = 0
            pair_scores.append(sc)
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

st.set_page_config(layout='wide');
st.title("Topic Modeling",)

o = st.sidebar.selectbox("select option",("About the Project","upload Text files","Topics Modelling Analysis","Topic Visualization",'Correct number of topics'))

if o == "About the Project":
    c1,c2 = st.beta_columns(2)
    c2.markdown('''
                Topic modeling discovers abstract topics that occur in a collection of documents (corpus) using 
                a probabilistic model. It's frequently used as a text mining tool to reveal semantic structures 
                with in a body of text. A document about a specific topic will have certain words appearing more 
                frequently than others.
                ''')
    c1.image('img.png',use_column_width=True)
    c2.markdown('''
                 In machine learning and natural language processing, a topic model is a type of statistical model 
                for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a 
                frequently used text-mining tool for discovery of hidden semantic structures in a text body.
                ''')
    c2.warning('Our Topic modeling tool is a text-mining tool for discovery of hidden semantic structures in a text body')
    
if o =="upload Text files":
    files = st.file_uploader('select a text file for extract topics',type=['txt'],accept_multiple_files=True)
    if files and st.button("upload"):
        for file in files:
            path = os.path.join('filedata',file.name)
            with open(path,'wb') as f:
                f.write(file.getbuffer())
                status = save_file(file,path)
                if status:
                    st.sidebar.success("file uploaded")
                
                else:
                    st.sidebar.error('upload failed')

if  o =="Topics Modelling Analysis":
    dbase =opendb()
    results = dbase.query(TextFile).all()
    dbase.close()
    st.sidebar.title('Project option panel')
    doc = st.sidebar.selectbox("select a text documents",results)
    if doc:
        if st.sidebar.checkbox('view raw document data'):
            with open(doc.filepath,encoding='utf-8') as f:
                st.markdown(f'```{f.read()}```')
        st.sidebar.info("AI topic modelling process")
        k= st.sidebar.number_input("select the starting number of topic",min_value=2, max_value=10,value=5)
        if st.sidebar.button("start"):
            with open(doc.filepath,encoding='utf-8') as f:
                raw_documents,snippets = load_file_raw_data(doc.filepath)
                st.sidebar.text('STEP 1. extracted sentences')
                
                stopwords = load_stopwords()
                st.sidebar.text('STEP 2. loaded stopwords')

                vectorizer, vectorA = vectorize_data(raw_documents,stopwords)
                st.sidebar.text('STEP 3.vectorized data')

                unique_terms = get_unique_terms(vectorizer)
                st.subheader('Top hundreds unique characters in dataset')
                st.markdown(f'```{unique_terms[:len(unique_terms) if len(unique_terms)<100 else 100]}```')
                
                tfidfA,tfvectorizer,terms = get_Tfidf_document(raw_documents,stopwords)
                st.sidebar.text('STEP 4.created TF-IDF matrix')
                st.info(f'Data has {len(terms)} unique terms')
                save_tfidf_article(snippets,tfidfA, terms, path = "articles-tfidf.pkl")

                terms_ranking =rank_terms(tfidfA,terms)
                df = pd.DataFrame(terms_ranking,columns=['word','rank %'])
                st.subheader('Ranking of unique terms on the basis of occurance')
                fig = px.bar(df,y='rank %',x='word',hover_name='word',color='rank %')
                st.plotly_chart(fig,use_container_width=True)

                with st.spinner("Model training in progress"):
                    model, W, H = create_model(tfidfA)
                    st.sidebar.text('STEP 5. generted base model')

                    descriptors = []
                    for topic_index in range(k):
                        descriptors.append( get_descriptor( terms, H, topic_index, 10))
                        str_descriptor = ", ".join( descriptors[topic_index])
                        st.markdown(f"""
                        ```
                        {"Topic %02d: %s" % (topic_index+1, str_descriptor)}
                        ```
                        """)
                    with st.spinner("saving the model"):
                        save_model(snippets,W, H, terms, doc.filepath, path = "articles-model-nmf.pkl")
                        st.balloons()

if o == "Topic Visualization":
    with st.spinner("model loading"):
        try:
            W,H,terms,snippets,datapath = load_NMF_model()
            st.sidebar.success("NMF model loaded into memory")
            c1 ,c2 = st.beta_columns(2)
            topic_index = c1.slider("select a topic to visualize relation",0,10)
            top = c2.slider("select number of words to get reference from",1,len(terms),value=10)
            fig = plot_top_term_weights( terms, H, topic_index, top )
            if fig!=None:
                st.pyplot(fig)
        except Exception as e:
            st.sidebar.error(e)

if o == 'Correct number of topics':

        tfidfA,terms,snippets = load_tfidf_article(path = "articles-tfidf.pkl")
        W,H,terms,snippets,datapath = load_NMF_model()
        with open(datapath,encoding='utf-8') as f:
            raw_documents,snippets = load_file_raw_data(datapath)
            st.sidebar.success("models loaded into memory")
        c1,c2 = st.beta_columns(2)

        kmin = c1.number_input("select min topic number",1,9,value=1)
        kmax = c2.number_input("select min topic number",kmin,10,value=2)
        topic_models = get_best_topic_number(tfidfA, kmin, kmax)
        with st.spinner("please wait, creating a word 2 vector model"):
            w2v_model = get_word2vec_model(raw_documents,snippets)
            st.balloons()

        k_values = []
        coherences = []
        for (k,W,H) in topic_models:
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = []
            for topic_index in range(k):
                term_rankings.append( get_descriptor( terms, H, topic_index, 10 ) )
            # Now calculate the coherence based on our Word2vec model
            k_values.append( k )
            coherences.append( calculate_coherence( w2v_model, term_rankings ) )
            st.info(f"K={k:02d}: Coherence={coherences[-1]:.4f}")
            fig,ax = plt.subplots(figsize=(10,5))
            # create the line plot
            plt.plot( k_values, coherences )
            plt.xticks(k_values)
            plt.xlabel("Number of Topics")
            plt.ylabel("Mean Coherence")
            plt.scatter( k_values, coherences, s=120)
            ymax = max(coherences)
            xpos = coherences.index(ymax)
            best_k = k_values[xpos]
            plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
            st.pyplot(fig)



