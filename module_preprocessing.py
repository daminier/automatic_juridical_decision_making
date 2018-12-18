#!/usr/bin/env python
# coding: utf-8

import os 
import sys
import shutil
import random
import re
import regex
import string 
import pandas as pd
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

def create_vector_dataset(df, model,df_vector=pd.DataFrame()):
    '''
     This function creates a well structured dataset, which consists of n columns: n features and the tag.
     It transforms strings to vectors through the model Doc2Vec.
     which consists of n columns: n features and the tag. 
      Args:
         df (DataFrame): an semi-structured dataframe, look at create_raw_dataset.  
         model (Doc2Vec) : a trained Doc2Vec model.
         df_vector (DataFrame): an initialized dataframe with shape (n_features+1,).
      Returns :
         (DataFrame) a Dataframe with shape (n_features+1,n_samples). 
    '''
    
    for index, sample in df.iterrows():
        vector = model.infer_vector(doc_words=sample['raw_text'], steps=20)
        dic = {}
        for i in range(len(vector)):
            dic['y_' + str(i)] = vector[i]
        dic['tag'] = sample['tag']
        df_vector = df_vector.append(dic, ignore_index=True)
    df_vector.loc[:, 'y_0':'y_' + str(len(vector) - 1)] = normalize(df_vector.loc[:, 'y_0':'y_' + str(len(vector) - 1)],norm='l2', axis=0)
    
    return df_vector


def apply_preprocessing(all_text,start="PROCEDURE",end="THE LAW",verbose=False) :  
    '''
    This function applyies preprocessing on a string. It returns a list of words.
     Args:
         all_text (string): a simple string.
         start (string): the inital string.
         end (string): the end string.
         verbose (Bool)
     Returns :
         (list) A list of string. 
    '''
    with open("stop.txt","rt") as f : 
            stop_w = f.read().split()
    stop_words= stopwords.words('english')
    for x in stop_w : stop_words.append(x)

    all_text = (all_text.split(start))[1].split(end)[0]
    temp_text = ""
    #deliting all the sentences which match with ...
    for line in all_text.lower().split("\n") :
        if ("the court" not in line) and ("echtr" not in line) and ("european court of human rights" not in line) :
                temp_text+=" "+line
    doc=temp_text
    doc = re.sub('\s+',' ', doc).strip()
    doc= doc.lower()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = regex.sub('', doc)
    doc= re.sub(r'[^\w]', ' ', doc)
    doc = re.sub('\s\d+\s[a-zA-Z]+\s+\d{4}', '',doc).strip()
    doc = re.sub('article\s*(\d+)', r'article\1', doc).strip()
    for word in stop_words :
        doc = re.sub(''.join((r'\b{}\b'.format(word))), ' ',doc)
    rx = r'(articles)(\s*[\d\s]*\d)\b'
    doc = re.sub(rx, lambda x: "{}{}".format(x.group(1), "-".join(x.group(2).split())), doc )
    doc = re.sub('(\s\d+)', '', doc).strip()
    doc = re.sub(r'[^\x00-\x7f]',r'', doc).strip()
    ps = PorterStemmer()
    if verbose : 
        print(str(len(all_text)-len(doc)) +" words has been deleted.")
        print("before: "+ str(len(all_text))+ " after: " + str(len(doc)))

    return [ps.stem(word) for word in doc.split()]


def apply_preprocessing_simple(all_text,start="PROCEDURE",end="THE LAW",verbose=False) :  
    '''
    This function applyies preprocessing on a string. It returns a list of words.
     Args:
         all_text (string): a simple string.
         start (string): the inital string.
         end (string): the end string.
         verbose (Bool)
     Returns :
         (list) A list of string. 
    '''
  
    stop_words= stopwords.words('english')
    doc = (all_text.split(start))[1].split(end)[0]
    
    temp_text = ""
    #deliting all the sentences which match with ...
    for line in doc.lower().split("\n") :
        if ("the court" not in line) and ("echtr" not in line) and ("european court of human rights" not in line) :
                temp_text+=" "+line
    doc=temp_text
    
    doc = re.sub('\s+',' ', doc).strip()
    doc= doc.lower()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = regex.sub('', doc)
    doc= re.sub(r'[^\w]', ' ', doc)
    doc = re.sub(r'[^\x00-\x7f]',r'', doc).strip()
    if verbose : 
        print(str(len(all_text)-len(doc)) +" words has been deleted.")
        print("before: "+ str(len(all_text))+ " after: " + str(len(doc)))

    return [word for word in doc.split() if word not in stop_words]

def apply_preprocessing_no_stopw(all_text,start="PROCEDURE",end="THE LAW",verbose=False) :  
    '''
    This function applyies preprocessing on a string. It returns a list of words.
     Args:
         all_text (string): a simple string.
         start (string): the inital string.
         end (string): the end string.
         verbose (Bool)
     Returns :
         (list) A list of string. 
    '''
  
    doc = (all_text.split(start))[1].split(end)[0]
    
    temp_text = ""
    #deliting all the sentences which match with ...
    for line in doc.lower().split("\n") :
        if ("the court" not in line) and ("echtr" not in line) and ("european court of human rights" not in line) :
                temp_text+=" "+line
    doc=temp_text
    
    doc = re.sub('\s+',' ', doc).strip()
    doc= doc.lower()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = regex.sub('', doc)
    doc= re.sub(r'[^\w]', ' ', doc)
    doc = re.sub(r'[^\x00-\x7f]',r'', doc).strip()
    if verbose : 
        print(str(len(all_text)-len(doc)) +" words has been deleted.")
        print("before: "+ str(len(all_text))+ " after: " + str(len(doc)))

    return [word for word in doc.split()]




#with open("crystal_ball_data/train/Article3/violation/001-150771.txt","rt") as f : 
      # text = apply_preprocessing(text,verbose=True)   
      # print(text)

import shutil

def merge_dataset(path_test, path_training,path,verbose=True) :
    """ 
    This function merges the files containted in test20 folder with the files in training folder in the crystal_ball_data daset.
    Note: it was an error splitting the dataset before. 
     Args:
        path_test (string): .
        path_training (string): .
        verbose (bool): .
     Returns:
        None.
    """
    try :
        if verbose : 
            print("Files in "+path_test+": "+ str(len(os.listdir(path_test)[1:])))
            print("Files in "+path_training+": "+ str(len(os.listdir(path_training)[1:])))
        for filename in os.listdir(path_test)[1:] :
            shutil.move(path_test+filename, path+filename)
        for filename in os.listdir(path_training)[1:] :
            shutil.move(path_training+filename, path+filename)       
    except Exception as e : print(e) 
    finally : 
        if verbose :
            print("Files in "+path+": "+ str(len(os.listdir(path_training)[1:])))

    
from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument 

def generate_model(vector_size=100,verbose=True) :
    documents=[] 
    count=0
    if verbose : print("start tagging the document")
    for a in range(2,15) :
             for b in ["/non-violation/","/violation/"] :
                path= "originale/crystal_ball_data/train/Article"+str(a)+str(b)
                for filename in os.listdir(path)[1:] : 
                    try :
                        with open(path+filename,"rt") as f :                                               
                            documents.append(TaggedDocument(words=apply_preprocessing(f.read()),tags=[1 if "non" in b else 0])) #tags=[1 if "non" in b else 0]
                            count+=1
                    except IndexError : 
                        print("Index Error at ",filename)
                        os.remove(path+filename)
    if verbose :       
        print("should be the same: ",count,len(documents))                    
        print("training the model")
    model = Doc2Vec(documents=documents,vector_size=vector_size,epochs=20, min_count=100, workers=4,dm=1)
    model.save("All_art_D2V_model_tagclass")


def create_raw_dataset(paths,df=pd.DataFrame(data= {'raw_text':[], 'tag':[]}),save=False,start="PROCEDURE",end="THE LAW",seed=-1) :
    '''
     This function creates a raw dataset, which consists of three columns: "index",raw_text" and "tag". If save= True performs the dataframe as a csv. 
     
      Args:
         path (list of string): the folders where the files are.
         df (DataFrame): an empty (or not) dataframe.
         save (Bool) : save flag.
         start (str) : 
         end (str) : 
         seed (int): seed for reproducibility.
         
      Returns :
         (DataFrame) a Dataframe with shape (3 columns,num_samples) . 
    '''


    for path in paths :
            for filename in os.listdir(path)[1:] :
                dic={}
                try:
                    with open(path+filename,"rt") as f :
                        dic['raw_text']  = apply_preprocessing(f.read(),start,end)
                except IndexError : print("Index Error at ",filename) 
                if "non-violation" in path :
                    dic['tag']=1 # "non-violation" 
                else :
                    dic['tag']=0 # "violation"
                df = df.append(dic,ignore_index=True)    
    if seed > 0 :
        random.seed(seed)
        df = df.sample(frac=1,random_state=seed).reset_index(drop=True)    
    else :
        df = df.sample(frac=1).reset_index(drop=True)
    if save :
        file_name = re.search(r'(Article\d+)', path)
        df.to_csv("crystal_ball_data_1/RAW_DATASET/"+file_name.group(1)+"_raw_text.csv",index_label="index")
    
    return df


#test20= "crystal_ball_data_1/test20/"
#train = "crystal_ball_data_1/train/"

#for art in os.listdir(test20)[1:] :
    #merge_dataset(test20+art+"/violation/", train+art+"/violation/", verbose= True)
    #merge_dataset(test20+art+"/non-violation/", train+art+"/non-violation/", verbose= True)  
    #df = create_raw_dataset([train+art+"/violation/",train+art+"/non-violation/"], save=True, seed=7896)
    #print(df.head(10))

#df = create_raw_dataset([train+"Article10/violation/",train+"Article10/non-violation/"], save=True, seed=7896)
#print(df.head(10))



##code used for creating the big raw dataset for training the Doc2Vec model
'''
path = "ALL_together_crystal_ball/crystal_ball_data/train/"
df=pd.DataFrame(data= {'raw_text':[], 'tag':[]})

for art in os.listdir(path)[1:] :

    for x in ["/violation/","/non-violation/"] :

            for filename in os.listdir(path+art+x)[1:] :
                dic={}
                try:
                    with open(path+art+x+filename,"rt") as f :
                        dic['raw_text']  = apply_preprocessing(f.read())    
                except IndexError : print("Index Error at ",filename) 
                if "non-violation" in x :
                    dic['tag']=1 # "non-violation" 
                else :
                    dic['tag']=0 # "violation"
                df = df.append(dic,ignore_index=True) 

df.to_csv("all_art_raw_dataset.csv",index_label="index") 
'''


from ast import literal_eval
'''
#verifying the integrety of data
print("loading dataset...")
dataset = pd.read_csv("all_art_raw_dataset.csv", index_col="index")
dataset.raw_text = dataset.raw_text.apply(literal_eval)
print("verifying...")
for index, row in dataset.iterrows() : 
        if type(row['raw_text']) == type(1.0)  :
            print(row)
            dataset = dataset.drop([index])  
#dataset= dataset.reset_index(drop=True) 
#dataset.to_csv("all_art_raw_dataset.csv",index_label="index") 
#print("TOTAL error:",count)    
'''
'''
path= "crystal_ball_data_1/RAW_DATASET/"
count=0
for filename in os.listdir(path)[1:] :
    print(filename)   
    dataset = pd.read_csv(path+filename, index_col="index")
    dataset.raw_text = dataset.raw_text.apply(literal_eval)
    for index, row in dataset.iterrows() : 
        if type(row['raw_text']) == type(1.0)  :
            print(row)
            dataset = dataset.drop([index])          
    dataset= dataset.reset_index(drop=True) 
    dataset.to_csv(path+filename,index_label="index")        
print("TOTAL error:",count)
'''
