#1
import requests, webbrowser
from bs4 import BeautifulSoup
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import math
from sklearn.feature_extraction import text
import random

#2
folder_names = []
all_files = []
count = 0
path = os.getcwd() + "/Datasets"
for dirpath, dirnames, filenames in os.walk(path):  # getcwd() for current work dir
    if count == 0:
        folder_names = dirnames
    else:
        all_files.append(filenames)
    count+=1

#3
text_seperate = []
text_all = []

for i in range(len(all_files)):
    text_folder = []
    text_documents = ""
    for file_name in all_files[i]:
        txt = Path(path+"/"+folder_names[i]+"/"+file_name).read_text()
        txt = txt.replace("\n", " ")
        txt = txt.replace("\t", " ")
        text_documents += txt
        text_folder.append(txt)
    text_all.append(text_documents)
    text_seperate.append(text_folder)

#4
new_text_all = []
for i in range(len(text_all)):
    txt = text_all[i].lower().split()
    new_text = ""
    for word in txt:
        occurence = txt.count(word)
        doc_with_word = 0
        for doc in text_seperate[i]:
            if word in doc:
                doc_with_word += 1
        
        if occurence >= 10 and doc_with_word >=2:
            new_text += word + " "
    new_text_all.append(new_text)

#5
vectorizer_1 = TfidfVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
vectors = vectorizer_1.fit_transform([new_text_all[0]])
feature_names_1 = vectorizer_1.get_feature_names()

#6
vectorizer_2 = TfidfVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
vectors = vectorizer_2.fit_transform([new_text_all[1]])
feature_names_2 = vectorizer_2.get_feature_names()

#7
vectorizer_3 = TfidfVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
vectors = vectorizer_3.fit_transform([new_text_all[2]])
feature_names_3 = vectorizer_3.get_feature_names()

#8
feature_names = []
feature_names.append(feature_names_1)
feature_names.append(feature_names_2)
feature_names.append(feature_names_3)

#9
features_all = []

for i in range(len(feature_names)):
    feature_topic = []
    for feature in feature_names[i]:
        counter = False
        if not feature.isdigit():
            for char in feature:
                if char.isdigit():
                    counter = True
                    break
            if not counter:
                feature_topic.append(feature)
    features_all.append(feature_topic)

#10
selected_feature_all = []
selected_num_all = []

for i in range(len(features_all)):
    
    count = 0
    feature_list = features_all[i]
    max_num = len(feature_list)-1
    
    selected_feature = []
    selected_num = []
    
    while count < 15:
        num = random.randint(0,max_num)
        if num not in selected_num:
            selected_feature.append(feature_list[num])
            selected_num.append(num)
            count += 1
    selected_feature_all.append(selected_feature)
    selected_num_all.append(selected_num)

#11
tfidf_all_1 = []
occurences_all_1 = []
for data in text_seperate[0]:
    doc_tfidf = []
    doc_occurences = []
    for feature in selected_feature_all[0]:
        occurences = data.count(feature)
        doc_occurences.append(occurences)
        num_words = len(data.split())
        tf = occurences/num_words
        doc_with_feature = 0
        idf = 0
        for doc in text_seperate[0]:
            if feature in doc:
                doc_with_feature += 1
        if doc_with_feature != 0:
            idf = math.log((10/doc_with_feature),2)
        tfidf = tf*idf
        doc_tfidf.append(tfidf)
    tfidf_all_1.append(doc_tfidf)
    occurences_all_1.append(doc_occurences)

#12
df_1 = pd.DataFrame(tfidf_all_1)
df_1.columns = selected_feature_all[0]
print(df_1)

#13
tfidf_all_2 = []
occurences_all_2 = []
for data in text_seperate[1]:
    doc_tfidf = []
    doc_occurences = []
    for feature in selected_feature_all[1]:
        occurences = data.count(feature)
        doc_occurences.append(occurences)
        num_words = len(data.split())
        tf = occurences/num_words
        doc_with_feature = 0
        idf = 0
        
        for doc in text_seperate[1]:
            if feature in doc:
                doc_with_feature += 1
        if doc_with_feature != 0:
            idf = math.log((10/doc_with_feature),2)
        tfidf = tf*idf
        doc_tfidf.append(tfidf)
    tfidf_all_2.append(doc_tfidf)
    occurences_all_2.append(doc_occurences)

#14
df_2 = pd.DataFrame(tfidf_all_2)
df_2.columns = selected_feature_all[1]
print(df_2)

#15
tfidf_all_3 = []
occurences_all_3 = []
for data in text_seperate[2]:
    doc_tfidf = []
    doc_occurences = []
    for feature in selected_feature_all[2]:
        occurences = data.count(feature)
        doc_occurences.append(occurences)
        num_words = len(data.split())
        tf = occurences/num_words
        doc_with_feature = 0
        idf = 0
        
        for doc in text_seperate[2]:
            if feature in doc:
                doc_with_feature += 1
        if doc_with_feature != 0:
            idf = math.log((10/doc_with_feature),2)
        tfidf = tf*idf
        doc_tfidf.append(tfidf)
    tfidf_all_3.append(doc_tfidf)
    occurences_all_3.append(doc_occurences)

#16
df_3 = pd.DataFrame(tfidf_all_3)
df_3.columns = selected_feature_all[2]
print(df_3)




