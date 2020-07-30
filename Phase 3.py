# 1 Imports
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import math
from sklearn.feature_extraction import text
import random
import numpy as np

import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# 2 - Load data collected from phase 1
data = load_files(os.getcwd() + "/Datasets")
X, y = data.data, data.target


# 3 - PRE-PROCESSING - removing unnecessary characters to clean up the data
documents = []

from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    document = re.sub('(?<=[A-Za-z])(?=[A-Z][a-z])', ' ', str(X[sen]))
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Removing prefixed 'b'
    document = re.sub("n ", "", document)
    
    # Converting to Lowercase
    document = document.lower()
    
    document = re.sub(r'(\s)x\w+', r'\1', document)
    
    # Lemmatization
    document = document.split()
    

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)


# 4 - Count vectorize and fitting the document to it
count_vect = CountVectorizer(stop_words=stopwords.words('english'))
X_train_counts = count_vect.fit_transform(documents)


# 5 - TDIDF to find the features
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts).toarray()

# 6 - splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2, random_state=0, shuffle = True)


# 7 GaussianNB Iplementation
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# 8 - Prediction 
y_pred = classifier.predict(X_test)

# 9 - Results
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# Test

# 10 - uploading test data
file_names = []
count = 0
path = os.getcwd() + "/Test"
for dirpath, dirnames, filenames in os.walk(path):  # getcwd() for current work dir
    file_names = filenames

file_content = []
for file_name in file_names:
    if ('.txt' in file_name):
        file_path = path+"/"+file_name
        file = open(file_path)
        txt = file.read()
        file.close()

        file_content.append(txt)


# 11 - Cleaning up test data
new_documents = []

from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

for sen in range(0, len(file_content)):
    document = re.sub('(?<=[A-Za-z])(?=[A-Z][a-z])', ' ', str(file_content[sen]))
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Removing prefixed 'b'
    document = re.sub("n ", "", document)
    
    # Converting to Lowercase
    document = document.lower()
    
    document = re.sub(r'(\s)x\w+', r'\1', document)
    
    # Lemmatization
    document = document.split()
    

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    new_documents.append(document)


# 12 Preparing to test
X_new_counts = count_vect.transform(new_documents)
X_new_tf = tf_transformer.transform(X_new_counts)

# 13 - Prediction
predicted = classifier.predict(X_new_tf.toarray())

predictions = []
for doc, category in zip(new_documents, predicted):
    predictions.append(data.target_names[category])


# Putting prediction, and expected result in DataFrame
file_names.remove('.DS_Store')
df = pd.DataFrame(file_names)
expected = ['Internet of Things', 'Blockchain', 'Blockchain', 'Artificial Intelligence', 'Internet of Things', 'Artificial Intelligence','Artificial Intelligence','Artificial Intelligence','Blockchain', 'Internet of Things','Blockchain', 'Internet of Things']
df['Prediction'] = predictions
df['Expected'] = expected
df.columns = ['File_names', 'Prediction', 'Expected']
print(df)
