from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
import pymysql
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np

# Ensure NLTK data is downloaded (only needed once)
# nltk.download('stopwords')
# nltk.download('wordnet')

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

filename = []
word_vector = []

# Text preprocessing
def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"[!(),?]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return cleanPost(string.strip().lower())

# Load dataset
with open('dataset/question_Data.json', "r") as f:
    lines = f.readlines()
    for line in lines:
        arr = line.strip().split("#")
        if len(arr) > 1:
            cleanedLine = clean_str(arr[0]).strip()
            if cleanedLine:  # Skip if cleaned line is empty
                word_vector.append(cleanedLine)
                filename.append(arr[1])
        else:
            print("Skipped invalid line:", line.strip())

if not word_vector:
    raise ValueError("No valid data found. Check your dataset.")

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf = tfidf_vectorizer.fit_transform(word_vector).toarray()
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names_out())
X = df.to_numpy()
filename = np.asarray(filename)
word_vector = np.asarray(word_vector)

# Django views
def MyChatBot(request):
    return render(request, 'index.html', {})

def User(request):
    return render(request, 'User.html', {})

def Logout(request):
    return render(request, 'index.html', {})

def test(request):
    return render(request, 'test.html', {})

def Register(request):
    return render(request, 'Register.html', {})

def ChatData(request):
    if request.method == 'GET':
        question = request.GET.get('mytext', '')
        cleanedLine = clean_str(question).strip()
        if not cleanedLine:
            return HttpResponse("Please enter a valid question.", content_type="text/plain")

        testVector = tfidf_vectorizer.transform([cleanedLine]).toarray()[0]

        similarity = 0
        response = 'Sorry! I am not trained for the given question'

        for i in range(len(X)):
            sim = dot(X[i], testVector) / (norm(X[i]) * norm(testVector) + 1e-10)
            if sim > similarity and sim > 0.50:
                similarity = sim
                response = filename[i]

        return HttpResponse(response, content_type="text/plain")

def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        success = False

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='chatbot', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    success = True
                    break

        if success:
            return render(request, 'UserScreen.html', {'data': f'Welcome {username}'})
        else:
            return render(request, 'User.html', {'data': 'Login failed'})

def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        address = request.POST.get('address')

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='chatbot', charset='utf8')
        with con:
            cur = con.cursor()
            try:
                query = "INSERT INTO register (username, password, contact, email, address) VALUES (%s, %s, %s, %s, %s)"
                cur.execute(query, (username, password, contact, email, address))
                con.commit()
                return render(request, 'Register.html', {'data': 'Signup Process Completed'})
            except Exception as e:
                return render(request, 'Register.html', {'data': f'Error in signup process: {str(e)}'})
