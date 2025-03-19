from django.shortcuts import render
# Create your views here.
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Load Dataset
df = pd.read_csv("dataset/plagiarism_dataset.csv")  # Make sure this file is in dataset folder

# Data Preprocessing
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_column']  # Replace with actual column name

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Django API View
@csrf_exempt
def check_plagiarism(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_text = preprocess_text(data.get('text', ''))
        vectorized_text = vectorizer.transform([user_text])
        prediction = model.predict(vectorized_text)[0]
        result = "Plagiarized" if prediction == 1 else "Not Plagiarized"
        return JsonResponse({"result": result})
