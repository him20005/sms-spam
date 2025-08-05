import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Dataset load
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
df['text'] = df['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'].map({'ham':0, 'spam':1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer together
joblib.dump((vectorizer, model), 'model.pkl')
print("Model and vectorizer saved as model.pkl")
