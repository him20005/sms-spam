import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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

# Prediction/Evaluation
y_pred = model.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# Predict your own message
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vect = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vect)[0]
    return 'spam' if prediction == 1 else 'ham'

print(predict_message("Congratulations! You've won a prize!"))
print(predict_message("Hey, are you coming to the party?"))
print(predict_message("Congratulations! You've won a prize!"))
print(predict_message("WELCOME TO PIET"))
print(predict_message("Congratulations! You've won a prize!"))
import joblib

# model और vectorizer को tuple/list में pack करके सेव करें
joblib.dump((vectorizer, model), 'model.pkl')
print("Model and vectorizer saved as model.pkl")
msg_clean = clean_text
msg_vect = vectorizer.transform([msg_clean])
prediction = model.predict(msg_vect)
