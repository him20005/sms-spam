import streamlit as st
import joblib
import string
import nltk

# ----- Clean Text Function -----
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ----- Load Model -----
vectorizer, model = joblib.load('model.pkl')

# ----- Streamlit App -----
st.title("SMS Spam Detection Web App")

sms = st.text_area("यहाँ अपना SMS लिखें:")

if st.button("Predict"):
    sms_clean = clean_text(sms)
    sms_vect = vectorizer.transform([sms_clean])
    prediction = model.predict(sms_vect)
    if prediction[0] == 1:
        st.error("यह SMS SPAM है 🚫")
    else:
        st.success("यह SMS SPAM नहीं है ✅")
