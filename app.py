import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

#st.title("EMail/SMS SPAM CLASSIFIER")
#input_sms = st.text_area("Enter the Message", height=200)

st.markdown(
    f"""
    <style>
    body {{
        background-image: url('D:\machine learning projects\email sms spam classifier\background.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.10.5/font/bootstrap-icons.min.css">
    <h1 style="color: #3498db; font-weight: bold; text-align: center; font-size: 36px;">
        <i class="bi bi-envelope-fill" style="vertical-align: middle; font-size: 36px; margin-right: 10px;"></i>
        EMail/SMS SPAM DETECTOR

    <h1 style="color: #008000; font-weight: normal; text-align: center; font-size: 15px;">
    Protect Your Inbox from Spam
    
    </h1>
    """,
    unsafe_allow_html=True
)

if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ""

# Create a text area with the current text in session state
st.session_state.input_sms = st.text_area("Enter your text here:", key="textarea", placeholder="Type here...",
                                           value=st.session_state.input_sms, height=200)

st.markdown(
    """
    <style>
    textarea {
        background-color: #2b2b2b !important;
        color: white !important;
        border-radius: 10px !important;
        border: 2px solid #4CAF50 !important;
        font-size: 16px !important;
        padding: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a "Clear Text" button
col1, col2, col3, col4 = st.columns(4)

with col3:
    st.write("Double click")
    if st.button("Clear Text"):
        st.session_state.input_sms = ""  # Clear the text

# Display the current text (optional)
#st.write("Current text:", st.session_state.input_sms)

with col4:
    st.write(f"You wrote {len(st.session_state.input_sms)} characters.")

# 1. Preprocess the text

def transformed_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("English") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

transformed_sms = transformed_text(st.session_state.input_sms)

# 2. Vectorize the text

vectorize_input = tfidf.transform([transformed_sms])

# 3. Predict

result = model.predict(vectorize_input)[0]

# 4. Check the output
with col2:
    st.write("Check result")
    if st.button("Predict"):
        if result == 1:
            st.subheader(":red[SPAM]")

        else:
            st.subheader(":green[NOT SPAM]")
    