import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main content width */
    .block-container {
        max-width: 800px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Title */
    .app-title {
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    /* Subtitle */
    .app-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }

    /* Result box */
    .result-box {
        padding: 1.4rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .result-confidence {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Info text */
    .info-text {
        color: #9ca3af;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="app-title">🛡️ SpamShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Email & SMS Spam Detection using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📩 Enter your message</div>',
    unsafe_allow_html=True
)

input_sms = st.text_area(
    "Message",
    placeholder="Type or paste an SMS/email message here...",
    height=160,
    label_visibility="collapsed"
)

st.write("")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Analyze Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")

    else:

        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # Get prediction probability
        probabilities = model.predict_proba(vector_input)[0]

        if result == 1:

            confidence = probabilities[1] * 100

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">🚨 Spam Detected</div>
                    <div class="result-confidence">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(probabilities[1]))

        else:

            confidence = probabilities[0] * 100

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">✅ Not Spam</div>
                    <div class="result-confidence">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(probabilities[0]))


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.write("")
st.divider()

st.markdown(
    '<div class="section-title">⚙️ How it works</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-text">
    Your message is processed using <b>tokenization</b>,
    <b>stopword removal</b> and <b>stemming</b>.
    The processed text is then converted into numerical features
    using <b>TF-IDF</b> and classified using a
    <b>Multinomial Naive Bayes</b> model.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Python · NLP · TF-IDF · Naive Bayes · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)