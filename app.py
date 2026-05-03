import streamlit as st
import numpy as np
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack


# LOAD MODEL + VOCAB + INDEX
with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\nb_model.pkl", "rb") as f:
    model = pickle.load(f)

fi1 = model["fi1"]
fi0 = model["fi0"]
fiy = model["fiy"]
vocab = model["vocab"]
word_index = model["word_index"]

stop = set(stopwords.words("english"))

# Precompute logs (faster prediction)
logfi1 = np.log(fi1)
logfi0 = np.log(fi0)
log_fiy1 = np.log(fiy)
log_fiy0 = np.log(1 - fiy)


# SAME FEATURE ENGINEERING AS PREPROCESSING


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def count_urls(text): return len(re.findall(r"http\S+|www.\S+", text))
def count_emails(text): return len(re.findall(r"\S+@\S+", text))
def count_numbers(text): return len(re.findall(r"\d+", text))
def count_uppercase(text): return sum(1 for c in text if c.isupper())
def count_exclamations(text): return text.count("!")
def avg_word_length(tokens): return np.mean([len(w) for w in tokens]) if len(tokens) > 0 else 0

spam_words = set([
    "free", "offer", "win", "cash", "money", "prize", "credit", "loan",
    "urgent", "winner", "click", "buy", "discount", "limited", "gift", "deal"
])

def get_features(email_text, tokens):
    return [
        count_urls(email_text),
        count_emails(email_text),
        count_numbers(email_text),
        count_uppercase(email_text),
        count_exclamations(email_text),
        len(tokens),
        avg_word_length(tokens),
        int(bool(re.search(r"<[^>]+>", email_text))),  # has_html
        0,  # sender_domain — unknown in free text input
        sum(1 for w in tokens if w in spam_words)
    ]

# PREDICTION FUNCTION (SPARSE-SAFE)

def predict_sparse(row):
    indices = row.indices
    logp1 = log_fiy1 + logfi1[indices].sum()
    logp0 = log_fiy0 + logfi0[indices].sum()
    return 1 if logp1 >= logp0 else 0


# STREAMLIT UI

st.title("📧 Email Spam Detector (NLP + Naive Bayes Model)")
st.write("Enter an email and the model will classify it as **Spam** or **Not Spam**.")

email = st.text_area("Paste the email text here:", height=250)

if st.button("Analyze Email"):
    if email.strip() == "":
        st.warning("Please enter email text!")
    else:
        # Clean + tokenize
        clean = clean_text(email)
        tokens = [w for w in word_tokenize(clean) if w not in stop and len(w) > 1]

        # ==== 1. Feature Engineering ====
        engineered = np.array(get_features(email, tokens)).reshape(1, -1)
        engineered_sparse = csr_matrix(engineered)

        # ==== 2. Bag of Words ====
        bow_data = []
        bow_row = []
        bow_col = []

        for w in tokens:
            if w in word_index:
                bow_row.append(0)
                bow_col.append(word_index[w])
                bow_data.append(1)

        bow_sparse = csr_matrix(
            (bow_data, (bow_row, bow_col)),
            shape=(1, len(vocab)),
            dtype=np.uint8
        )

        # ==== 3. Final Sparse Vector ====
        x = hstack([engineered_sparse, bow_sparse], format="csr")

        # ==== 4. Predict ====
        result = predict_sparse(x[0])

        # ==== 5. Output ====
        if result == 1:
            st.error("🚨 This email is likely **SPAM**")
        else:
            st.success("✔ This email is **Not Spam**")

        # Extra Information
        st.write("---")
        st.write("### Extracted Tokens:")
        st.write(tokens)

        st.write("### Feature Values:")
        feature_names = [
            "URL count", "Email count", "Number count", "Uppercase count",
            "Exclamation count", "Word count", "Avg word length",
            "HTML detected", "Sender domain (N/A)", "Spam keyword count"
        ]
        for name, val in zip(feature_names, engineered[0]):
            st.write(f"**{name}:** {val}")
