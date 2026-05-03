# ============================
# IMPORTS
# ============================
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
from scipy.sparse import csr_matrix, hstack, save_npz  


# READ DATA
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\dataset\Email_spam_balanced_dataset.csv")

stop = set(stopwords.words('english'))


# CLEAN TEXT FUNCTION

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " url ", text)     
    text = re.sub(r"\S+@\S+", " emailaddr ", text)       
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             
    text = re.sub(r"\s+", " ", text).strip()             
    return text


df["clean_body"] = df["Body"].apply(clean_text)

df["tokens"] = df["clean_body"].apply(
    lambda x: [w for w in word_tokenize(x) if w not in stop and len(w) > 1]
)



# 2. CREATE VOCABULARY (TOP 10,000 WORDS)
text_all = " ".join(df["clean_body"])
tokens_all = word_tokenize(text_all)
tokens_all = [w for w in tokens_all if w not in stop and 1 < len(w) < 15]

freq = Counter(tokens_all)
vocab = [word for word, count in freq.most_common(10000)]

print("Vocabulary size:", len(vocab))

# Save vocab
with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)



# WORD → INDEX MAPPING

word_index = {word: idx for idx, word in enumerate(vocab)}



# FEATURE ENGINEERING FUNCTIONS
def count_urls(text): return len(re.findall(r"http\S+|www.\S+", text))
def count_emails(text): return len(re.findall(r"\S+@\S+", text))
def count_numbers(text): return len(re.findall(r"\d+", text))
def count_uppercase(text): return sum(1 for c in text if c.isupper())
def count_exclamations(text): return text.count("!")
def avg_word_length(tokens): return np.mean([len(w) for w in tokens]) if len(tokens) > 0 else 0



# APPLY FEATURE ENGINEERING

df["url_count"] = df["Body"].apply(count_urls)
df["email_count"] = df["Body"].apply(count_emails)
df["number_count"] = df["Body"].apply(count_numbers)
df["uppercase_count"] = df["Body"].apply(count_uppercase)
df["exclamation_count"] = df["Body"].apply(count_exclamations)
df["word_count"] = df["tokens"].apply(len)
df["avg_word_length"] = df["tokens"].apply(avg_word_length)
df["has_html"] = df["Body"].apply(lambda x: int(bool(re.search(r"<[^>]+>", x))))

def get_domain(body):
    match = re.search(r"From:\s.*@([A-Za-z0-9.-]+)", body)
    return match.group(1).lower() if match else "unknown"

df["sender_domain"] = df["Body"].apply(get_domain)
df["sender_domain_encoded"] = df["sender_domain"].astype("category").cat.codes

spam_words = set([
    "free", "offer", "win", "cash", "money", "prize", "credit", "loan",
    "urgent", "winner", "click", "buy", "discount", "limited", "gift", "deal"
])

df["spam_keyword_count"] = df["tokens"].apply(lambda t: sum(1 for w in t if w in spam_words))



# engineered feature matrix

feature_cols = [
    "url_count", "email_count", "number_count", "uppercase_count",
    "exclamation_count", "word_count", "avg_word_length", "has_html",
    "sender_domain_encoded", "spam_keyword_count"
]
engineered_features = df[feature_cols].astype(np.int16).values



#  CREATE SPARSE BAG-OF-WORDS MATRIX
rows = len(df)
cols = len(vocab)

# sparse CSR matrix builder lists
data = []
row_idx = []
col_idx = []

for i in range(rows):
    for word in df["tokens"][i]:
        if word in word_index:
            row_idx.append(i)
            col_idx.append(word_index[word])
            data.append(1)

bow_sparse = csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.uint8)



# FINAL SPARSE MATRIX + LABELS

engineered_sparse = csr_matrix(engineered_features)

final_X_sparse = hstack([engineered_sparse, bow_sparse], format="csr")

y = df["Label"].values

print("Final shape:", final_X_sparse.shape)
print("Memory used:", final_X_sparse.data.nbytes / (1024**2), "MB")



# 8. SAVE EVERYTHING
save_npz("final_X_sparse.npz", final_X_sparse)

with open("labels_y.pkl", "wb") as f:
    pickle.dump(y, f)

with open("word_index.pkl", "wb") as f:
    pickle.dump(word_index, f)

print("All preprocessing assets saved successfully!")
