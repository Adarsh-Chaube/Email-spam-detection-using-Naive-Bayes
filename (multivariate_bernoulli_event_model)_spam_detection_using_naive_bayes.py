import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


# LOAD SPARSE MATRIX + LABELS
X = load_npz(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\final_X_sparse.npz")

with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\labels_y.pkl", "rb") as f:
    y = pickle.load(f)

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)


# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# BINARIZE SPARSE INPUT (FAST)
x_train_bin = x_train.copy()
x_train_bin.data[:] = 1

x_test_bin = x_test.copy()
x_test_bin.data[:] = 1


# PRIOR
fiy = np.mean(y_train)


# CONDITIONAL PROBABILITIES
sum1 = x_train_bin[y_train == 1].sum(axis=0)
sum0 = x_train_bin[y_train == 0].sum(axis=0)

# Convert matrix results to array
sum1 = np.asarray(sum1).ravel()
sum0 = np.asarray(sum0).ravel()

# Laplace smoothing
fi1 = (sum1 + 1) / (sum1.sum() + x_train_bin.shape[1])
fi0 = (sum0 + 1) / (sum0.sum() + x_train_bin.shape[1])

logfi1 = np.log(fi1)
logfi0 = np.log(fi0)

log_fiy1 = np.log(fiy)
log_fiy0 = np.log(1 - fiy)


# PREDICTION
def predict_sparse(row):
    indices = row.indices
    logp1 = log_fiy1 + logfi1[indices].sum()
    logp0 = log_fiy0 + logfi0[indices].sum()
    return 1 if logp1 >= logp0 else 0

pred = np.array([predict_sparse(r) for r in x_test_bin])


# ACCURACY and EVALUATION METRICS
accuracy = (pred == y_test).mean() * 100

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# PRECISION
precision = precision_score(y_test, pred)

# RECALL
recall = recall_score(y_test, pred)

# F1 SCORE
f1 = f1_score(y_test, pred)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, pred)

# CLASSIFICATION REPORT
report = classification_report(y_test, pred, target_names=["Ham", "Spam"])

print("\n--- Model Evaluation Metrics ---")
print("Accuracy:", accuracy, "%")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# LOAD VOCAB + WORD_INDEX
with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\word_index.pkl", "rb") as f:
    word_index = pickle.load(f)


# SAVE NB MODEL PARAMETERS
nb_model = {
    "fi1": fi1,
    "fi0": fi0,
    "fiy": fiy,
    "vocab": vocab,
    "word_index": word_index
}

with open(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

print("Model saved as nb_model.pkl")