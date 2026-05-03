import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

with open(r'C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\vocab.pkl','rb') as f:
    vocab=list(pickle.load(f))

'''Word to index mapping'''
word_index={word:idx for idx,word in enumerate(vocab)}
print(len(word_index))

'''Load data'''
df=pd.read_csv(r"C:\Users\lenovo\Desktop\Projects\ML\Naive_Bayes\Email_spam_balanced_dataset.csv")
text=df['Body']
labels=df["Label"].values

'''cleaning and vectorizing text'''
text=[re.sub(r"[^a-zA-Z\s]",'',txt.lower()) for txt in text]
text=[word_tokenize(txt) for txt in text]
vector=[]
for i in range(0,len(text)):
    v1=[]
    for word in text[i]:
        if word in word_index:
            v1.append(word_index[word])
    vector.append(v1)
print(len(vector))

'''train test split'''
x_train, x_test, y_train, y_test = train_test_split(vector, labels, test_size=0.2, random_state=42)

'''Initializing Parameters'''
c0=np.zeros(len(vocab))
c1=np.zeros(len(vocab))
for j in range(len(x_train)):
    for i in x_train[j]:
        if y_train[j]==0 :
            c0[i]+=1;
        else:
            c1[i]+=1;
fiy=np.mean(y_train)
fi1=(c1+1)/(np.sum(c1)+len(vocab))
fi0=(c0+1)/(np.sum(c0)+len(vocab))

'''Convert x_test into bag of words format'''
def convert(x):
    f=np.zeros((len(x),len(vocab)))
    for i in range(len(x)):
        for j in x[i]:
            f[i][j]+=1
    return f

'''Predicting'''

def predict(x,fiy,fi1,fi0):
    logfi1=np.log(fi1)
    logfi0=np.log(fi0)
    logfiy1=np.log(fiy)
    logfiy0=np.log(1-fiy)
    p1=(x@logfi1)+logfiy1
    p2=(x@logfi0)+logfiy0
    y_pred=(p1>p2).astype(int)
    return y_pred

x_test=convert(x_test)
yp=predict(x_test,fiy,fi1,fi0)
acc=(yp==y_test).mean()*100
print("Accuracy :  ",acc)
print(confusion_matrix(y_test, yp))
print(classification_report(y_test, yp))