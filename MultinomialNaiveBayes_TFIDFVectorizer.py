from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Getting rid of null values
df = pd.read_csv('D:\Final_Year_Project\Sentiment_Analysis/reviews.csv')

# Taking a 60% representative sample
df = df.dropna()
np.random.seed(34)

# Adding the sentiments column
df1 = df.sample(frac=0.6)
df1['sentiments'] = df1.score.apply(lambda x: 0 if x in [1, 2] else 1)

X = df1[['content', 'at']]
y = df1['sentiments']

# TFIDF Vectorizer Multinomial Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()

# Vectorizing the text data
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
mb = MultinomialNB()
mb.fit(X_train_vec, y_train)

# Accuracy score
mb_score = mb.score(X_test_vec, y_test)
print("Results for Multinomial Naive Bayes with tfidf")
print(mb_score)

# Predicting the labels for test data
y_pred_mb = mb.predict(X_test_vec)

# Confusion matrix
cm_mb = confusion_matrix(y_test, y_pred_mb)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_mb).ravel()
print(tn, fp, fn, tp)

# True positive and true negative rates
tpr_mb = round(tp/(tp + fn), 4)
tnr_mb = round(tn/(tn+fp), 4)
print(tpr_mb, tnr_mb)

# Precision score
mb_prec = round(tp/(tp+fp), 4)
print(mb_prec)

# Recall score
mb_rec = round(tp/(tp+fn), 4)
print(mb_rec)

# F1 score
mb_f = round((2*mb_prec*mb_rec)/(mb_prec+mb_rec), 4)
print(mb_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [mb_prec, mb_rec, mb_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Multinomial Naive Bayes with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_mb, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Multinomial Naive Bayes with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
