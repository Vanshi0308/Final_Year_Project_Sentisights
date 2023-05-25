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

# TFIDF Vectorizer Decision Tree
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()

# Vectorizing the text data
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
dt = DecisionTreeClassifier()
dt.fit(X_train_vec, y_train)

# Accuracy score
dt_score = dt.score(X_test_vec, y_test)
print("Results for Decision Tree Classifier with tfidf")
print(dt_score)

# Predicting the labels for test data
y_pred_dt = dt.predict(X_test_vec)

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
print(tn, fp, fn, tp)

# True positive and true negative rates
tpr_dt = round(tp/(tp + fn), 4)
tnr_dt = round(tn/(tn+fp), 4)
print(tpr_dt, tnr_dt)

# Precision score
dt_prec = round(tp/(tp+fp), 4)
print(dt_prec)

# Recall score
dt_rec = round(tp/(tp+fn), 4)
print(dt_rec)

# F1 score
dt_f = round((2*dt_prec*dt_rec)/(dt_prec+dt_rec), 4)
print(dt_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [dt_prec, dt_rec, dt_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Decision Tree Classifier with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_dt, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Decision Tree Classifier with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
