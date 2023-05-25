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

# Count Vectorizer Support Vector Machine
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=123)

# Vectorizing the text data
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
svcl = svm.SVC()
svcl.fit(ctmTr, y_train)
svcl_score = svcl.score(X_test_dtm, y_test)
print("Results for Support Vector Machine with CountVectorizer")
print(svcl_score)
y_pred_sv = svcl.predict(X_test_dtm)

# Confusion matrix
cm_sv = confusion_matrix(y_test, y_pred_sv)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sv).ravel()
print(tn, fp, fn, tp)
tpr_sv = round(tp/(tp + fn), 4)
tnr_sv = round(tn/(tn+fp), 4)
print(tpr_sv, tnr_sv)

# Precision score
svcl_prec = round(tp/(tp+fp), 4)
print(svcl_prec)

# Recall score
svcl_rec = round(tp/(tp+fn), 4)
print(svcl_rec)

# F1 score
svcl_f = round((2*svcl_prec*svcl_rec)/(svcl_prec+svcl_rec), 4)
print(svcl_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [svcl_prec, svcl_rec, svcl_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Support Vector Machine with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_sv, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Support Vector Machine with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
