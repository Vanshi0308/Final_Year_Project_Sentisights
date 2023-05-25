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

# Count Vectorizer Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=24)
cv = CountVectorizer()

# Vectorizing the text data
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
lr = LogisticRegression()
lr.fit(ctmTr, y_train)

# Accuracy score
lr_score = lr.score(X_test_dtm, y_test)
print("Results for Logistic Regression with CountVectorizer")
print(lr_score)

# Predicting the labels for test data
y_pred_lr = lr.predict(X_test_dtm)

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
print(tn, fp, fn, tp)

# True positive and true negative rates
tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)
print(tpr_lr, tnr_lr)

# Precision score
lr_prec = round(tp/(tp+fp), 4)
print(lr_prec)

# Recall score
lr_rec = round(tp/(tp+fn), 4)
print(lr_rec)

# F1 score
lr_f = round((2*lr_prec*lr_rec)/(lr_prec+lr_rec), 4)
print(lr_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [lr_prec, lr_rec, lr_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Logistic Regression with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_lr, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Logistic Regression with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

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

# Count Vectorizer K Nearest Neighbour
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=143)
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ctmTr, y_train)
knn_score = knn.score(X_test_dtm, y_test)
print("Results for KNN Classifier with CountVectorizer")
print(knn_score)
y_pred_knn = knn.predict(X_test_dtm)

# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
print(tn, fp, fn, tp)
tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn+fp), 4)
print(tpr_knn, tnr_knn)

# Precision score
knn_prec = round(tp/(tp+fp), 4)
print(knn_prec)

# Recall score
knn_rec = round(tp/(tp+fn), 4)
print(knn_rec)

# F1 score
knn_f = round((2*knn_prec*knn_rec)/(knn_prec+knn_rec), 4)
print(knn_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [knn_prec, knn_rec, knn_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('KNN Classifier with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_knn, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for KNN Classifier with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# TFIDF Vectorizer Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=45)
# tfidf vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
lr_score = lr.score(X_test_vec, y_test)
print("Results for Logistic Regression with tfidf")
print(lr_score)
y_pred_lr = lr.predict(X_test_vec)

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
print(tn, fp, fn, tp)
tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)
print(tpr_lr, tnr_lr)

# Precision score
lr_prec = round(tp/(tp+fp), 4)
print(lr_prec)

# Recall score
lr_rec = round(tp/(tp+fn), 4)
print(lr_rec)

# F1 score
lr_f = round((2*lr_prec*lr_rec)/(lr_prec+lr_rec), 4)
print(lr_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [lr_prec, lr_rec, lr_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Logistic Regression with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_lr, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Logistic Regression with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# TFIDF Vectorizer Support Vector Machine
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=55)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

#params = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
svcl = svm.SVC(kernel='rbf')

#clf_sv = GridSearchCV(svcl, params)
svcl.fit(X_train_vec, y_train)
svcl_score = svcl.score(X_test_vec, y_test)
print("Results for Support Vector Machine with tfidf")
print(svcl_score)
y_pred_sv = svcl.predict(X_test_vec)

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
ax.set_title('Support Vector Machine with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_sv, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Support Vector Machine with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# TFIDF Vectorizer K Nearest Neighbour
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=65)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)
knn_score = knn.score(X_test_vec, y_test)
print("Results for KNN Classifier with tfidf")
print(knn_score)
y_pred_knn = knn.predict(X_test_vec)

# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
print(tn, fp, fn, tp)
tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn+fp), 4)
print(tpr_knn, tnr_knn)

# Precision score
knn_prec = round(tp/(tp+fp), 4)
print(knn_prec)

# Recall score
knn_rec = round(tp/(tp+fn), 4)
print(knn_rec)

# F1 score
knn_f = round((2*knn_prec*knn_rec)/(knn_prec+knn_rec), 4)
print(knn_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [knn_prec, knn_rec, knn_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('KNN Classifier with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_knn, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for KNN Classifier with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Count Vectorizer Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
cv = CountVectorizer()

# Vectorizing the text data
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
rf = RandomForestClassifier()
rf.fit(ctmTr, y_train)

# Accuracy score
rf_score = rf.score(X_test_dtm, y_test)
print("Results for Random Forest Classifier with CountVectorizer")
print(rf_score)

# Predicting the labels for test data
y_pred_rf = rf.predict(X_test_dtm)

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
print(tn, fp, fn, tp)

# True positive and true negative rates
tpr_rf = round(tp/(tp + fn), 4)
tnr_rf = round(tn/(tn+fp), 4)
print(tpr_rf, tnr_rf)

# Precision score
rf_prec = round(tp/(tp+fp), 4)
print(rf_prec)

# Recall score
rf_rec = round(tp/(tp+fn), 4)
print(rf_rec)

# F1 score
rf_f = round((2*rf_prec*rf_rec)/(rf_prec+rf_rec), 4)
print(rf_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [rf_prec, rf_rec, rf_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Random Forest Classifier with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Random Forest Classifier with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# TFIDF Vectorizer Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()

# Vectorizing the text data
X_train_vec = vectorizer.fit_transform(
    X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
rf = RandomForestClassifier()
rf.fit(X_train_vec, y_train)

# Accuracy score
rf_score = rf.score(X_test_vec, y_test)
print("Results for Random Forest Classifier with tfidf")
print(rf_score)

# Predicting the labels for test data
y_pred_rf = rf.predict(X_test_vec)

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
print(tn, fp, fn, tp)

# True positive and true negative rates
tpr_rf = round(tp/(tp + fn), 4)
tnr_rf = round(tn/(tn+fp), 4)
print(tpr_rf, tnr_rf)

# Precision score
rf_prec = round(tp/(tp+fp), 4)
print(rf_prec)

# Recall score
rf_rec = round(tp/(tp+fn), 4)
print(rf_rec)

# F1 score
rf_f = round((2*rf_prec*rf_rec)/(rf_prec+rf_rec), 4)
print(rf_f)

# Precision, Recall, and F1 score bar plot
score_labels = ['Precision', 'Recall', 'F1 Score']
scores = [rf_prec, rf_rec, rf_f]

fig, ax = plt.subplots()
ax.bar(score_labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Random Forest Classifier with tfidf Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Random Forest Classifier with tfidf Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Count Vectorizer Decision Tree
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
cv = CountVectorizer()

# Vectorizing the text data
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
dt = DecisionTreeClassifier()
dt.fit(ctmTr, y_train)

# Accuracy score
dt_score = dt.score(X_test_dtm, y_test)
print("Results for Decision Tree Classifier with CountVectorizer")
print(dt_score)

# Predicting the labels for test data
y_pred_dt = dt.predict(X_test_dtm)

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
ax.set_title('Decision Tree Classifier with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_dt, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Decision Tree Classifier with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

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

# Count Vectorizer Multinomial Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
cv = CountVectorizer()

# Vectorizing the text data
ctmTr = cv.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_dtm = cv.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Training the model
mb = MultinomialNB()
mb.fit(ctmTr, y_train)

# Accuracy score
mb_score = mb.score(X_test_dtm, y_test)
print("Results for Multinomial Naive Bayes with CountVectorizer")
print(mb_score)

# Predicting the labels for test data
y_pred_mb = mb.predict(X_test_dtm)

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
ax.set_title('Multinomial Naive Bayes with Count Vectorizer')
ax.set_ylim([0, 1])
for i, v in enumerate(scores):
    ax.text(i-0.1, v+0.05, str(v), color='black', fontweight='bold')
plt.show()

# Heatmap of confusion matrix
sns.heatmap(cm_mb, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Multinomial Naive Bayes with Count Vectorizer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

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
