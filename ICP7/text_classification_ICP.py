from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score_m = round(metrics.accuracy_score(twenty_test.target, predicted), 2)
print("\nMultinomial DB accuracy is")
print(score_m)

# KNN
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train_tfidf, twenty_train.target)

predicted_knn = clf_knn.predict(X_test_tfidf)

score_knn = round(metrics.accuracy_score(twenty_test.target, predicted_knn), 2)
print("\nKNN accuracy is")
print(score_knn)

# Apply bigrams
tfidf_Vect_b = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf_b = tfidf_Vect_b.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf_b = MultinomialNB()
clf_b.fit(X_train_tfidf_b, twenty_train.target)

twenty_test_b = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf_b = tfidf_Vect_b.transform(twenty_test_b.data)

predicted_b = clf_b.predict(X_test_tfidf_b)

score_m_b = round(metrics.accuracy_score(twenty_test_b.target, predicted_b), 2)
print("\nMultinomial DB accuracy after applying bigram is")
print(score_m_b)

# KNN
clf_knn_b = KNeighborsClassifier(n_neighbors=5)
clf_knn_b.fit(X_train_tfidf_b, twenty_train.target)

predicted_knn_b = clf_knn_b.predict(X_test_tfidf_b)

score_knn_b = round(metrics.accuracy_score(twenty_test_b.target, predicted_knn_b), 2)
print("\nKNN accuracy after applying bigrams is")
print(score_knn_b)

# Apply stop words
tfidf_Vect_s = TfidfVectorizer(stop_words='english')
X_train_tfidf_s = tfidf_Vect_s.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf_s = MultinomialNB()
clf_s.fit(X_train_tfidf_s, twenty_train.target)

twenty_test_s = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf_s = tfidf_Vect_s.transform(twenty_test_s.data)

predicted_s = clf_s.predict(X_test_tfidf_s)

score_m_s = round(metrics.accuracy_score(twenty_test_s.target, predicted_s), 2)
print("\nMultinomial DB accuracy after applying stop words English is")
print(score_m_s)

# KNN
clf_knn_s = KNeighborsClassifier(n_neighbors=5)
clf_knn_s.fit(X_train_tfidf_s, twenty_train.target)

predicted_knn_s = clf_knn_s.predict(X_test_tfidf_s)

score_knn_s = round(metrics.accuracy_score(twenty_test_s.target, predicted_knn_s), 2)
print("\nKNN accuracy after applying stop words English is")
print(score_knn_s)