
#10.06.'17
#jarvis
#udacity lesson
#source:http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[-0.8, -1]]))

#pred = clf.predict([-0.7,-0.9])
#print(accuracy_score(X,pred))