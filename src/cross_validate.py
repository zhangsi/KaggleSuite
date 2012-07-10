'''
Created on Jul 9, 2012

@author: chris
'''
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn import cross_validation
import logloss
import numpy as np
from timeit import Timer

def main():
    # read in the data, parse into training/test sets
    dataset = np.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])

    # In this case we'll use a random forest, but this could be any classifier
    #cfr = RandomForestClassifier(n_estimators=100, n_jobs = 1)
    #cfr = sklearn.ensemble.GradientBoostingClassifier()
    cfr = SVC(probability = True)
    

    # k-fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), k = 5, indices = False)

    # iterate through the training and test cross validation segments and
    # run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append(logloss.llfun(target[testcv], [x[1] for x in probas]))
        #print results

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    t = Timer("main()", setup = "from __main__ import main")
    #main()
    print t.timeit(number = 1)