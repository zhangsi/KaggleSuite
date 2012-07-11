'''
Created on Jul 9, 2012

@author: chris
'''
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
import logloss, visualize_data
import numpy as np
from timeit import Timer
import operator
from sklearn.metrics import roc_curve, auc

def main():
    # read in the data, parse into training/test sets
    #dataset = np.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    dataset = np.genfromtxt(open('../Data/credit_data/cs-training.csv','r'), delimiter=',', dtype='f8')[1:, 1:]
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])

    # In this case we'll use a random forest, but this could be any classifier
    classifiers = []
    classifiers.append(RandomForestClassifier(n_estimators = 100, n_jobs = -1))
    classifiers.append(GradientBoostingClassifier())
    classifiers.append(SVC(probability = True))
    

    # k-fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), k = 5, indices = False)
    
    # for all classifiers, use k-fold cv to evaluate performance and select the best one.
    avg_log_loss = [0] * len(classifiers)
    for i in range(len(classifiers)):
        for traincv, testcv in cv:
            probas = classifiers[i].fit(train[traincv], target[traincv]).predict_proba(train[testcv])
            
            fpr, tpr, thresholds = roc_curve(target[testcv], probas[:, 1])
            roc_auc = auc(fpr, tpr)
            print "Area under the ROC curve : %f" % roc_auc
            visualize_data.drawROCCurve(fpr, tpr, roc_auc)
            avg_log_loss[i] += roc_auc
            #avg_log_loss[i] += (logloss.llfun(target[testcv], [x[1] for x in probas]))

    print avg_log_loss
    for i in range(len(avg_log_loss)):
        avg_log_loss[i] = avg_log_loss[i] / len(cv)
        
    # switching this between min and max, check which is which!
    min_index, min_value = max(enumerate(avg_log_loss), key = operator.itemgetter(1))
    print "Selected classifier: ", classifiers[min_index]
    #print out the mean of the cross-validated results
    #print "Results: " + str( np.array(avg_log_loss).mean() )

if __name__=="__main__":
    t = Timer("main()", setup = "from __main__ import main")
    #main()
    print t.timeit(number = 1)