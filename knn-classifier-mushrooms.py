import pandas as pd
import numpy as np

import performancemetrics as pcp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

def classifyMushroomData(mode):
    #Read the data set
    data = pd.read_csv('mushrooms.csv')

    #check if there are null values present
    nullValuesPresent = pd.isnull(data).values.any()
    if (nullValuesPresent) :
        print ("There are null values present so need to handle them")
    else :
        print ("No null values found, we are good to go")

    #Split values of X and Y
    Y = data['class']
    X = data.drop(['class'],axis=1)

    #Convert X to dummy representation
    X=pd.get_dummies(X, drop_first=True)

    # Spli the data using K-Fold Model Selection approach. We start with 2Fold and then test till 10Fold
    mapKFoldAndModelScore={}
    kFolds=2
    mode = int (mode)

    if (mode!=2):
        kFolds =2
        limit =10
    else  :
        kFolds = 5
        limit = 6

    while (kFolds<limit):
        kf = KFold(n_splits=kFolds, shuffle=True, random_state=10)

        sizeOfFeatures = X.shape[1]
        neighbours = np.round(np.sqrt(sizeOfFeatures))
        neighbours = int(neighbours)
        print ('\nNumber of Neighbours to be used in the classifier are ', neighbours)

        classifier = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform')

        runningModelScore = 0
        for train_index, test_index in kf.split(X):

            train_x = X.iloc[train_index]
            test_x = X.iloc[test_index]
            train_y = Y.iloc[train_index]
            test_y = Y.iloc[test_index]

            #see if the classifier is doing an overfitting of the training data
            model = classifier.fit(train_x, train_y.values.ravel()) 
            modelScore = model.score(test_x, test_y)
            predicted_y_with_test_data = predicted_y=classifier.predict(test_x)

            if (mode==2 ) :
                pcp.printKNNClassifierPerformance(classifier, modelScore, test_x, test_y, predicted_y_with_test_data)

            runningModelScore = runningModelScore + modelScore

        averageScore = runningModelScore/kFolds

        if (mode==2 ) :
            print ('\n\nAverage Model Score is %s for %s-fold'%(averageScore, kFolds))
        else :
            print ('Average Model Score is %s for %s-fold'%(averageScore, kFolds))
        
        mapKFoldAndModelScore[kFolds]=averageScore
        kFolds=kFolds+1
        runningModelScore = 0
    
    print ('\nMapping of each k and the corresponding model accuracy score in (kFold, score) format')
    print (mapKFoldAndModelScore)

#Display execution options to the user and accept the mode in which the classifier needs to run.
print ('Please press 1 if you want to find best value of k in cross validation approach')
print ('Please press 2 if you want to see the performance measures of the classifier with k=5')

mode = input("Please enter your choice: ")
classifyMushroomData(mode)