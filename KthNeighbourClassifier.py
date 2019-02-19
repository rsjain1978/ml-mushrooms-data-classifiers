#Question 2
#Consider the one-dimensional data set shown below.
#Classify the data point x = 5.0 according to its 1st, 3rd, 5th, and 9th nearest neighbours using
#K-nearest neighbour classifier.

#Answer:
#Predicted value for Y for X=5 at its 1 neighbour is ['+']
#Predicted value for Y for X=5 at its 3 neighbour is ['-']
#Predicted value for Y for X=5 at its 5 neighbour is ['+']
#Predicted value for Y for X=5 at its 9 neighbour is ['-']


import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

def printValueAtKthNearestNeighbour(X,Y,kthNeighbour,testY):
    #initialize KNNClassifier
    neighbour = KNeighborsClassifier(n_neighbors=kthNeighbour)
    model = neighbour.fit(X, Y)
    print ('Predicted value for Y for X=%s at its %s neighbour is %s'%(5,kthNeighbour, neighbour.predict(testY)))


#prepare data set
X=[0.5, 3.0, 4.5, 4.6, 4.9, 5.2, 5.3, 5.5, 7.0, 9.5]
Y=['-','-','+','+','+','-','-','+','-','-']
dfX = pd.DataFrame(X)

#initialize test data
testY = pd.DataFrame([5])

printValueAtKthNearestNeighbour(dfX,Y,1, testY)
printValueAtKthNearestNeighbour(dfX,Y,3, testY)
printValueAtKthNearestNeighbour(dfX,Y,5, testY)
printValueAtKthNearestNeighbour(dfX,Y,9, testY)
