import pandas as pd

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from IPython.display import Image
import pydotplus

wine = datasets.load_wine()
#print (wine.shape)
X= wine.data
Y = wine.target

trainX, testX, trainY, testY = train_test_split(X,Y)

classifier = tree.DecisionTreeClassifier()
treeFit = classifier.fit(trainX, trainY)
predictedY = treeFit.predict(testX)
print (confusion_matrix(testY, predictedY))
print (accuracy_score(testY, predictedY))
print (classifier.tree_)

print ('---------------')
classifier = tree.DecisionTreeClassifier(criterion='entropy')
treeFit = classifier.fit(trainX, trainY)
predictedY = treeFit.predict(testX)
print (confusion_matrix(testY, predictedY))
print (accuracy_score(testY, predictedY))
print (classifier.tree_)

# Create DOT data. The first argument in tree.export_graphviz is the model name, out_file is used to write 
# model into out_file, next parameters are information on indicator and predictive parameters 

dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=wine.feature_names, class_names=wine.target_names)

# Draw graph

graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
