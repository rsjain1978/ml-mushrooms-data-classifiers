import pandas as  pd

from matplotlib import pyplot as plt
from matplotlib import mlab as mlb

import seaborn as sns
from scipy import stats

def classesDistribution():
    sns.countplot(x=Y)
    plt.xlabel('Mushroom Classes')
    plt.show()

#load mushroom data file
df = pd.read_csv('mushrooms.csv')

#Split dataset into X & Y
Y=df.iloc[:,0]
X=df.drop(['class'],axis=1)

#This function shows how the values of both class variables are distributed. We see that the data set
#have roughly both types of mushroom classes, 'p and 'e'; with 'e' type mushrooms being slightly more
#than 'p' type of mushrooms.
classesDistribution()

