# Import libraries
import pandas
import numpy
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot

def graddesc(x, y, parameters):
    lengthx = len(x)
    lengthy = len(y)
    learningrate = 0.0009
    RMSEarrx = []
    RMSEarry = []
    for i in range(1000):
        diffx = []
        diffy = []
        squareddiffx = []
        squareddiffy = []

        # Find RMSEs for train set
        for i in range(0,lengthx):
            temp = numpy.dot(parameters,x[i][0:14]) - x[i][14]
            diffx.append(temp)
            squareddiffx.append(temp*temp)

        diffx = numpy.array(diffx)
        squareddiffx = numpy.array(squareddiffx)
        sumofsquareddiffx = sum(squareddiffx)
        RMSEarrx.append(math.sqrt(sumofsquareddiffx/lengthx))


        # Find RMSEs for test set
        for i in range(0,lengthy):
            temp = numpy.dot(parameters,y[i][0:14]) - y[i][14]
            diffy.append(temp)
            squareddiffy.append(temp*temp)
            
        diffy = numpy.array(diffy)
        squareddiffy = numpy.array(squareddiffy)
        sumofsquareddiffy = sum(squareddiffy)
        RMSEarry.append(math.sqrt(sumofsquareddiffy/lengthx))


        # Update parameters
        for j in range(0,14):
            parameters[j] = parameters[j] - learningrate*numpy.dot(x[:,j],diffx[:,numpy.newaxis])/lengthx

    RMSEarrx = numpy.array(RMSEarrx)
    RMSEarry = numpy.array(RMSEarry)

    return RMSEarrx,RMSEarry
    
# Define variables
numentries = 506
numfeatures = 13


# Put data into numpy array
data = pandas.read_csv('C:\\Users\\Surabhi\\Desktop\\IIITD\\5th SEM\\ML\\Assignments\\boston_CSV.csv')
x = numpy.array(data)

# Perform Feature Normalization (exclude output column)
for i in range(0,numfeatures):
    mini = min(x[:,i])
    maxi = max(x[:,i])
    x[:,i] = (x[:,i]-mini)/(maxi-mini)

# Append column of 1s
col1 = numpy.ones(numentries)
x = numpy.concatenate((col1[:,numpy.newaxis],x),axis=1)

# Define theta(i)s
parameters = numpy.zeros(14)

# Perform KFold

partitions = KFold(5)
partitions.get_n_splits(x)


for i,j in partitions.split(x):
    train = x[i]
    val = x[j]
    numiter = list(range(1000))
    numiter.append(1000)
    numiter.remove(0)
    t = graddesc(train,val,parameters)

    matplotlib.pyplot.plot(numiter,t[0])
    #matplotlib.pyplot.plot(numiter,t[1])

print("hello") 
matplotlib.pyplot.show()
