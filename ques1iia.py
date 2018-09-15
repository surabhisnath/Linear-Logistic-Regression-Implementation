# RIDGE REGULARIZATION

# Import libraries
import pandas
import numpy
import math
import matplotlib.pyplot
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


def graddescl2(x, parameters):
    lengthx = len(x)
    learningrate = 0.05
    RMSEarrx = []
    for i in range(numiterations):
        diffx = []
        squareddiffx = []

        # Find RMSEs for train set
        for i in range(0,lengthx):
            temp = numpy.dot(parameters,x[i][0:(numfeatures+1)]) - x[i][lastcolumn]
            diffx.append(temp)
            squareddiffx.append(temp*temp)

        diffx = numpy.array(diffx)
        squareddiffx = numpy.array(squareddiffx)
        sumofsquareddiffx = sum(squareddiffx)
        RMSEarrx.append(math.sqrt(sumofsquareddiffx/lengthx))

        # Update parameters
        parameters[0] = parameters[0] - learningrate*numpy.dot(x[:,0],diffx[:,numpy.newaxis])/lengthx
        for j in range(1,numfeatures+1):
            parameters[j] = parameters[j] - learningrate*numpy.dot(x[:,j],diffx[:,numpy.newaxis])/lengthx - lambdaval*learningrate*parameters[j]/lengthx

    RMSEarrx = numpy.array(RMSEarrx)

    return RMSEarrx




# Define variables
numentries = 506
numfeatures = 13
lastcolumn = 14
numiterations = 1000
numfolds = 5

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

testdata = x[0:103,:]
x = x[103:506,:]
#testdata = x[405:506,:]
#x = x[0:405,:]


classifier = GridSearchCV(linear_model.Ridge(),{'alpha':[25,10,3.5,3,2.5,2,1.5,1,0.8,0.5,0.3,0.1,0.05,0.03]},"neg_mean_squared_error",cv=5)
classifier.fit(x[:,0:14],x[:,14])
lambdaval = classifier.best_params_.get('alpha')
print("Lambda value: ", end='')
print(lambdaval)

parameters = numpy.zeros(numfeatures+1)
numiter = list(range(1,numiterations+1))
t = graddescl2(x,parameters)

# Find test RMSE
diff = []
squareddiff = []

for i in range(0,len(testdata)):
    temp = numpy.dot(parameters,testdata[i][0:(numfeatures+1)]) - testdata[i][lastcolumn]
    diff.append(temp)
    squareddiff.append(temp*temp)

diff = numpy.array(diff)
squareddiff = numpy.array(squareddiff)
sumofsquareddiff = sum(squareddiff)
RMSEtest = math.sqrt(sumofsquareddiff/len(testdata))


print("RMSE error of test set is: ",end="")
print(RMSEtest)
matplotlib.pyplot.plot(numiter,t)
matplotlib.pyplot.xlabel('Number of Iterations')
matplotlib.pyplot.ylabel('RMSE Error')
matplotlib.pyplot.title('RMSE vs Number of Iterations for Held Out Test set with L2 Regularization')
matplotlib.pyplot.show()
