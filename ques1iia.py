# RIDGE REGRESSION

# Import libraries
import pandas
import numpy
import math
import matplotlib.pyplot
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


def logregr(x, parameters):
    lengthx = len(x)
    learningrate = 0.001
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
        for j in range(0,numfeatures+1):
            parameters[j] = parameters[j] - learningrate*numpy.dot(x[:,j],diffx[:,numpy.newaxis])/lengthx - lambdaval*learningrate*parameters[j]/lengthx

    RMSEarrx = numpy.array(RMSEarrx)

    return RMSEarrx




# Define variables
numentries = 506
numfeatures = 13
lastcolumn = 14
numiterations = 5000
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

testdata = x[405:506,:]
x = x[0:405,:]

classifier = GridSearchCV(linear_model.Ridge(),{'alpha':[25,10,3,1,0.8,0.5,0.3,0.1,0.05,0.03]},cv=5)
classifier.fit(x[:,0:14],x[:,14])
lambdaval = classifier.best_params_.get('alpha')

parameters = numpy.zeros(numfeatures+1)
numiter = list(range(1,numiterations+1))
t = logregr(x,parameters)

# Find test RMSE
diff = []
squareddiff = []

for i in range(0,len(testdata)):
    temp = numpy.dot(parameters,x[i][0:(numfeatures+1)]) - x[i][lastcolumn]
    diff.append(temp)
    squareddiff.append(temp*temp)

diff = numpy.array(diff)
squareddiff = numpy.array(squareddiff)
sumofsquareddiff = sum(squareddiff)
RMSEtest = math.sqrt(sumofsquareddiff/len(testdata))


print("RMSE error of test set is: ",end="")
print(RMSEtest)
matplotlib.pyplot.plot(numiter,t)
matplotlib.pyplot.show()


'''
output = classifier.predict(testdata[:,0:14]);

print(output)
print(testdata[:,14])

sumofsquarederrors = 0
for i in range(0,len(output)):
    sumofsquarederrors = sumofsquarederrors + ((output[i]-testdata[i,14])*(output[i]-testdata[i,14]))

meansumofsquarederrors = sumofsquarederrors/len(output)

print(math.sqrt(meansumofsquarederrors))
'''
