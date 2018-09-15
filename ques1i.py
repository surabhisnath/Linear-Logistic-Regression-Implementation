# Import libraries
import pandas
import numpy
import math
import matplotlib.pyplot
import statistics
from sklearn.model_selection import KFold


# Define variables
numentries = 506
numfeatures = 13
lastcolumn = 14
numiterations = 1000
numfolds = 5

plotarrtrain = numpy.zeros([numfolds,numiterations])
plotarrval = numpy.zeros([numfolds,numiterations])

def graddesc(x, y, parameters,cnt):
    lengthx = len(x)
    lengthy = len(y)
    learningrate = 0.05
    RMSEarrx = []
    RMSEarry = []
    for i in range(numiterations):
        diffx = []
        diffy = []
        squareddiffx = []
        squareddiffy = []

        # Find RMSEs for train set
        for i in range(0,lengthx):
            temp = numpy.dot(parameters,x[i][0:(numfeatures+1)]) - x[i][lastcolumn]
            diffx.append(temp)
            squareddiffx.append(temp*temp)

        diffx = numpy.array(diffx)
        squareddiffx = numpy.array(squareddiffx)
        sumofsquareddiffx = sum(squareddiffx)
        RMSEarrx.append(math.sqrt(sumofsquareddiffx/lengthx))


        # Find RMSEs for validation set
        for i in range(0,lengthy):
            temp = numpy.dot(parameters,y[i][0:numfeatures+1]) - y[i][lastcolumn]
            diffy.append(temp)
            squareddiffy.append(temp*temp)
            
        diffy = numpy.array(diffy)
        squareddiffy = numpy.array(squareddiffy)
        sumofsquareddiffy = sum(squareddiffy)
        RMSEarry.append(math.sqrt(sumofsquareddiffy/lengthy))


        # Update parameters
        for j in range(0,numfeatures+1):
            parameters[j] = parameters[j] - learningrate*numpy.dot(x[:,j],diffx[:,numpy.newaxis])/lengthx

    RMSEarrx = numpy.array(RMSEarrx)
    RMSEarry = numpy.array(RMSEarry)
    
    plotarrtrain[cnt] = RMSEarrx
    plotarrval[cnt] = RMSEarry

    return RMSEarrx,RMSEarry
    





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

# Perform KFold

partitions = KFold(numfolds)
partitions.get_n_splits(x)

finalRMSEx = []
finalRMSEy = []

cnt = 0
numiter = list(range(1,numiterations+1))
for i,j in partitions.split(x):

    # Define theta(i)s
    parameters = numpy.zeros(numfeatures+1)
    
    train = x[i]
    val = x[j]
    t = graddesc(train,val,parameters,cnt)
    cnt+=1
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(numiter,t[0])
    finalRMSEx.append(t[0][len(t[0])-1])
    matplotlib.pyplot.xlabel('Number of Iterations')
    matplotlib.pyplot.ylabel('RMSE Error')
    matplotlib.pyplot.title('RMSE for the 5 Train Folds vs Number of Iterations')
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.plot(numiter,t[1])
    finalRMSEy.append(t[1][len(t[1])-1])
    matplotlib.pyplot.xlabel('Number of Iterations')
    matplotlib.pyplot.ylabel('RMSE Error')
    matplotlib.pyplot.title('RMSE for the 5 Validation Folds vs Number of Iterations')

print(finalRMSEx)
print(finalRMSEy)
print("Mean and standard deviation of RMS error for train set for 5 folds:", end=" ")
print(statistics.mean(finalRMSEx)," ","+-"," ",statistics.stdev(finalRMSEx))
print("Mean and standard deviation of RMS error for validation set for 5 folds:", end=" ")
print(statistics.mean(finalRMSEy)," ","+-"," ",statistics.stdev(finalRMSEy))

trainmean = numpy.mean(plotarrtrain,axis=0)
trainstd = numpy.mean(plotarrtrain,axis=0)
valmean = numpy.std(plotarrval,axis=0)
valstd = numpy.std(plotarrval,axis=0)

numiter = numpy.array(numiter)
tempnumiter = numiter[0::10]
temptrainmean = trainmean[0::10]
temptrainstd = trainstd[0::10]
tempvalstd = valstd[0::10]
tempvalmean = valmean[0::10]

matplotlib.pyplot.figure(3)
matplotlib.pyplot.bar(tempnumiter,temptrainmean,yerr=temptrainstd)
matplotlib.pyplot.plot(numiter,trainmean)
matplotlib.pyplot.xlabel('Number of Iterations')
matplotlib.pyplot.ylabel('RMSE Error')
matplotlib.pyplot.title('Mean RMSE for the 5 Train Folds with Standard Deviation Error Bars vs Number of Iterations')
matplotlib.pyplot.figure(4)
matplotlib.pyplot.bar(tempnumiter,tempvalmean,yerr=tempvalstd)
matplotlib.pyplot.plot(numiter,valmean)
matplotlib.pyplot.xlabel('Number of Iterations')
matplotlib.pyplot.ylabel('RMSE Error')
matplotlib.pyplot.title('Mean RMSE for the 5 Validation Folds with Standard Deviation Error Bars vs Number of Iterations')
#matplotlib.pyplot.axis([-2,1000,0,20])
matplotlib.pyplot.show()
