from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy
import math

mnist = fetch_mldata('MNIST original')

trainx,testx,trainy,testy = train_test_split(mnist.data,mnist.target,test_size=1/7)
'''
trainx = trainx[:1000]
trainy = trainy[:1000]
testx = testx[:100]
testy = testy[:100]
'''

trainyeach = numpy.zeros([10,len(trainy)])
arrofy = testy
probeachtesteachclass = numpy.zeros([len(testy),10])
probeachtraineachclass = numpy.zeros([len(trainy),10])

for i in range(0,10):
    for j in range(0,len(trainy)):
        if trainy[j]==i:
            trainyeach[i,j] = 1
        else:
            trainyeach[i,j] = 0
    model = LogisticRegression(penalty='l1')
    model.fit(trainx,trainyeach[i])
    parameters = model.coef_

    for k in range(0,len(testy)):
        temp = numpy.array(testx[k])
        val = -1*numpy.dot(parameters,temp)
        probeachtesteachclass[k][i] = 1/(1+math.exp(val))

    for n in range(0,len(trainy)):
        temp = numpy.array(trainx[n])
        val = -1*numpy.dot(parameters,temp)
        probeachtraineachclass[n][i] = 1/(1+math.exp(val))


predictedytest = numpy.zeros(len(testy))
predictedytrain = numpy.zeros(len(trainy))

for l in range(0,len(testy)):
    predictedytest[l] = numpy.argmax(probeachtesteachclass[l])
for l in range(0,len(trainy)):
    predictedytrain[l] = numpy.argmax(probeachtraineachclass[l])

restest = numpy.zeros([2,10])
for m in range(0,len(testy)):
    index = int(testy[m]);
    restest[0,index] = restest[0,index] + 1;
    if testy[m] == predictedytest[m]:
        restest[1,index] = restest[1,index] + 1;

restrain = numpy.zeros([2,10])
for m in range(0,len(trainy)):
    index = int(trainy[m]);
    restrain[0,index] = restrain[0,index] + 1;
    if trainy[m] == predictedytrain[m]:
        restrain[1,index] = restrain[1,index] + 1;

#print(restest[0])
#print(restest[1])
#print(testy)
#print(predictedytest)

classvizeacctest = restest[1]/restest[0]*100
differencestest = restest[0]-restest[1]
print(classvizeacctest)
testacctest = (1 - sum(differencestest)/len(testy))*100
print(testacctest)



#print(restrain[0])
#print(restrain[1])
#print(trainy)
#print(predictedytrain)

classvizeacctrain = restrain[1]/restrain[0]*100
differencestrain = restrain[0]-restrain[1]
print(classvizeacctrain)
testacctrain = (1 - sum(differencestrain)/len(trainy))*100
print(testacctrain)
