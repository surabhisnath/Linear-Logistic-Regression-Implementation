from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

mnist = fetch_mldata('MNIST original')
trainx,testx,trainy,testy = train_test_split(mnist.data,mnist.target,test_size=1/7)
model = LogisticRegression()
model.fit(trainx,trainy)
outputs = model.predict(testx)
score = model.score(testx, testy)
print(score)

