import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt


data=pd.read_csv('data_sets.csv')
print("The data first 5 record \n",data.head(5))
print("\n")
print("The dimension of the csv \nt ",data.shape)
print("\n")
print("Information about the csv  ",data.info())
print("\n")

train=np.array(data.iloc[0:518])
test=np.array(data.iloc[518:768])

print("The training data sets \n ",train.shape)
print("The training data set percentage \n", (518/768)*100)
print("\n")
print("The testing data sets \n) ",test.shape)
print("The training data set percentage \n", (250/768)*100)
print("\n")

model =GaussianNB()

model.fit(train[:,0:8], train[:,8])

predicted = model.predict(test[:,0:8])
print(test[:,8])
print("\n")
print(predicted)

count=0
for l in range(250):
    if(predicted[l]==test[1,8]):
        count=count+1



print("\nThe count of correct prediction \n",count)

print("\nThe accuracy of the algorithm\n",count/250)
