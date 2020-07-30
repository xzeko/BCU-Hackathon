
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 03:13:57 2020

@author: Zak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler


import tensorflow

from tensorflow import  keras as ks

data = pd.read_csv("covid_19_data.csv")


Countries  = ['South Africa','Egypt','South Sudan','Libya','Algeria']


saData = data[data['Country/Region']=="Libya"][['Confirmed','Deaths','Recovered']]



cnsa = saData.copy()

for i in range(1,14):
    shiftedSA = saData.shift(i)
    shiftedSA.columns=['C'+str(i),'D'+str(i),'R'+str(i)]    
    cnsa =  pd.concat([cnsa,shiftedSA],axis=1)


shiftedSA = saData.shift(-14)[['Confirmed','Deaths']]
shiftedSA.columns = ['CP','DP']
cnsa = pd.concat([cnsa,shiftedSA],axis=1)

cnsa = cnsa.iloc[13:-14]


xtrain = cnsa.iloc[:78,:-2]
xtest = cnsa.iloc[78:,:-2]
ytrain = cnsa['CP'].iloc[:78]
ytest = cnsa['CP'].iloc[78:]

ytrainDP = cnsa['DP'].iloc[:78]
ytestDP = cnsa['DP'].iloc[78:]

mmsX = MinMaxScaler()

mmsX.fit(pd.concat([cnsa.iloc[-1:,:-2]*15,xtrain],axis=0))

mmsXtrain = mmsX.transform(xtrain)
mmsXtest = mmsX.transform(xtest)


mmsY  = MinMaxScaler()

mmsY.fit(pd.concat([cnsa['CP'].iloc[-1:]*15,ytrain],axis=0).values.reshape(-1, 1))

mmsYtrain = mmsY.transform(ytrain.values.reshape(-1, 1))
mmsYtest = mmsY.transform(ytest.values.reshape(-1, 1))


mmsYDP = MinMaxScaler()
mmsYDP.fit(pd.concat([cnsa['DP'].iloc[-1:]*15,ytrainDP],axis=0).values.reshape(-1, 1))
mmsYDPtrain = mmsYDP.transform(ytrainDP.values.reshape(-1, 1))
mmsYDPtest = mmsYDP.transform(ytestDP.values.reshape(-1, 1))






model = ks.Sequential()

model.add(ks.layers.Dense(100,activation=ks.activations.relu,input_dim=mmsXtrain.shape[1]))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(100,activation=ks.activations.relu))
model.add(ks.layers.Dense(1,activation=ks.activations.linear))
model.compile(loss=ks.losses.mean_squared_error,optimizer=ks.optimizers.Adam())


model.fit(mmsXtrain,mmsYtrain,epochs=45,batch_size=3,validation_data=(mmsXtest,mmsYtest))
model.fit(mmsXtrain,mmsYtrain,epochs=1,batch_size=3,validation_data=(mmsXtest,mmsYtest))
ypred = model.predict(mmsXtest)


plt.plot(ytest.reset_index(drop=True))
plt.plot(mmsY.inverse_transform(ypred))




DataTOPlotConf = data[data['Country/Region']=="Libya"][['ObservationDate','Confirmed','Deaths']]


DataTOPlotConf['testConf'] = DataTOPlotConf['Confirmed']
DataTOPlotConf['testConf'].iloc[-14:]  = ytest.reset_index()['CP']


DataTOPlotConf['predictedConf'] = DataTOPlotConf['Confirmed']
DataTOPlotConf['predictedConf'].iloc[-14:] = mmsY.inverse_transform(ypred).reshape(14)



modelDP = ks.Sequential()

modelDP.add(ks.layers.Dense(100,activation=ks.activations.relu,input_dim=mmsXtrain.shape[1]))
modelDP.add(ks.layers.Dense(100,activation=ks.activations.relu))
modelDP.add(ks.layers.Dense(100,activation=ks.activations.relu))
modelDP.add(ks.layers.Dense(100,activation=ks.activations.relu))
modelDP.add(ks.layers.Dense(100,activation=ks.activations.relu))
modelDP.add(ks.layers.Dense(1,activation=ks.activations.linear))
modelDP.compile(loss=ks.losses.mean_squared_logarithmic_error,optimizer=ks.optimizers.Adam())


modelDP.fit(mmsXtrain,mmsYDPtrain,epochs=10,batch_size=2,validation_data=(mmsXtest,mmsYDPtest))
modelDP.fit(mmsXtrain,mmsYDPtrain,epochs=1,batch_size=2,validation_data=(mmsXtest,mmsYDPtest))
ypred = modelDP.predict(mmsXtest)


plt.plot(ytestDP.reset_index(drop=True))
plt.plot(mmsYDP.inverse_transform(ypred))









DataTOPlotConf['testDeath'] = DataTOPlotConf['Deaths']
DataTOPlotConf['testDeath'].iloc[-14:]  = ytestDP.reset_index()['DP']


DataTOPlotConf['predictedDeath'] = DataTOPlotConf['Deaths']
DataTOPlotConf['predictedDeath'].iloc[-14:]  =  mmsYDP.inverse_transform(ypred).reshape(14)



DataTOPlotConf.plot()

#DataTOPlotConf.to_csv("LibyaDataVizData.csv")
