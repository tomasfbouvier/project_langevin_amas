# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:01:05 2021

@author: LikeNoOthers
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf


c= pd.read_csv("foo3.csv")
#print(c)
tester_2E4=pd.read_csv("test_2E4.csv")
#plt.figure()


#xplot = np.linspace(0,len(c.loc[0][0:-3]))


"""
plt.figure()
plt.plot(range(len(c.loc[0][0:-3])),c.loc[0][0:-3])


plt.figure()
plt.plot(range(len(c.loc[1][0:-3])),c.loc[1][0:-3])

"""
array = c.loc[0][0:-3]
test = np.reshape(np.array(array),(110,110))


def get_model2(n_inputs=12100):
    initializer = tf.keras.initializers.GlorotUniform()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(n_inputs,)))
    model.add(tf.keras.layers.Dense(25, activation='linear', kernel_initializer=initializer))
    #model.add(tf.keras.layers.Dense(25, activation='linear', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(12, activation='linear', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(3, activation='linear',kernel_initializer=initializer))
    model.compile(optimizer='Adam',
              loss='mse',
              metrics=['accuracy'])
    return(model)

#print(c)
X= c.iloc[:,0:-3]
y= c.iloc[:,-3:]
X= X.to_numpy()
y= y.to_numpy()

y_means=[] ; y_stds=[]
for i in range(np.shape(y)[1]):
    y_mean=np.mean(y[:,i]); y_means.append(y_mean);y_std= np.std(y[:,i]); y_stds.append(y_std)
    y[:,i]=y[:,i]-y_mean;  y[:,i]=y[:,i]/y_std;


def models_generator(n_models=5, epochs=1000):
    models=[]
    plt.figure()
    for i in range(n_models):
        model= get_model2()
        history=model.fit(X,y, epochs=epochs)
        plt.plot(history.history['accuracy'], label= ("model "+ str(i+1)))
        models.append(model)
    plt.legend(loc="best", fontsize=15)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    return(models)
    
#history=model.fit(X,y, epochs=300)
#history2= model2.fit(X,y, epochs=300)
#plt.plot(history.history['accuracy'])
#plt.plot(history2.history['accuracy'])
""""
# summarize performance
#print('MAE: %.3f (%.3f)' % (mean(results), std(results)));

tester_2E4=pd.read_csv("test_2E4.csv")
#tester
X= tester_2E4.iloc[:,0:-3]
y= tester_2E4.iloc[:,-3:]
X= X.to_numpy()
y= y.to_numpy()


for i in range(np.shape(y)[1]):
    y[:,i]=y[:,i]-y_means[i];  y[:,i]=y[:,i]/y_stds[i];

mae = model.evaluate(X, y, verbose=0)
print(mae)
#print("test1", mae

"""
def inverse_scaling(y):
    for i in range(len(y_means)):
        y[0,i]*=y_stds[i]; y[0,i]+=y_means[i]
    return y
def u_inverse_scaling(u_pred):
    for i in range(len(u_pred[0])):
        u_pred[0,i]*=y_stds[i]; 
    return u_pred


def predictor(X, models):
    X=np.reshape(X, (1,12100))
    ys=[]
    for i in range(len(models)):
        ys.append(models[i].predict(X))
    y=inverse_scaling(np.mean(ys, axis=0))
    y_stds= u_inverse_scaling(np.std(ys, axis=0))
    return(y, y_stds)    
