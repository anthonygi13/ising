#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : mc_functions.py
# Created by Anthony Giraudo adn Clement Sebastiao the 31/10/2020

"""
Nous utilisons tensorflow pour vérifier les capacités de nos données à
être interprété par un modèle de machine learning.
"""

# Modules
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Functions
def split_data(data,val,ratio):
    # index=list(zip(data,val))
    # np.random.shuffle(index)
    # data,val=zip(*index)
    data_train=[]
    val_train=[]    
    data_test=[]
    val_test=[]
    count0=0
    count1=0
    ratio=0.8
    for i in range(len(val)):
        if val[i][0]==1:
            if count1<len(val)*0.5*ratio:
                count1+=1
                data_train+=[data[i]]
                val_train+=[val[i][0]]
            else:
                data_test+=[data[i]]
                val_test+=[val[i][0]]
        else:
            if count0<len(val)*0.5*ratio:
                count0+=1
                data_train+=[data[i]]
                val_train+=[val[i][0]]
            else:
                data_test+=[data[i]]
                val_test+=[val[i][0]]
    data_train=np.asarray(data_train)
    data_test=np.asarray(data_test)
    val_train=np.asarray(val_train)
    val_test=np.asarray(val_test)
    return data_train,val_train,data_test,val_test

def plot_loss(history, label):
    plt.plot(history.epoch,history.history['loss'], 
                 label='Train '+label)
    plt.plot(history.epoch,history.history['val_loss'],
             label='Val '+label, linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Valeur de la fonction de coût au cours des epochs')
    plt.legend()  

def show(array):
    size=int(np.sqrt(len(array)))
    image=np.reshape(array,(size,size))
    plt.figure()
    plt.imshow(image)
    plt.show()
    
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    size=int(np.sqrt(len(img)))
    img=np.reshape(img,(size,size))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                  100*np.max(predictions_array),
                                  true_label),color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

    
# Main

if __name__ == "__main__":
    data=[]
    for line in open("set1/vectors", "r"):
        data.append([int(x) for x in line.split()])
    val=[]
    for line in open("set1/classification","r"):
        val.append([int(x) for x in line.split()])
    
    ratio=0.9
    data_train,val_train,data_test,val_test=split_data(data,val,ratio)
    
    model=tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation='relu'),
                                      tf.keras.layers.Dense(2)])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history=model.fit(data_train,val_train,epochs=10,validation_data=(data_test,val_test))
    model.evaluate(data_test,val_test,verbose=2)
    
    
    plot_loss(history,'data')
    
    
    proba=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    predictions=proba(data_test)
    
    i=4
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.title('Prédiction sur la donnée de test n°{}'.format(i))
    plot_image(i,predictions[i],val_test,data_test)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],val_test)
    plt.show()

    

    
    
                
                
            
    