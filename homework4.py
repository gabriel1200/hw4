import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

def normlize_rgb(x):
    x/= 255.0
    return x
def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test
    classes = 10

    targets = ytrain.reshape(-1)
    ytrain_1hot = np.eye(classes)[targets]
    
    targets2 = ytest.reshape(-1)
    ytest_1hot = np.eye(classes)[targets2]
  
    xtrain = np.divide(xtrain,255)
    xtest = np.divide(xtest,255)


    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape =(32,32,3)))
    
    hidden = Dense(units=100, activation="relu", input_shape=(32,32,3))
    nn.add(hidden)
    
    output = Dense(units=10, activation="softmax")
    nn.add(output)

    

    return nn


def train_multilayer_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)        
 

def build_convolution_nn():
    nn = Sequential()
    conv1 = Conv2D(32, (3, 3), activation='relu', padding="same",input_shape =(32,32,3))
    nn.add(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding="same")
    nn.add(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))
    nn.add(pool1)
    drop1 = Dropout(0.25)
    nn.add(drop1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding="same")
    nn.add(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding="same")
    nn.add(conv4)
    pool3 = MaxPooling2D(pool_size=(4, 4))
    nn.add(pool3)
    drop2 = Dropout(0.5)
    nn.add(drop2)
    nn.add(Flatten())
    hidden1 = Dense(units=250, activation="relu", input_shape=(32,32,3))
    hidden2 = Dense(units=100, activation="relu", input_shape=(32,32,3))
    nn.add(hidden1)
    nn.add(hidden2)
    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn



def train_convolution_nn(model,xtrain,ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)        
 

def get_binary_cifar10():    
    pass


def build_binary_classifier():    
    pass


def train_binary_classifier():
    pass


if __name__ == "__main__":
    load_cifar10()
    nn  = build_convolution_nn()
    print(nn.summary())
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    x =train_convolution_nn(nn, xtrain, ytrain_1hot)
    nn.evaluate(xtest,ytest_1hot)
    
    # Write any code for testing and evaluation in this main section.


