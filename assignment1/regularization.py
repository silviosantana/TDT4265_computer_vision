# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import mnist
import matplotlib.pyplot as plt

def filter_2_and_3(X, Y):
    df = pd.DataFrame(data=np.concatenate((X, Y.reshape(Y.shape[0],1)), axis=1))
    df = df.loc[df[784].isin([2,3])]
    
    df.loc[(df[784] == 2), 784] = 0
    df.loc[(df[784] == 3), 784] = 1
    
    array = df.values;
    X = array[:,0:784]
    Y = array[:,784]
    Y = Y.reshape(Y.shape[0], 1)
    
    return [X, Y];


def train_val_split(X, Y, val_percentage):
    dataset_size = X.shape[0];
    idx = np.arange(0, dataset_size);
    np.random.shuffle(idx);
      
    train_size = int(dataset_size*(1-val_percentage));
    idx_train = idx[:train_size];
    idx_val = idx[train_size:];
    X_train, Y_train = X[idx_train], Y[idx_train];
    X_val, Y_val = X[idx_val], Y[idx_val];
    
    return X_train, Y_train, X_val, Y_val;

def shuffle_data(X, Y):
    idx = np.arange(0, X.shape[0]);
    np.random.shuffle(idx);

    X, Y = X[idx], Y[idx];
    
    return X, Y;

def logistic(z):
    s = 1/(1 + np.exp(-z));
    
    return s;

""""Derivative of Cross Entropy Error"""
def DECE(W, X, y):
    s = logistic(np.dot(X, W));
    dece = (s - y)*X;
    
    return dece;

"""Stochastic Gradisnt Descent"""
def gradient_descent(W1, W2, W3, W1_l, W2_l, W3_l, X, Y, X_val, Y_val, X_test, Y_test, Ece, Ece_t, Ece_v, 
                     y_pp_test, y_pp_val_1, y_pp_val_2, y_pp_val_3, y_pp_train, n): 
    X, Y = shuffle_data(X, Y);
    
    epoch_lenght = X.shape[0];
    measure_interval = np.floor(epoch_lenght/3);
    
    for i in range(0, X.shape[0]):
        dece = DECE(W1, X[i], Y[i]);
        W1 = W1 - n*(dece.reshape(W1.shape[0],1) + 2*1*W1);
        
        dece = DECE(W2, X[i], Y[i]);
        W2 = W2 - n*(dece.reshape(W2.shape[0],1) + 2*0.01*W2);
        
        dece = DECE(W3, X[i], Y[i]);
        W3 = W3 - n*(dece.reshape(W3.shape[0],1) + 2*0.00000001*W3);
        
        if ((i%measure_interval) == 0.0):
            Ece = np.append(Ece, cross_entropy(W1, X_train, Y_train));
            Ece_t = np.append(Ece_t, cross_entropy(W1, X_test, Y_test));
            Ece_v = np.append(Ece_v, cross_entropy(W1, X_val, Y_val));
            
            y_pp_test = np.append(y_pp_test, accuracy(W1, X_test, Y_test));
            y_pp_train = np.append(y_pp_train, accuracy(W1, X_train, Y_train));
            
            y_pp_val_1 = np.append(y_pp_val_1, accuracy(W1, X_val, Y_val));
            y_pp_val_2 = np.append(y_pp_val_2, accuracy(W2, X_val, Y_val));
            y_pp_val_3 = np.append(y_pp_val_3, accuracy(W3, X_val, Y_val));
            
            W1_l = np.append(W1_l, np.linalg.norm(W1));
            W2_l = np.append(W2_l, np.linalg.norm(W2));
            W3_l = np.append(W3_l, np.linalg.norm(W3));
            
    
    return W1, W2, W3, W1_l, W2_l, W3_l, Ece, Ece_t, Ece_v, y_pp_test, y_pp_val_1, y_pp_val_2, y_pp_val_3, y_pp_train;

def cross_entropy(W, X, y):
    CE_acc = 0;
    y_pred = np.zeros((y.shape[0],1))
    
    for i in range(0, X.shape[0]):
        s = logistic(np.dot(X[i], W));
        y_pred[i] = s;
        #print("i: {}  s: {}".format(i,s))
        if (s < 0.00001):
            s = 0.00001;
        if (s==1):
            s=0.999999;
        CE_acc += y[i]*np.log(s) + (1 - y[i])*np.log(1 - s);
        
    CE = -CE_acc/ X.shape[0];
    #CE = sklearn.metrics.log_loss(y,y_pred)
    return CE

"""Get classification Y for a given X and W"""
def classify (W,X):
    Y = np.zeros(X.shape[0]);
    
    for i in range (0, X.shape[0]):
        s = logistic(np.dot(X[i], W));
        if (s >= 0.5):
            Y[i] = 1.0;
        else:
            Y[i] = 0.0;
            
    return Y;

def accuracy (w, X, Y):
     Y_predict = classify(w, X);
     acr = 1 - np.sum(np.abs(Y - Y_predict.reshape(Y.shape[0],1)))/Y.shape[0];
     
     return acr;

def annealing_LR(lr,t):
    T = 100;
    lr = lr/(1 + t/T);
    
    return lr;

if __name__ == '__main__':
    LR = 0.00001;
    IT = 1000;
    
    #mnist.init()
    X_train, Y_train, X_test, Y_test = mnist.load()
    
    #get subset of training and test datasets 
    X_train = X_train[0:20000, :];
    Y_train = Y_train[0:20000];
    X_test = X_test[8000:10000, :];
    Y_test = Y_test[8000:10000];
    
    X_test, Y_test = filter_2_and_3(X_test, Y_test);
    X_train, Y_train = filter_2_and_3(X_train, Y_train);

    #add '1' to include bias on input vector x
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1);
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1);
    
    #subset for development
#    X_train = X_train[0:100, :];
#    Y_train = Y_train[0:100];
#    X_test = X_test[0:100, :];
#    Y_test = Y_test[0:100];
    
    #slipt into training and Validation
    X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1);
    
    
    #errors
    Ece = np.zeros(1);
    Ece_t = np.zeros(1);
    Ece_v = np.zeros(1);
    
    #predictions percentage
    y_pp_test = np.zeros(1);
    y_pp_train = np.zeros(1);
    y_pp_val_1 = np.zeros(1);
    y_pp_val_2 = np.zeros(1);
    y_pp_val_3 = np.zeros(1);
    
    
    lr = np.zeros(IT);
   
    
    Ece_v_min_index = 0;
    Ece_v_counter = 0
        
    w1 = np.random.random_sample(size=(785,1));
    w2 = w1;
    w3 = w1;
    
    #one weight vector per lambda value
    W1 = np.zeros((IT,785,1));
    W2 = W1;
    W3 = W1;
    
    #keep history of weight vectors
    W1_l = np.zeros(1);
    W2_l = np.zeros(1);
    W3_l = np.zeros(1);
    
    i = 0;
    
    #epochs
    while (Ece_v_counter < 3 and i < IT):
        lr[i] = annealing_LR(LR,i);
        w1, w2, w3, W1_l, W2_l, W3_l, Ece, Ece_t, Ece_v, y_pp_test, y_pp_val_1, y_pp_val_2, y_pp_val_3, y_pp_train = gradient_descent(w1, w2, w3, W1_l, W2_l, W3_l, X_train, Y_train, X_val, Y_val, 
                                 X_test, Y_test,Ece, Ece_t, Ece_v, 
                                 y_pp_test, y_pp_val_1, y_pp_val_2, y_pp_val_3, y_pp_train, lr[i]);
        W1[i] = w1;
        W2[i] = w2;
        W3[i] = w3;
        
        if (i > 1):
            if (Ece_v[-1] >= Ece_v[-2]):
                Ece_v_counter += 1;
            else:
                Ece_v_counter = 0;
        
        i += 1;  
    
    if (i < IT):
        w1 = W1[i-1];
        w2 = W2[i-1];
        w3 = W3[i-1];
        
    
    
        
    Y_predict = classify(w1, X_test);
    
    plt.figure();
    plt.plot(Ece[1:-1], label='Train');
    plt.plot(Ece_v[1:-1], label='Validation');
    plt.plot(Ece_t[1:-1], label='Test');
    plt.legend();
    plt.ylim([0,3]);
    plt.title("Cross-Entropy Loss Function")
    plt.xlabel('Iterations');
    plt.ylabel('error');
    plt.show();
    
    plt.figure();
    plt.plot(y_pp_train[1:-1], label='Train');
    plt.plot(y_pp_val_3[1:-1], label='Validation');
    plt.plot(y_pp_test[1:-1], label='Test');
    plt.legend();
    plt.ylim([0.8,1]);
    plt.title("Classification Accuracy")
    plt.xlabel('Iterations');
    plt.ylabel('%');
    plt.show();
    
    plt.figure();
    plt.plot(y_pp_val_1[1:-1], label='1');
    plt.plot(y_pp_val_2[1:-1], label='0.01');
    plt.plot(y_pp_val_3[1:-1], label='0.00001');
    plt.legend();
    plt.ylim([0.8,1]);
    plt.title("Validation set Accuracy for each Lambda")
    plt.xlabel('Iterations');
    plt.ylabel('%');
    plt.show();
    
    plt.figure();
    plt.plot(W1_l[1:-1], label='1');
    plt.plot(W2_l[1:-1], label='0.01');
    plt.plot(W3_l[1:-1], label='0.00001');
    plt.legend();
    plt.title("Length of weights for each Lambda")
    plt.xlabel('Iterations');
    plt.ylabel('Magnitude');
    plt.show();
    
    
    plt.figure();
    plt.title("Weigth for lambda = 1")
    plt.imshow(w1[1:].reshape(28,28))
    
    plt.figure();
    plt.title("Weigth for lambda = 0.01")
    plt.imshow(w2[1:].reshape(28,28))
    
    plt.figure();
    plt.title("Weigth for lambda = 0.00001")
    plt.imshow(w3[1:].reshape(28,28))
    
    
    
    
    
    
    
    