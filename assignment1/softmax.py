import numpy as np
import mnist
import matplotlib.pyplot as plt

def one_hot_encode(Y):
    aux = np.zeros((Y.shape[0], 10));
    
    i = 0;
    for value in Y:
        aux[i, value] = 1;
        i += 1;
        
    return aux;

def filter_numbers(X, Y, number):
    Y = np.argmax(Y, axis=1);
    X = X[Y==number];

    return X;

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

def cross_entropy(W, X, y):
    k_acc = 0;
    
    for i in range(0, X.shape[0]):
        for k in range (0, 10):
            s = logistic(np.dot(X[i], W[k]));
            if (s < 0.00001):
                s = 0.00001;
            k_acc += y[i,k]*np.log(s);
        
    CE = -k_acc/ (X.shape[0]*10);
    return CE

def accuracy (w, X, Y):
     Y_predict = classify(w, X);
     cmp = np.argmax(Y, axis=1) == np.argmax(Y_predict, axis=1);
     
     pp = np.sum(cmp)/ cmp.shape[0];
     
     return pp;

"""Stochastic Gradisnt Descent"""
def gradient_descent(W, X, Y, X_val, Y_val, X_test, Y_test, Ece, Ece_t, Ece_v, y_pp_test, y_pp_train, y_pp_val, n): 
    LAMBDA = 0.0001;
    
    X, Y = shuffle_data(X, Y);
    
    epoch_lenght = X.shape[0];
    measure_interval = np.floor(epoch_lenght/1);
    
    for i in range(0, X.shape[0]):
        for k in range(0,10):
            dece = DECE(W[k].reshape(785,1), X[i], Y[i,k]);
            W[k] = W[k] - n*(dece + 2*LAMBDA*W[k]);
            
        if ((i%measure_interval) == 0.0):
            Ece = np.append(Ece, cross_entropy(W, X, Y));
            Ece_t = np.append(Ece_t, cross_entropy(W, X_test, Y_test));
            Ece_v = np.append(Ece_v, cross_entropy(W, X_val, Y_val));
            
            y_pp_test = np.append(y_pp_test, accuracy(W, X_test, Y_test));
            y_pp_train = np.append(y_pp_train, accuracy(W, X, Y));
            y_pp_val = np.append(y_pp_val, accuracy(W, X_val, Y_val));

            
    return W, Ece, Ece_t, Ece_v, y_pp_test, y_pp_train, y_pp_val;

"""Get one-hot encode classification for a given X and W"""
def classify (W,X):
    Y = np.zeros(X.shape[0]);
    s = np.zeros((X.shape[0], 10));
    
    for i in range (0, X.shape[0]):
        for k in range (0, 10):
            s[i,k] = logistic(np.dot(X[i], W[k]));
        Y[i] = np.argmax(s[i]);
    
    Y = Y.astype(int);
    Y = one_hot_encode(Y);
    return Y;


def annealing_LR(lr,t):
    T = 100;
    lr = lr/(1 + t/T);
    
    return lr;

if __name__ == '__main__':
    LR = 0.00001;
    IT = 100;
    
    #mnist.init()
    X_train, Y_train, X_test, Y_test = mnist.load()
    
    #get subset of training and test datasets 
    X_train = X_train[0:20000, :];
    Y_train = Y_train[0:20000];
    X_test = X_test[8000:10000, :];
    Y_test = Y_test[8000:10000];
    
        #subset for development
#    X_train = X_train[0:2000, :];
#    Y_train = Y_train[0:2000];
#    X_test = X_test[0:200, :];
#    Y_test = Y_test[0:200];
    
    #add '1' to include bias on input vector x
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1);
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1);
    
    #one-hot encoding
    Y_train = one_hot_encode(Y_train);
    Y_test = one_hot_encode(Y_test);
    
    #slipt into training and Validation
    X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1);
    
    #create random Weight vector
    w = np.random.random_sample(size=(10,785));
    
    #errors
    Ece = np.zeros(1);
    Ece_t = np.zeros(1);
    Ece_v = np.zeros(1);
    
    #predictions percentage
    y_pp_test = np.zeros(1);
    y_pp_train = np.zeros(1);
    y_pp_val = np.zeros(1);
    
    
    w_list = np.zeros((IT,10,785));
    w_list[0] = w;
    i = 0;
    Ece_v_counter = 0;
    lr = np.zeros(IT);
    
    #epochs
    while (Ece_v_counter < 4 and i < IT):
        lr[i] = annealing_LR(LR,i);
        
        w, Ece, Ece_t, Ece_v, y_pp_test, y_pp_train, y_pp_val = gradient_descent(w, X_train, Y_train,  X_val, Y_val, X_test, Y_test, 
                                                                                 Ece, Ece_t, Ece_v, y_pp_test, y_pp_train, y_pp_val, lr[i]);

        w_list[i] = w;
        
        #early stop
        if (i > 1):
            if (Ece_v[-1] >= Ece_v[-2]):
                Ece_v_counter += 1;
            else:
                Ece_v_counter = 0;

        i += 1;
        
    Y_predict = classify(w, X_test);
    
    
    plt.figure();
    plt.plot(Ece[2:-1], label='Train');
    plt.plot(Ece_v[2:-1], label='Validation');
    plt.plot(Ece_t[2:-1], label='Test');
    plt.legend();
    plt.ylim([0,0.7]);
    plt.title("Cross-Entropy Loss Function")
    plt.xlabel('Iterations');
    plt.ylabel('error');
    plt.show();
    
    plt.figure();
    plt.plot(y_pp_train[1:-1], label='Train');
    plt.plot(y_pp_val[1:-1], label='Validation');
    plt.plot(y_pp_test[1:-1], label='Test');
    plt.legend();
    plt.ylim([0.4,1]);
    plt.title("Percentage of right classification")
    plt.xlabel('Iterations');
    plt.ylabel('%');
    plt.show();
    
    #plot all weight images together
    fig=plt.figure(figsize=(6, 8))
    columns = 2
    rows = 5
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(w[i-1,1:].reshape(28,28))
    plt.show()
    
    #plot weihts along with mean of X
    for i in range (10):
        fig=plt.figure(figsize=(6, 4))
        columns = 2
        rows = 1
        X = filter_numbers(X_train, Y_train, i);
        X = X.mean(axis=0);
        fig.add_subplot(rows, columns, 1)
        plt.imshow(X[1:].reshape(28,28));
        fig.add_subplot(rows, columns, 2)
        plt.imshow(w[i,1:].reshape(28,28)) 
        plt.show()
    
    
        
        
    

    
    
    
    
    
    
    
