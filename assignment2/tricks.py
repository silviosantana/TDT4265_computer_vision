import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm
import copy
import time


def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing) 

def shuffle_data(X, Y):
    idx = np.arange(0, X.shape[0]);
    np.random.shuffle(idx);

    X, Y = X[idx], Y[idx];
    
    return X, Y;

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx) 
  
  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_train, Y_train = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def sigmoid(z):
    s = 1/(1 + np.exp(-z)); 
    return s;

def sigmoid_tanh(z):
    s = 1.7159*np.tanh((2/3)*z)
    return s;

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def forward(a, w):
    z = a.dot(w.T)
    return softmax(z)

def forward_hidden(X, w):
    z = X.dot(w.T)
    #return sigmoid(z)
    return sigmoid_tanh(z)

def calculate_accuracy(X, targets, w):
    j = 0
    k = 1
    
    a_hidden = forward_hidden(X, w[j])
    output = forward(a_hidden, w[k])

    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(X, targets, w):
    j = 0
    k = 1
    
    a_hidden = forward_hidden(X, w[j])
    output = forward(a_hidden, w[k])

    assert output.shape == targets.shape 
    #output[output == 0] = 1e-8
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    #print(cross_entropy.shape)
    return cross_entropy.mean()

def check_gradient(X, targets, w, epsilon, computed_gradient, layer):    
    print("Checking gradient...")
    dw = np.zeros_like(w[layer])
    for k in range(w[layer].shape[0]):
        for j in range(w[layer].shape[1]):
            new_weight1, new_weight2 = copy.deepcopy(w), copy.deepcopy(w)
            new_weight1[layer][k,j] += epsilon
            loss1 = cross_entropy_loss(X, targets, new_weight1)
            new_weight2[layer][k,j] -= epsilon
            loss2 = cross_entropy_loss(X, targets, new_weight2)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
    
    maximum_abosulte_difference = abs(computed_gradient-dw).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)

def gradient_descent(X, targets, w, delta, learning_rate, should_check_gradient, dw_k_prev, dw_j_prev):
    j = 0
    k = 1
    
    normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
    
    a_hidden = forward_hidden(X, w[j])
    outputs = forward(a_hidden, w[k])
    
    delta[k] = - (targets - outputs)
#    delta[j] = np.multiply( (delta[k].dot(w[k])), (a_hidden*(1 - a_hidden)) )
    z = X.dot(w[j].T)
    d_sig = 1.1439 / ((np.cosh((2/3)*z))**2) 
    delta[j] = np.multiply( (delta[k].dot(w[k])), d_sig )

    dw_k = delta[k].T.dot(a_hidden)
    dw_k = dw_k / normalization_factor # Normalize gradient equally as loss normalization
    assert dw_k.shape == w[k].shape, "dw shape was: {}. Expected: {}".format(dw_k.shape, w[k].shape)
    
    dw_j = delta[j].T.dot(X)
    dw_j = dw_j / normalization_factor # Normalize gradient equally as loss normalization
    assert dw_j.shape == w[j].shape, "dw shape was: {}. Expected: {}".format(dw_j.shape, w[j].shape)

    if should_check_gradient:
        check_gradient(X, targets, w, 1e-2, dw_k, k)
        check_gradient(X, targets, w, 1e-2, dw_j, j)
        
    vk = (learning_rate * dw_k + 0.9*dw_k_prev)
    vj = (learning_rate * dw_j + 0.9*dw_j_prev)
    w[k] = w[k] - vk
    w[j] = w[j] - vj
        
#    vk = learning_rate * dw_k
#    vj = learning_rate * dw_j
#    w[k] = w[k] - vk
#    w[j] = w[j] - vj
    
    return w, vk, vj


X_train, Y_train, X_test, Y_test = mnist.load()

    #subset for development
#X_train = X_train[0:2000, :];
#Y_train = Y_train[0:2000];
#X_test = X_test[0:200, :];
#Y_test = Y_test[0:200];

# Pre-process data
X_train, X_test = (X_train / 127.5) - 1, (X_test / 127.5) - 1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


# Hyperparameters

hidde_layer_size = 64
batch_size = 32
learning_rate = 0.01
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 32

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []

def train_loop(X_train, Y_train):
    w = []
#    w.append(np.random.uniform(-1,1,size=(hidde_layer_size,X_train.shape[1]))) #append hidden layer weights  
#    w.append(np.random.uniform(-1,1,size=(Y_train.shape[1], hidde_layer_size))) #append output layer weights
    std = 1/np.sqrt(784)
    w.append(np.random.normal(0, std, size=(hidde_layer_size, X_train.shape[1]))) #append hidden layer weights 
    std = 1/np.sqrt(64)
    w.append(np.random.normal(0, std, size=(Y_train.shape[1], hidde_layer_size))) #append output layer weights
    
    delta = []
    delta.append(np.zeros((hidde_layer_size, 1)))
    delta.append(np.zeros((Y_train.shape[1], 1)))
    dw_k_prev = np.zeros((w[1].shape))
    dw_j_prev = np.zeros((w[0].shape))
    for e in range(max_epochs): # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            w, dw_k_prev, dw_j_prev = gradient_descent(X_batch, Y_batch, w, delta, learning_rate, should_gradient_check, dw_k_prev, dw_j_prev)
            #print(cross_entropy_loss(X_batch, Y_batch, w))
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w))
                

                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w
        #shuffle data between epochs
        X_train, Y_train = shuffle_data(X_train, Y_train)
    return w

start = time.time()
w = train_loop(X_train, Y_train)
end = time.time()
print("Training time was: {}".format((end - start)))

train_loss = TRAIN_LOSS
val_loss = VAL_LOSS
test_loss = TEST_LOSS

train_acc = TRAIN_ACC
val_acc = VAL_ACC
test_acc = TEST_ACC

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.title("Cross-Entropy Loss Function")
plt.xlabel('Iterations');
plt.ylabel('error');
plt.ylim([0, 0.2])
plt.show()

#plt.clf()
plt.figure()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim([0.5, 1.0])
plt.title("Percentage of right classification")
plt.xlabel('Iterations');
plt.ylabel('%');
plt.legend()
plt.show()






