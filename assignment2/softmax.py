import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm


def should_early_stop(validation_loss, num_steps=3):
    """
    Returns true if the validation loss increases
    or stays the same for num_steps.
    --
    validation_loss: List of floats
    num_steps: integer
    """
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing) 


def train_val_split(X, Y, val_percentage):
    """
    Selects samples from the dataset randomly to be in the validation set.
    Also, shuffles the train set.
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
    """
    X: shape[batch_size, num_features(784)]
    -- 
    Returns [batch_size, num_features+1 ]
    """
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def check_gradient(X, targets, w, epsilon, computed_gradient):
    """
    Computes the numerical approximation for the gradient of w,
    w.r.t. the input X and target vector targets.
    Asserts that the computed_gradient from backpropagation is 
    correct w.r.t. the numerical approximation.
    --
    X: shape: [batch_size, num_features(784+1)]. Input batch of images
    targets: shape: [batch_size, num_classes]. Targets/label of images
    w: shape: [num_classes, num_features]. Weight from input->output
    epsilon: Epsilon for numerical approximation (See assignment)
    computed_gradient: Gradient computed from backpropagation. Same shape as w.
    """
    print("Checking gradient...")
    dw = np.zeros_like(w)
    for k in range(w.shape[0]):
        for j in range(w.shape[1]):
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            loss1 = cross_entropy_loss(X, targets, new_weight1)
            loss2 = cross_entropy_loss(X, targets, new_weight2)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
            
    computedmax = computed_gradient.max()
    dwmax = dw.max()
    maximum_abosulte_difference = abs(computed_gradient-dw).max()
    
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)


def softmax(a):
    """
    Applies the softmax activation function for the vector a.
    --
    a: shape: [batch_size, num_classes]. Activation of the output layer before activation
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)


def forward(X, w):
    """
    Performs a forward pass through the network
    --
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a = X.dot(w.T)
    return softmax(a)


def calculate_accuracy(X, targets, w):
    """
    Calculated the accuracy of the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()


def cross_entropy_loss(X, targets, w):
    """
    Computes the cross entropy loss given the input vector X and the target vector.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w)
    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()


def gradient_descent(X, targets, w, learning_rate, should_check_gradient):
    """
    Performs gradient descents for all weights in the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns updated w, with same shape
    """

    # Since we are taking the .mean() of our loss, we get the normalization factor to be 1/(N*C)
    # If you take loss.sum(), the normalization factor is 1.
    # The normalization factor is identical for all weights in the network (For multi-layer neural-networks as well.)
    normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
    outputs = forward(X, w)
    delta_k = - (targets - outputs)

    dw = delta_k.T.dot(X)
    dw = dw / normalization_factor  # Normalize gradient equally as we do with the loss
    assert dw.shape == w.shape, "dw shape was: {}. Expected: {}".format(dw.shape, w.shape)

    if should_check_gradient:
        check_gradient(X, targets, w, 1e-2,  dw)

    w = w - learning_rate * dw
    return w

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

    #subset for development
#X_train = X_train[0:2000, :];
#Y_train = Y_train[0:2000];
#X_test = X_test[0:200, :];
#Y_test = Y_test[0:200];


# Pre-process data
X_train, X_test = X_train / 255, X_test / 255
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


# Hyperparameters
batch_size = 64
learning_rate = 0.5
num_batches = X_train.shape[0] // batch_size
should_gradient_check = True
check_step = num_batches // 10
max_epochs = 20

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []


def train_loop():
    #w = np.zeros((Y_train.shape[1], X_train.shape[1]))
    w = np.random.uniform(-1,1,size=(Y_train.shape[1], X_train.shape[1]))
    for e in range(max_epochs):  # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            w = gradient_descent(X_batch,
                                 Y_batch,
                                 w,
                                 learning_rate,
                                 should_gradient_check)
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
    return w


w = train_loop()

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim([0, 0.05])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim([0.8, 1.0])
plt.legend()
plt.show()

plt.clf()
w = w[:, :-1]  # Remove bias
w = w.reshape(10, 28, 28)
w = np.concatenate(w, axis=0)
plt.imshow(w, cmap="gray")
plt.show()







