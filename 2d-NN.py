import numpy as np
import time

#variables
n_hidden = 10
n_in = 10
#outputs
n_out = 10
#sample data
n_samples = 300
n_iterations = 100


#hyperparameters
learning_rate = 0.01
momentum = 0.9

#non deterministic seeding
np.random.seed(0)

#activation function 1
def sigmoid(x):
    #turns numbers in probabilities
    return 1.0/(1.0 + np.exp(-x))

#activation function 2 (layer 2)- to improve loss
def tanh_prime(x):
        return 1 - np.tanh(x)**2

#input data, transpose, layer 1, layer 2, bias for layer 1, bias for layer 2
def train(x, t, V, W, bv, bw):

    #forward propagation -- matrix multiplication + biases
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    #backword propagation - Errors
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    #predict out loss - deltas
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    #cross entropy
    loss = -np.mean( t * np.log(Y) + (1 - t) * np.log(1-Y))

    # Note that we use error for each layer as a gradient
    # for biases

    return loss, (dV, dW, Ev, Ew)

def predict(x ,V ,W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Setup initial parameters
# Note that initialization is cruxial for first-order methods!

#create layers
V = np.random.normal(scale=0.1 , size=(n_in,n_hidden))
W = np.random.normal(scale=0.1 , size=(n_hidden,n_out))

#set up biases
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

#generate data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^1

#TRAINING TIME
for epoch in range(n_iterations):
        err = [] #[9] increases error
        upd = [0]*len(params)

        t0 = time.clock()
        #for each datapoint we want to update out weight of the network
        for i in range(X.shape[0]):
            loss,grad = train(X[i],T[i], *params)

            #update loss
            for j in range(len(params)):
                params[j] -= upd[j]

            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]

            err.append( loss )

        print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(
            epoch, np.mean( err ), time.clock()-t0))


#try to predict something
x = np.random.binomial(1, 0.5, n_in)
print "XOR prediction: "
print x
print predict(x,*params)
