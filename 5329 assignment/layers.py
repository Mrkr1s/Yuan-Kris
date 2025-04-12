import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weight_decay=0.0):
        #fully connected layer
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        self.weight_decay = weight_decay
        self.x = None      #input to the layer
        self.dW = None     #gradient for weights
        self.db = None     #gradient for biases

    def forward(self, x):
        self.x = x
        #output = x * W + b
        return x.dot(self.W) + self.b

    def backward(self, dout):
        self.dW = self.x.T.dot(dout) + self.weight_decay * self.W
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout.dot(self.W.T)
        return dx

class BatchNorm:
    def __init__(self, D, momentum=0.9, eps=1e-5):
        #batchnorm
        self.gamma = np.ones((1, D))
        self.beta = np.zeros((1, D))
        self.momentum = momentum 
        self.eps = eps         
        self.running_mean = np.zeros((1, D))
        self.running_var = np.ones((1, D))
        self.cache = None     
        self.mode = 'train'  

    def forward(self, x, mode='train'):
        self.mode = mode
        if mode == 'train':
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            #normalize the input
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma * x_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            #save cache for backward pass
            self.cache = (x, x_norm, mu, var)
        else:
            #use running mean and variance
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        x, x_norm, mu, var = self.cache
        N, D = x.shape
        #compute the inverse standard deviation
        std_inv = 1.0 / np.sqrt(var + self.eps)
        #gradient of the normalized output
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmu = np.sum(dx_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mu), axis=0, keepdims=True)
        dx = dx_norm * std_inv + dvar * 2 * (x - mu) / N + dmu / N
        self.dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        return dx

class Dropout:
    def __init__(self, p=0.5):
        self.p = p         #drop probability
        self.mask = None   #mask to zero out some activations
        self.mode = 'train'

    def forward(self, x, mode='train'):
        self.mode = mode
        if mode == 'train':
            # 1/(1-p) where the neuron is kept and 0 where dropped
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        if self.mode == 'train':
            return dout * self.mask
        else:
            return dout

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha 
        self.x = None

    def forward(self, x):
        self.x = x  #save input for backprop
        #output x if positive else alpha * x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        dx = np.where(self.x > 0, 1, self.alpha)
        return dout * dx