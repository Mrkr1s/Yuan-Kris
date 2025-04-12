from layers import Linear, BatchNorm, Dropout, LeakyReLU
from loss import SoftmaxCrossEntropyLoss

class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_p=0.3, weight_decay=0.01):

        self.layers = []
        dims = [input_dim] + hidden_dims + [num_classes]
        self.num_layers = len(dims) - 1
        
        for i in range(self.num_layers):
            linear_layer = Linear(dims[i], dims[i+1], weight_decay=weight_decay)
            self.layers.append(("linear", linear_layer))
            if i < self.num_layers - 1:  # 隐藏层部分
                bn_layer = BatchNorm(dims[i+1])
                self.layers.append(("batchnorm", bn_layer))
                activation_layer = LeakyReLU(alpha=0.01)
                self.layers.append(("leakyrelu", activation_layer))
                dropout_layer = Dropout(p=dropout_p)
                self.layers.append(("dropout", dropout_layer))
        self.loss_layer = SoftmaxCrossEntropyLoss()
    
    def forward(self, x, mode='train'):
        out = x
        for (layer_type, layer) in self.layers:
            if layer_type in ['dropout', 'batchnorm']:
                out = layer.forward(out, mode)
            else:
                out = layer.forward(out)
        return out
    
    def backward(self, dout):
        for layer_type, layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def get_layers(self):
        return self.layers