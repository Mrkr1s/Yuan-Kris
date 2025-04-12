import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y = y
        N = x.shape[0]
        correct_logprobs = -np.log(self.probs[range(N), y.flatten()])
        loss = np.sum(correct_logprobs) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        dx = self.probs.copy()
        dx[range(N), self.y.flatten()] -= 1
        dx /= N
        return dx