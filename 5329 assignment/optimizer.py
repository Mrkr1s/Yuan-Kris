import numpy as np

def update_model_adam(model, lr, beta1, beta2, epsilon, t, adam_state):
    layers = model.get_layers()
    for layer_type, layer in layers:
        if layer_type == "linear":
            for param_name, param, grad in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
                key = (id(layer), param_name)
                if key not in adam_state:
                    adam_state[key] = {"m": 0, "v": 0}
                adam_state[key]["m"] = beta1 * adam_state[key]["m"] + (1 - beta1) * grad
                adam_state[key]["v"] = beta2 * adam_state[key]["v"] + (1 - beta2) * (grad ** 2)
                m_hat = adam_state[key]["m"] / (1 - beta1 ** t)
                v_hat = adam_state[key]["v"] / (1 - beta2 ** t)
                update_val = lr * m_hat / (np.sqrt(v_hat) + epsilon)
                if param_name == "W":
                    layer.W -= update_val
                else:
                    layer.b -= update_val
        elif layer_type == "batchnorm":
            for param_name, param, grad in [("gamma", layer.gamma, layer.dgamma), ("beta", layer.beta, layer.dbeta)]:
                key = (id(layer), param_name)
                if key not in adam_state:
                    adam_state[key] = {"m": 0, "v": 0}
                adam_state[key]["m"] = beta1 * adam_state[key]["m"] + (1 - beta1) * grad
                adam_state[key]["v"] = beta2 * adam_state[key]["v"] + (1 - beta2) * (grad ** 2)
                m_hat = adam_state[key]["m"] / (1 - beta1 ** t)
                v_hat = adam_state[key]["v"] / (1 - beta2 ** t)
                update_val = lr * m_hat / (np.sqrt(v_hat) + epsilon)
                if param_name == "gamma":
                    layer.gamma -= update_val
                else:
                    layer.beta -= update_val

def update_model_sgd_momentum(model, lr, momentum, velocity):

    layers = model.get_layers()
    for layer_type, layer in layers:
        if layer_type == 'linear':
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                key = f"W_{id(layer)}"
                if key not in velocity:
                    velocity[key] = np.zeros_like(layer.W)
                velocity[key] = momentum * velocity[key] - lr * layer.dW
                layer.W += velocity[key]
            if hasattr(layer, 'b') and hasattr(layer, 'db'):
                key = f"b_{id(layer)}"
                if key not in velocity:
                    velocity[key] = np.zeros_like(layer.b)
                velocity[key] = momentum * velocity[key] - lr * layer.db
                layer.b += velocity[key]
        elif layer_type == 'batchnorm':
            if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                key = f"gamma_{id(layer)}"
                if key not in velocity:
                    velocity[key] = np.zeros_like(layer.gamma)
                velocity[key] = momentum * velocity[key] - lr * layer.dgamma
                layer.gamma += velocity[key]
            if hasattr(layer, 'beta') and hasattr(layer, 'dbeta'):
                key = f"beta_{id(layer)}"
                if key not in velocity:
                    velocity[key] = np.zeros_like(layer.beta)
                velocity[key] = momentum * velocity[key] - lr * layer.dbeta
                layer.beta += velocity[key]