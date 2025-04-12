import optuna
import numpy as np
import time
from data_utils import load_data, normalize_data, get_minibatches
from model import MLP
from optimizer import update_model_adam  #Adam optimizer

def objective(trial):
    #hidden layers between 2 and 4.
    hidden_depth = trial.suggest_int("hidden_depth", 2, 4)
    hidden_sizes = []
    for i in range(hidden_depth):
        # 256, 512, 1024 for each hidden layer
        hidden_size = trial.suggest_categorical(f"hidden_size_{i}", [256, 512, 1024])
        hidden_sizes.append(hidden_size)
    
    #dropout
    dropout_p = trial.suggest_float("dropout_p", 0.2, 0.6)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 5e-2)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    #Adam, use the common fixed defaults:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = normalize_data(X_train, X_test)
    split_idx = int(0.9 * X_train.shape[0])
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    model = MLP(input_dim=X_tr.shape[1],
                hidden_dims=hidden_sizes,
                num_classes=10,
                dropout_p=dropout_p,
                weight_decay=weight_decay)
    
    epochs = 20
    batch_size = 128
    adam_state = {}  
    t = 1 

    for epoch in range(epochs):
        for X_batch, y_batch in get_minibatches(X_tr, y_tr, batch_size):
            #forward pass
            logits = model.forward(X_batch, mode='train')
            loss = model.loss_layer.forward(logits, y_batch)
            #backward pass
            dout = model.loss_layer.backward()
            model.backward(dout)
            update_model_adam(model, lr, beta1, beta2, epsilon, t, adam_state)
            t += 1 

    #calculate accuracy and average loss
    def evaluate(model, X, y, batch_size=64):
        correct = 0
        total_loss = 0
        count = 0
        for X_batch, y_batch in get_minibatches(X, y, batch_size, shuffle=False):
            logits = model.forward(X_batch, mode='test')
            loss = model.loss_layer.forward(logits, y_batch)
            total_loss += loss * X_batch.shape[0]
            count += X_batch.shape[0]
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == y_batch.flatten())
        acc = correct / count
        avg_loss = total_loss / count
        return acc, avg_loss

    train_acc, train_loss = evaluate(model, X_tr, y_tr)
    val_acc, val_loss = evaluate(model, X_val, y_val)
    
    lambda_penalty = 0.1 
    balance_penalty = abs(train_loss - val_loss)
    composite_score = val_loss + lambda_penalty * balance_penalty

    return composite_score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Composite Score:", best_trial.value)
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")