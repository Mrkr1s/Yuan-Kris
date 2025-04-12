import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from data_utils import load_data, normalize_data, get_minibatches
from model import MLP
from optimizer import update_model_adam  #use Adam

def evaluate(model, X, y, batch_size=64, return_loss=False):
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
    return (acc, avg_loss) if return_loss else acc

def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8, patience=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    #memory for Adam
    adam_state = {}
    t = 1
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        # Start training each epoch
        for X_batch, y_batch in get_minibatches(X_train, y_train, batch_size):
            #forward pass in train mode
            logits = model.forward(X_batch, mode='train')
            loss = model.loss_layer.forward(logits, y_batch)
            epoch_loss += loss

            #backward pass
            dout = model.loss_layer.backward()
            model.backward(dout)
            
            #Adam optimizer
            update_model_adam(model, lr, beta1, beta2, epsilon, t, adam_state)
            t += 1  
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        train_acc, _ = evaluate(model, X_train, y_train, batch_size, return_loss=True)
        val_acc, val_loss = evaluate(model, X_val, y_val, batch_size, return_loss=True)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        #Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with val loss {best_val_loss:.4f}.")
            break

    if best_model_state is not None:
        model = best_model_state

    return train_losses, train_accs, val_losses, val_accs, best_epoch

def main():
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = normalize_data(X_train, X_test)
    
    #split training data into training and validation sets (90% train, 10% validation)
    split_idx = int(0.9 * X_train.shape[0])
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    

    model = MLP(input_dim=X_tr.shape[1],
                hidden_dims=[1024, 512],
                num_classes=10,
                dropout_p=0.30181191234013716,
                weight_decay=0.002547228012520423)

    start_time = time.time()
    train_losses, train_accs, val_losses, val_accs, best_epoch = train(
        model, X_tr, y_tr, X_val, y_val,
        epochs=50,
        batch_size=128,
        lr=0.0005,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        patience=5
    )
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Best epoch: {best_epoch}")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.show()

    #evaluate on the test set
    test_acc, test_loss = evaluate(model, X_test, y_test, batch_size=128, return_loss=True)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()