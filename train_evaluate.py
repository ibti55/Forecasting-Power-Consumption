import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Hyperparameters
input_dim = X_train.shape[2]
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 3
dropout = 0.2
learning_rate = 0.001
batch_size = 32

# Model initialization
model = VanillaTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

def train_model(model, X_train, y_train, X_val, y_val, num_epochs=20):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        model.eval()
        val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
        val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.float32))
        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    return train_losses, val_losses

train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)

# Evaluation
model.eval()
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test[:100, 0], label='Actual')
plt.plot(y_pred[:100, 0], label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Power Consumption (Zone 1)')
plt.show()