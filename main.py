import json
import tensorflow as tf
import tqdm
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import PINN
from losses import physics_loss
from tune import objective
from plot_models import plot_pinn_performance
# 1. Hyperparameter Tuning
print("\n[1] Running Hyperparameter Tuning with Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20,catch=(ValueError,))

# Extract and save best parameters
best_params = study.best_params
print("\nBest Hyperparameters:", best_params)
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

# 2. Final Training
print("\n[2] Starting Final Model Training with Best Hyperparameters...")
best_lr = best_params['lr']
best_weight = best_params['weight']
best_delta = best_params['delta']
best_A = study.best_params['A']
best_B = study.best_params['B']
best_C = study.best_params['C']


# Load your training and test data
# from data import X_train, y_train, X_test, y_test  # Uncomment when available
X_train = np.random.rand(100, 3).astype(np.float32)
y_train = np.random.rand(100, 1).astype(np.float32)
X_test = np.random.rand(20, 3).astype(np.float32)
y_test = tf.convert_to_tensor(np.random.rand(20, 1).astype(np.float32))

# Model setup
best_model = PINN(best_A, best_B, best_C)
optimizer = tf.keras.optimizers.Adam(learning_rate=best_lr)
huber_loss = tf.keras.losses.Huber(delta=best_delta)
mse_loss = tf.keras.losses.MeanSquaredError()

# Training loop
epochs = 300
p_loss, d_loss, t_loss = [], [], []

for epoch in tqdm.tqdm(range(epochs)):
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
        targets = tf.convert_to_tensor(y_train, dtype=tf.float32)
        pred_K = best_model(inputs)

        data_loss = mse_loss(targets, pred_K)
        physics_loss_ = physics_loss(best_model, inputs,best_delta)
        total_loss = data_loss + best_weight * physics_loss_
        p_loss.append(physics_loss_)
        d_loss.append(data_loss)
        t_loss.append(total_loss)
    all_vars_to_train = best_model.trainable_variables 
    gradients = tape.gradient(total_loss, all_vars_to_train)  
    optimizer.apply_gradients(zip(gradients, all_vars_to_train))

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss.numpy():.4f}, "
              f"Data = {data_loss.numpy():.4f}, Physics = {physics_loss_.numpy():.4f},\n"
              f"A = {best_model.A.numpy()}, B = {best_model.B.numpy()}, C = {best_model.C.numpy()}")

print("\n Training Complete. Best model is ready.")

# 3. Plot Results
print("\n[4] Plotting Results...")
y_pred_pinn = best_model.predict(X_test)
plot_pinn_performance(y_test, y_pred_pinn, p_loss, d_loss, t_loss,best_weight)

# Saving model
best_model.save('pinn_model.keras') 
