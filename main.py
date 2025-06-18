import json
import tensorflow as tf
import tqdm
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import create_pinn_model
from losses import physics_loss
from tune import objective
from plot import plot_training_and_predictions

# 1. Hyperparameter Tuning
print("\n[1] Running Hyperparameter Tuning with Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

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

# Load your training and test data
# from data import X_train, y_train, X_test, y_test  # Uncomment when available
X_train = np.random.rand(100, 3).astype(np.float32)
y_train = np.random.rand(100, 1).astype(np.float32)
X_test = np.random.rand(20, 3).astype(np.float32)
y_test = tf.convert_to_tensor(np.random.rand(20, 1).astype(np.float32))

# Model setup
best_model = create_pinn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=best_lr)
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
        physics_loss_ = physics_loss(best_model, inputs, delta_=best_delta)
        total_loss = data_loss + best_weight * physics_loss_

        p_loss.append(physics_loss_)
        d_loss.append(data_loss)
        t_loss.append(total_loss)

    gradients = tape.gradient(total_loss, best_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, best_model.trainable_variables))

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss.numpy():.4f}, "
              f"Data = {data_loss.numpy():.4f}, Physics = {physics_loss_.numpy():.4f}")

print("\n[3] Training Complete. Best model is ready.")

# 3. Plot Results
print("\n[4] Plotting Results...")
plot_training_and_predictions(best_model, X_test, y_test, p_loss, d_loss, t_loss, best_weight)

# Saving model
best_model.save('pinn_model.keras') 