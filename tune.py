import tensorflow as tf
import optuna
from model import create_pinn_model
from losses import physics_loss

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight = trial.suggest_float("weight", 1.0, 10.0)
    delta = trial.suggest_float("delta", 0.1, 2.0)

    tf.random.set_seed(21)
    pinn_model = create_pinn_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    mse_loss = tf.keras.losses.MeanSquaredError()

    epochs = 300
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
            targets = tf.convert_to_tensor(y_train, dtype=tf.float32)
            pred_K = pinn_model(inputs)

            data_loss = mse_loss(targets, pred_K)
            physics_loss_ = physics_loss(pinn_model, inputs, delta_=delta)
            total_loss = data_loss + weight * physics_loss_

        gradients = tape.gradient(total_loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))

    return total_loss.numpy()


