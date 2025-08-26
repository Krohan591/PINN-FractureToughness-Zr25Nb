import tensorflow as tf
import optuna
from model import PINN
from losses import physics_loss

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight = trial.suggest_float("weight", 0.01, 10.0)
    delta = trial.suggest_float("delta", 0.01, 0.1)
    A = trial.suggest_float("A", 0, 10)
    B = trial.suggest_float("B", 0, 10)
    C = trial.suggest_float("C", -2.0, 2.0)
    
    # Model and optimizer
    pinn_model = PINN(A, B, C)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    huber_loss = tf.keras.losses.Huber(delta=delta)
    mse_loss = tf.keras.losses.MeanSquaredError()

    # Convert data to tensors once
    inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
    targets = tf.convert_to_tensor(y_train, dtype=tf.float32)
    
    # Early stopping if NaN is detected
    epochs = 300
    try:
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                pred_K = pinn_model(inputs)
                data_loss = mse_loss(targets, pred_K)
                physics_loss_ = physics_loss(pinn_model, inputs, delta_=delta)
                total_loss = data_loss + weight * physics_loss_
                
                # Check for NaN in loss
                if tf.math.is_nan(total_loss):
                    return float('inf')  # Penalize trial with NaN loss
                
            all_vars_to_train = pinn_model.trainable_variables + [pinn_model.A, pinn_model.B, pinn_model.C]
            gradients = tape.gradient(total_loss, all_vars_to_train)
            
            # Check for NaN in gradients
            if any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None):
                return float('inf')  # Penalize trial with NaN gradients
                
            optimizer.apply_gradients(zip(gradients, all_vars_to_train))

        # Evaluate R² score
        X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_pred = pinn_model(X_test_tensor).numpy()
        
        # Check for NaN in predictions
        if np.isnan(y_pred).any():
            return float('inf')  # Penalize trial with NaN predictions
            
        r2 = r2_score(y_test, y_pred)
        trial.set_user_attr("r2_score", r2)  # Store R² for inspection

        # Combine objective: minimize total loss + (penalty for low R²)
        combined_score = total_loss.numpy() + 1.0 * (1 - r2)
        return combined_score
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')  # Penalize failed trials


