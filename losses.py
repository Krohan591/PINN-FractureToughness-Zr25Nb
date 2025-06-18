import tensorflow as tf

def physics_loss(model, inputs, delta_=0.5):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        pred_K = model(inputs)

        # Extract individual input features
        C_H, HCC, T = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    # Compute gradients
    dK_dInputs = tape.gradient(pred_K, inputs)
    dK_dT = dK_dInputs[:, 2:3]  # Gradient w.r.t. T
    dk_dT = tf.sigmoid(dK_dT)

    # Compute expected derivatives from empirical laws
    sigmoid_dK_dT = (model.A / model.C) * tf.exp(-(T - model.B) / model.C) / (1 + tf.exp(-(T - model.B) / model.C))**2
    sigmoid_dk_dT = tf.sigmoid(sigmoid_dK_dT)

    # Compute physics loss
    sigmoid_loss = tf.keras.losses.Huber(delta=delta_)(sigmoid_dK_dT, dK_dT)

    del tape
    return sigmoid_loss
