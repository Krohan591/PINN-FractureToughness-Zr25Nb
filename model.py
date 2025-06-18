import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, hidden_layers=None, dropout_rate=0.6, **kwargs):
        super(PINN, self).__init__(**kwargs)

        self.initializer = tf.keras.initializers.HeNormal()
        self.activation_function = tf.keras.activations.relu
        self.hidden = self.build_hidden_layers(hidden_layers, dropout_rate)

        # Trainable physics constants
        self.A = tf.Variable(18.0, dtype=tf.float32, trainable=True)
        self.B = tf.Variable(18.0, dtype=tf.float32, trainable=True)
        self.C = tf.Variable(2.0, dtype=tf.float32, trainable=True)

    def build_hidden_layers(self, hidden_layers, dropout_rate):
        if hidden_layers is None:
            hidden_layers = [128, 64]

        layers = []
        for units in hidden_layers:
            layers.append(tf.keras.layers.Dense(units, activation=self.activation_function, kernel_initializer=self.initializer))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer
        return tf.keras.Sequential(layers)

    def call(self, x):
        return self.hidden(x)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_pinn_model(hidden_layers=None, dropout_rate=0.6):
    return PINN(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
