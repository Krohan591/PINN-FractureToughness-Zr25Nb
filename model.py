# coding: utf-8
import tqdm as tqdm
from keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

@register_keras_serializable()

class PINN(tf.keras.Model):
    def __init__(self,A,B,C,**kwargs):
        super(PINN,self).__init__(**kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.initializer = tf.keras.initializers.HeNormal()
        self.activation_function = tf.keras.activations.relu
        self.hidden = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(256, activation=self.activation_function, kernel_initializer=self.initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, activation=self.activation_function, kernel_initializer=self.initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1,activation='sigmoid')  # Output: K (fracture toughness)
            ])
        
        # Trainable physics constants
        self.A = tf.Variable(self.A, dtype=tf.float32, trainable=True)
        self.B = tf.Variable(self.B, dtype=tf.float32, trainable=True)
        self.C = tf.Variable(self.C, dtype=tf.float32, trainable=True)

    def call(self, x):
        return self.hidden(x)
