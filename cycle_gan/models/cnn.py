import tensorflow as tf

def get_encoder(num_latent_dim, final_activation=None):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(16, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_latent_dim, activation=final_activation)
        ])
