import tensorflow as tf

class CycleGAN(tf.keras.models.Model):
    def __init__(self, a_to_b, b_to_a, a_disc, b_disc):
        self.a_to_b = a_to_b()
        self.b_to_a = b_to_a()
        self.a_disc = a_disc()
        self.b_disc = b_disc()

        self.a_to_b_opt = tf.keras.optimizers.Adam()
        self.b_to_a_opt = tf.keras.optimizers.Adam()
        self.a_disc_opt = tf.keras.optimizers.Adam()
        self.b_disc_opt = tf.keras.optimizers.Adam()

        self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


    @tf.function
    def same_image_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true-y_pred))

    @tf.function
    def generator_loss(self, disc_fake):
        return self.binary_cross_entropy(tf.zeros_like(disc_fake), disc_fake)

    @tf.function
    def discriminator_loss(self, disc_real, disc_fake):
        return self.binary_cross_entropy(tf.zeros_like(disc_real), disc_real) + self.binary_cross_entropy(tf.ones_like(disc_fake), disc_fake)

    @tf.function
    def train_step(self, real_a, real_b):
        with tf.GradientTape(persistent=True) as tape:
            fake_a = self.b_to_a(real_b)
            fake_b = self.a_to_b(real_a)
            cycled_a = self.b_to_a(fake_b)
            cycled_b = self.a_to_b(fake_a)
            same_a = self.b_to_a(real_a)
            same_b = self.a_to_b(real_b)

            disc_real_a = self.a_disc(real_a)
            disc_real_b = self.b_disc(real_b)
            disc_fake_a = self.a_disc(fake_a)
            disc_fake_b = self.b_disc(fake_b)

            cycle_loss = self.same_image_loss(cycled_a, real_a) + self.same_image_loss(cycled_b, real_b)
            a_to_b_loss = self.generator_loss(disc_fake_b) + self.same_image_loss(real_a, same_a) + cycle_loss
            b_to_a_loss = self.generator_loss(disc_fake_a) + self.same_image_loss(real_b, same_b) + cycle_loss

            disc_a_loss = self.discriminator_loss(disc_real_a, disc_fake_a)
            disc_b_loss = self.discriminator_loss(disc_real_b, disc_fake_b)

        a_to_b_gradients = tape.gradient(a_to_b_loss, self.a_to_b.trainable_variables)
        b_to_a_gradients = tape.gradient(b_to_a_loss, self.b_to_a.trainable_variables)
        disc_a_gradients = tape.gradient(disc_a_loss, self.disc_a.trainable_variables)
        disc_b_gradients = tape.gradient(disc_b_loss, self.disc_b.trainable_variables)

        self.a_to_b_opt.apply_gradient(zip(a_to_b_gradients, self.a_to_b.trainable_variables))
        self.b_to_a_opt.apply_gradient(zip(b_to_a_gradients, self.b_to_a.trainable_variables))
        self.disc_a_opt.apply_gradient(zip(disc_a_gradients, self.disc_a.trainable_variables))
        self.disc_b_opt.apply_gradient(zip(disc_b_gradients, self.disc_b.trainable_variables))

        return a_to_b_loss, b_to_a_loss, disc_a_loss, disc_b_loss

    def fit(self, a_ds, b_ds, epochs=1):
        for epoch in range(epochs):
            for real_a, real_b in zip(a_ds, b_ds):
                    losses = self.train_step(real_a, real_b)
                    print(losses)

                        


