import tensorflow as tf

class EmbeddingRegression(tf.keras.Model):
  def __init__(self, discrete_vars):
    super().__init__()
    self.embeddings = [tf.keras.layers.Embedding(len(items), 1, input_length=1) for items in discrete_vars]
    self.reg_layer = tf.keras.layers.Dense(1)
    self.num_discrete_vars = len(discrete_vars)

  def call(self, x):
    continuous, discrete = x
    embeddeds = []
    for i in range(self.num_discrete_vars):
      embeddeds.append(self.embeddings[i](discrete[:, i]))
    return self.reg_layer(tf.concat([continuous] + embeddeds, axis=-1))