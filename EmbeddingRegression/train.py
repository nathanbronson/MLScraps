from embeddingregression import EmbeddingRegression
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train(df, discrete_cols, y_col, TEST_SIZE=.2, return_data=False):
    y = df[y_col].values
    x_cols = filter(lambda e: e != y_col, df.columns)
    continuous_cols = filter(lambda e: e not in discrete_cols, x_cols)
    discrete_cols_with_dim = [list(set([str(n) for n in df[i].values.tolist()])) for i in discrete_cols]
    discrete_tokenize = lambda i: [g.index(i[n]) for n, g in enumerate(discrete_cols_with_dim)]
    emb_model = EmbeddingRegression(len(continuous_cols), discrete_cols_with_dim)
    emb_model.compile(optimizer="adam", loss="mse")
    x = (df[continuous_cols].values.astype(np.float32), np.array([discrete_tokenize(i) for i in df[discrete_cols].values.astype(str).tolist()], dtype=np.int32))
    tx0, vx0, tx1, vx1, ty, vy = train_test_split(x[0], x[1], y, test_size=TEST_SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices(((tx0, tx1), ty)).repeat(1000).batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices(((vx0, vx1), vy)).repeat(1000).batch(64)
    emb_model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    if return_data:
        return emb_model, train_dataset, val_dataset
    else:
        return emb_model