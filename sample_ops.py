import tensorflow as tf

a = tf.constant([[2.0, 3.0], [1.0, 2.0]])
b = tf.constant([[1.0, 4.0], [2.0, 0.0]])

# tf.matmul performs matrix multiplication
c = tf.matmul(a, b)

print(c)
