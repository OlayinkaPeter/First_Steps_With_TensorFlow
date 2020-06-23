import tensorflow as tf
import numpy as np

radius_values         = np.array([2.0, 4.0, 8.0, 7.0, 6.0, 5.0, 1.0, 11.0, 3.0, 5.0, 4.0, 2.0],  dtype=float)
circumference_values  = np.array([12.57, 25.13, 50.27, 43.98, 37.70, 31.42, 6.28, 69.12, 18.85, 31.42, 25.13, 12.57],  dtype=float)

for i, r in enumerate(radius_values):
  print("Given radius to be = {}, the Circumference = {}".format(r, circumference_values[i]))
  
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(radius_values, circumference_values, epochs=500, verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([10.0]))
