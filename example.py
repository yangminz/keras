from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

data = np.random.normal(size=(50, 20))
labels = np.random.randint(0, 2, (50))

inputs = Input(shape=(20,))
'''
- `keras/layers/core.py: class Dense(Layer): def __init__()`
    - Sequential Model & Functional API
        `keras/engine/topology.py: class Layer(object): def __init__()`
    - Functional API
        `keras/engine/topology.py: class Layer(object): def __call__()`
'''
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

'''
- `keras/engine/topology.py: class Container(Layer): def __init__()`
'''
model = Model(input=inputs, output=predictions)
'''
- `keras/engine/training.py: class Model(Container): def compile()`
'''
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
'''
- `keras/engine/training.py: class Model(Container): def fit()`
    - `keras/engine/training.py: class Model(Container): def _fit_loop()`
'''
model.fit(data, labels)  # starts training