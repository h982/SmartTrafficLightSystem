
import keras
import keras.backend as K

import math


class PriorProbability(keras.initializers.Initializer):

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = K.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result
