import random
import torch
from value import Value
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        print(self.w)
        self.b = Value(random.uniform(-1,1))
        print(self.b)

    def __call__(self, x):
        #w*x + b
        act = sum(wi * xi for wi,xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out
    

if __name__ == "__main__":
    x = [2.0, 3.0]
    n = Neuron(2)
    print(n(x))