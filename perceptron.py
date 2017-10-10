import random
from numpy import dot


class Perceptron:

    def __init__(self, number_of_inputs, weights=None, threshold=None):
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = random.uniform(-0.5, 0.5)
        self.weights = weights
        if self.weights is None:
            self.weights = [random.uniform(-0.5, 0.5) for i in range(number_of_inputs)]
        print("Input weights: " + self.weights.__str__())
        self.alpha = 0.1

    def activate(self, inputs, desired):
        x = dot(inputs, self.weights)
        error = desired - self.step(x-self.threshold)
        self.train(error, inputs)

    def step(self, value):
        return 0 if value < 0 else 1

    def train(self, error, inputs):
        for i, inp in enumerate(inputs):
            w = self.weights[i] + (self.alpha * inp * error)
            self.weights[i] = w

    def test(self, inputs):
        x = dot(inputs, self.weights)
        return self.step(x-self.threshold)


def print_out(data, perceptron):
    for item in data:
        print(item.__str__() + " => ", perceptron.test(item))


or_p = Perceptron(2, None, 1)
data = [[0, 0], [1, 0], [0, 1], [1, 1]]

print("OR-Perceptron:")
for i in range(100):
    or_p.activate(data[0], 0)
    or_p.activate(data[1], 1)
    or_p.activate(data[2], 1)
    or_p.activate(data[3], 1)

print_out(data, or_p)
print("Weights OR: " + or_p.weights.__str__())

print("\n\n")
and_p = Perceptron(2, [0.5, 0.5], 0.6)
print("AND-Perceptron:")

for i in range(100):
    and_p.activate(data[0], 0)
    and_p.activate(data[1], 0)
    and_p.activate(data[2], 0)
    and_p.activate(data[3], 1)

print_out(data, and_p)
print("Weights AND: " + and_p.weights.__str__())

