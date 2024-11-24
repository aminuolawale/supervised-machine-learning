import numpy 
import matplotlib.pyplot as pyplot
from typing import List
from random import randint
from math import pow

class GradientDescent:
    TOLERANCE = 0.0001
    LEARNING_RATE = 0.01
    def __init__(self, x_train: List[float], y_train: List[float] , order:int)-> None:
        self.x_train = x_train
        self.y_train = y_train
        self.order = order
    
    
    @classmethod
    def _randfloat(cls) -> float:
        return randint(-200,200)/100000.0

    @classmethod    
    def origin_model(cls, x:float,order: int)-> float:
        return sum([cls._randfloat() * pow(x,i) for i in range(order) ])
    
    def do_gradient_descent(self, coeffs: List[float]) -> List[float]:
        new_coeffs = self.compute_coeffs(coeffs)
        diffs = [abs(a -b) for a,b in zip(new_coeffs, coeffs)]
        if not any([a > self.TOLERANCE for a in diffs]):
            return new_coeffs
        return self.do_gradient_descent(new_coeffs)
    
    def compute_coeffs(self, vals: List[float]) -> List[float]:
        result = []
        for index, val in enumerate(vals):
            ans = numpy.subtract(val, numpy.multiply(self.LEARNING_RATE, self.error_derivative(index, vals)))
            result.append(ans)
        return result

    def error_derivative(self, index: int, coeffs: List[float]) -> float:
        result = 0.0
        for x, y in zip(self.x_train, self.y_train):
            ans = numpy.multiply(self.entry_error(x, y, coeffs), pow(x, index))
            result = numpy.add(result, ans)
        return result/len(self.y_train)    
    
    def entry_error(self, x: float, y: float, coeffs: List[float]) -> float:
        return numpy.subtract(self.model_function(x, coeffs), y)
    
    def model_function(self, x:float, coeffs: List[float]) -> float:
        result = 0.0
        for index, coeff in enumerate(coeffs):
            ans = numpy.multiply(coeff , pow(x,index))
            result = numpy.add(ans, result)
        return result
    
    def execute(self):
        pyplot.scatter(self.x_train, self.y_train, marker="*", c = "r")
        params = self.do_gradient_descent([0 for _ in range(self.order)])
        y_output = numpy.array([self.model_function(x, params) for x in self.x_train])
        pyplot.plot(self.x_train, y_output)
        pyplot.show()


if __name__ == "__main__":
    order = 2
    x_train = numpy.array(list(range(-5, 5)))
    y_train = numpy.array([GradientDescent.origin_model(x, order+1) for x in x_train])
    gradient_descent = GradientDescent(x_train, y_train,order+1)
    gradient_descent.execute()

    