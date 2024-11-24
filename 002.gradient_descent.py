import numpy
import matplotlib.pyplot  as pyplot
from random import randint
from typing import List, Tuple

"""
"""
class GradientDescent:

    def __init__(self, x_train: List[float], y_train: List[float] )-> None:
        self.x_train = x_train
        self.y_train = y_train
        self.tolerance = 0.001
        self.alpha = 0.001

    @classmethod
    def _origin_model(cls, x: float) -> float:
        return 0.58 * x + 7.6 + randint(-8, 17)
    

    def _error(self, w:float, b:float) -> float:
        return sum([ (w*x + b -y)**2 for x, y in zip(x_train,y_train)])/(2*len(y_train))
    
    def _error_derivative1(self, w:float, b:float) -> float:
        return sum([x* (w* x + b - y) for x, y in zip(x_train, y_train)]) /len(y_train)
    
    def _error_derivative2(self, w:float, b:float) -> float:
        return sum([(w*x +b -y)for x, y in zip(x_train, y_train)]) /len(y_train)

    def do_gradient_descent(self, w: float, b: float) -> Tuple[int, int]:
        new_w = w - self.alpha * self._error_derivative1(w, b)
        new_b  = b - self.alpha * self._error_derivative2(w, b)
        if abs(new_w -w) <= self.tolerance or abs(new_b-b) <= self.tolerance:
            return (new_w, new_b)
        return self.do_gradient_descent(new_w, new_b)
    
    def execute(self):
        pyplot.scatter(x_train, y_train, marker="x", c="r")

        w , b = self.do_gradient_descent(0, 0)
        y_output = numpy.array([w * x + b for x in x_train])
        pyplot.plot(x_train, y_output)
        pyplot.show()


if __name__ == "__main__":
    x_train = numpy.array(list(range(20)))
    y_train = numpy.array([GradientDescent._origin_model(x) for x in x_train])
    gradient_descent = GradientDescent(x_train, y_train)
    gradient_descent.execute()








