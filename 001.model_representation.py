import numpy 
import matplotlib.pyplot as pyplot
from typing import List



def model_function(x: float, weight: float, bias: float) -> float:
    return weight * x + bias

def compute_model_output(input_data:List[float], weight: float, bias: float) -> List[float]:
    return [weight * input_data_point + bias for input_data_point in input_data]

if __name__ == "__main__":
    x_train = numpy.array([1.0, 2.0])
    y_train = numpy.array([300.0, 500.0])
    print(f"x_train = {x_train}")
    print(f"y_train = {y_train}")

    print(f"x_train_shape = {x_train.shape[0]}")
    print(f"y_train_shape = {y_train.shape[0]}")

    print(f"len_x_train = {len(x_train)}")
    print(f"len_y_train = {len(y_train)}")

    print("Plotting the data")

    pyplot.scatter(x_train, y_train, marker="x", c="b")
    pyplot.title("Housing prices")
    pyplot.ylabel("Price (in 1000s of dollars)")
    pyplot.xlabel("Size (1000 sqft)")
    pyplot.show()

    """
    Now we want to model the function f_wb(x) = wx + b, we can assume the values of 100 for both w and b
    """
    weight = 100
    bias = 100
    model_output = compute_model_output(x_train, weight, bias)

    """
    Now let's plot both the training data and our model outputs
    """
    pyplot.scatter(x_train, y_train, marker="x", c="b")
    pyplot.plot(x_train, model_output, c="r", label="Our prediction")
    pyplot.title("Housing prices")
    pyplot.ylabel("Price (in 1000s of dollars)")
    pyplot.xlabel("Size (1000 sqft)")
    pyplot.legend()
    pyplot.show()

    """
    Now let's do some prediction with our model with weights and biase 200 and 100 respectively. 
    """
    price = model_function(1.2, 200, 100)
    print(f"The price for a house of size 1200sqft is {price}")
