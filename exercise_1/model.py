"""
Implement a linear regression model, optimize it with gradient decent

Model: y_pred(x) = b + x * m
n: size of dataset
Quadratic loss: l(y_pred, y_true) = ∑ i = 1..n (y_pred_i - y_true_i) ^ 2
Gradients loss for m: dL / dm = -2 / n ∑ i = 1..n (x_i * (y_true_i- y_pred_i))
Weight update: weight = weight - learning_rate * gradient
"""

def model(x: list[float], weight_b: float, weight_m: float) -> list[float]:
    """
    Run the linear regression model for the given input and weights, predict values with the given weights

    >>> model([], 1.0, 2.0)
    []
    >>> model([2, 3], 1.0, 2.0)
    [5.0, 7.0]
    >>> model([-3, -6, -9], -5.0, -2.0)
    [1.0, 7.0, 13.0]
    """
    return []

def loss(y_true: float, y_pred: float) -> float:
    """
    Calculate the loss function for the linear regression model

    >>> loss([], [])
    0.0
    >>> loss([5, 7], [-1, 4])
    45.0
    >>> loss([1, 2], [-5, 4])
    40.0
    """
    return 0.0

def loss_gradient_m(x: list[float], y_pred: list[float], y_true: list[float]) -> float:
    """
    Calculate the gradients for the loss function for the weight m

    >>> loss_gradient_m([], [], [])
    0.0
    >>> loss_gradient_m([5, 7], [-1, 4], [5, 4])
    -30.0
    >>> loss_gradient_m([-3, 4, 4], [-2, 0, 4], [1, 4, 0])
    6.0
    """
    return 0.0

def loss_gradient_b(x: list[float], y_pred: list[float], y_true: list[float]) -> float:
    """
    Calculate the gradients for the loss function for the weight b

    >>> loss_gradient_b([], [], [])
    0.0
    >>> loss_gradient_b([5, 7], [-1, 4], [5, 4])
    -6.0
    >>> loss_gradient_b([-6, 4, 4], [-2, 0, 4, -16], [6, 4, 0, 8])
    -16.0
    """
    return 0.0

def train_one_epoch(epoch: int, x: list[float], y_true: list[float], weight_b: float, weight_m: float, learning_rate: float) -> tuple[float, float]:
    """
    Train one epoch, calculate the new set of weights

    >>> train_one_epoch(1, [], [], 5.0, -2.0, 1.0)
    epoch: 1, loss: 0.0, gradient: 0.0, 0.0, model(x): 5.0 + -2.0 * x
    (5.0, -2.0)
    >>> train_one_epoch(1, [1, 2], [-5, -7], 1.0, 2.0, 1.0)
    epoch: 1, loss: 208.0, gradient: 20.0, 32.0, model(x): -19.0 + -30.0 * x
    (-19.0, -30.0)
    >>> train_one_epoch(1, [-1, -2], [5, 7], 5.0, -2.0, 1.0)
    epoch: 1, loss: 8.0, gradient: 4.0, -6.0, model(x): 1.0 + 4.0 * x
    (1.0, 4.0)
    >>> train_one_epoch(1, [-1, -2], [5, 7], 5.0, -2.0, 2.0)
    epoch: 1, loss: 8.0, gradient: 4.0, -6.0, model(x): -3.0 + 10.0 * x
    (-3.0, 10.0)
    """
    # Run the model, and calculate the true labels
    y_pred = None
    # Calculate the loss
    l = None
    # Calculate the gradients
    grad_b = None
    grad_m = None
    # Calculate the new weights
    weight_m = None
    weight_b = None
    # Log the model run
    print(f"epoch: {epoch}, loss: {l}, gradient: {grad_b}, {grad_m}, model(x): {weight_b} + {weight_m} * x")
    return weight_b, weight_m