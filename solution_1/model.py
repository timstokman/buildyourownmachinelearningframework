def model(x: list[float], weight_b: float, weight_m: float) -> list[float]:
    """
    Run the linear regression model for the given input and weights

    >>> model([], 1.0, 2.0)
    []
    >>> model([2, 3], 1.0, 2.0)
    [5.0, 7.0]
    >>> model([-3, -6, -9], -5.0, -2.0)
    [1.0, 7.0, 13.0]
    """
    all_y = []
    for x_i in x:
        y = weight_b + weight_m * x_i
        all_y.append(y)
    return all_y

def loss(y_true: float, y_pred: float) -> float:
    """
    Calculate the loss function for the linear regression model

    >>> loss([], [])
    0.0
    >>> round(loss([5, 7], [-1, 4]), 0)
    45.0
    >>> round(loss([1, 2], [-5, 4]), 0)
    40.0
    """
    loss = 0.0
    for i in range(len(y_true)):
        l_i = (y_true[i] - y_pred[i]) ** 2
        loss += l_i
    return loss

def loss_gradient_m(x: list[float], y_pred: list[float], y_true: list[float]) -> float:
    """
    Calculate the gradients for the loss function for the weight m

    >>> loss_gradient_m([], [], [])
    0.0
    >>> round(loss_gradient_m([5, 7], [-1, 4], [5, 4]), 0)
    -30.0
    >>> round(loss_gradient_m([-3, 4, 4], [-2, 0, 4], [1, 4, 0]), 0)
    6.0
    """
    gradients = 0.0
    n = len(y_pred)
    for i in range(n):
        gradients += (-2.0 / n) * x[i] * (y_true[i] - y_pred[i])
    return gradients

def loss_gradient_b(x: list[float], y_pred: list[float], y_true: list[float]) -> float:
    """
    Calculate the gradients for the loss function for the weight b

    >>> round(loss_gradient_b([], [], []), 0)
    0.0
    >>> round(loss_gradient_b([5, 7], [-1, 4], [5, 4]), 0)
    -6.0
    >>> round(loss_gradient_b([-3, 4, 4], [-2, 0, 4], [1, 4, 0]), 0)
    -2.0
    """
    gradients = 0.0
    n = len(y_pred)
    for i in range(n):
        gradients += (-2.0 / n) * (y_true[i] - y_pred[i])
    return gradients