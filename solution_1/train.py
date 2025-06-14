from model import model, loss, loss_gradient_b, loss_gradient_m

if __name__ == "__main__":
    x = list(range(-10, 10))
    y_true = [-46.93, -41.98, -36.97, -31.99, -26.94, -21.94, -16.98, -11.91, -6.91, -1.91, 3.01, 8.08, 13.05, 18.03, 23.01, 28.01, 33.01, 38.07, 43.07, 48.07]
    learning_rate = 0.001
    weight_m = 1
    weight_b = 0
    epochs = 5000
    for i in range(epochs):
        y_pred = model(x, weight_b, weight_m)
        l = loss(y_pred, y_true)
        grad_b = loss_gradient_b(x, y_pred, y_true)
        grad_m = loss_gradient_m(x, y_pred, y_true)
        weight_m = weight_m - learning_rate * grad_m
        weight_b = weight_b - learning_rate * grad_b
        print(f"epoch: {i}, loss: {l}, gradient: {grad_b}, {grad_m}, model(x): {weight_b} + {weight_m} * x")