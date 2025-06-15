from model import train_one_epoch

if __name__ == "__main__":
    x = list(range(-10, 10))
    y_true = [-46.93, -41.98, -36.97, -31.99, -26.94, -21.94, -16.98, -11.91, -6.91, -1.91, 3.01, 8.08, 13.05, 18.03, 23.01, 28.01, 33.01, 38.07, 43.07, 48.07]
    learning_rate = 0.001
    weight_m = 1
    weight_b = 0
    epochs = 5000
    print(f"Starting model(x) = {weight_b} + {weight_m} * x")
    for i in range(epochs):
        weight_b, weight_m = train_one_epoch(i, x, y_true, weight_b, weight_m, learning_rate)
    print(f"Final model(x) = {weight_b} + {weight_m} * x")