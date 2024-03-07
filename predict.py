import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from math import sqrt


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def visualize(mileage, price, theta0, theta1, predicted_mileage, predicted_price):
    plt.scatter(mileage, price, color='blue', label='Actual Data')
    plt.scatter(predicted_mileage, predicted_price, color='red', label='Predicted Value', marker='o', s=200)
    plt.plot(mileage, [estimate_price(x, theta0, theta1) for x in mileage], color='green', label='Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Scatter Plot with Predicted Value')
    plt.legend()
    plt.show()


def predict(stage):
    # load datasets
    data = pd.read_csv("data.csv")
    model = pd.read_csv("model.csv", header=None)
    # get columns from datasets
    mileage = np.array(data['km'])
    price = np.array(data['price'])
    theta0 = model.iloc[0, 0]
    theta1 = model.iloc[0, 1]
    # USER INPUT STAGE
    if stage == 1:
        # input from user
        mileage_input = float(input("Please provide a mileage: "))
        # use trained model to estimate price
        estimated_price = estimate_price(mileage_input, theta0, theta1)
        print(f"With a mileage of {mileage_input} your car is estimated to be worth {estimated_price:.2f} moneys.")
        # Plot the scatter graph with the predicted value
        visualize(mileage, price, theta0, theta1, mileage_input, estimated_price)
    # ACCURACY EVALUATION STAGE
    elif stage == 2:
        predictions = np.array([estimate_price(value, theta0, theta1) for value in mileage])
        true_data = np.array(price)
        precision = r2_score(true_data, predictions)
        print(f'Precision: {precision:.2f}')
        
if __name__ == '__main__':
    try:
       predict(1)
    except FileNotFoundError:
        print("Error: model.csv or data.csv not found.")
        exit()
    except ValueError:
        print("Error: mileage input is not a number.")
        exit()
