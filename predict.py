import csv
import matplotlib.pyplot as plt

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def plot_scatter_and_prediction(mileage, price, theta0, theta1, predicted_mileage, predicted_price):
    plt.scatter(mileage, price, color='blue', label='Actual Data')
    plt.scatter(predicted_mileage, predicted_price, color='red', label='Predicted Value', marker='o', s=200)
    plt.plot(mileage, [estimate_price(x, theta0, theta1) for x in mileage], color='green', label='Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Scatter Plot with Predicted Value')
    plt.legend()
    plt.show()

def main():
    mileage = []
    price = []

    with open("data.csv", 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            mileage.append(float(row[0]))
            price.append(float(row[1]))

    with open("trained_vars.txt", 'r') as file:
        lines = file.readlines()
        theta0 = float(lines[0].strip())
        theta1 = float(lines[1].strip())

#    mileage_input = float(input("Please provide a mileage: "))
    mileage_input = 50000
    estimated_price = estimate_price(mileage_input, theta0, theta1)
    print(f"With a mileage of {mileage_input} your car is estimated to be worth ${estimated_price:.2f}.")

    # Plot the scatter graph with the predicted value
    plot_scatter_and_prediction(mileage, price, theta0, theta1, mileage_input, estimated_price)

if __name__ == '__main__':
    main()

