from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_dataset(file_path):
    mileage = []
    price = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            mileage.append(float(row[0]))
            price.append(float(row[1]))
    return mileage, price

mileage, price = load_dataset('data.csv')
mileage = np.array(mileage).reshape(-1, 1)
price = np.array(price)

model = LinearRegression()
model.fit(mileage, price)

theta0_sklearn = model.intercept_
theta1_sklearn = model.coef_[0]

mileage_to_predict = np.array([[50000]])
predicted_price_sklearn = model.predict(mileage_to_predict)

plt.scatter(mileage, price, color='blue', label='Actual Data')
plt.plot(mileage, model.predict(mileage), color='red', label='Linear Regression (sklearn)')
plt.scatter(mileage_to_predict, predicted_price_sklearn, color='green', label='Predicted Value', marker='o', s=200)
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Linear Regression using scikit-learn')
plt.legend()
plt.show()

print(f'Theta0 (sklearn): {theta0_sklearn}')
print(f'Theta1 (sklearn): {theta1_sklearn}')
print(f'Predicted Price (sklearn) for Mileage 50000: {predicted_price_sklearn[0]}')

