import pandas as pd
import numpy as np
from predict import estimate_price


def training(mileage, price, lr):
	m = len(price)
	theta0, theta1 = 0, 0

	tmp_theta0 = (lr / m) * sum((theta0 + theta1 * mileage - price))
	tmp_theta1 = (lr / m) * sum((theta0 + theta1 * mileage - price) * mileage)

	theta0 -= tmp_theta0
	theta1 -= tmp_theta1

	return theta0, theta1


def scale(mileage, mileage_norm, theta0, theta1):
	price_lr = estimate_price(mileage_norm, theta0, theta1)
	p1 = [mileage[0], mileage[23]]
	p2 = [price_lr[0], price_lr[23]]

	theta1 = (p2[1] - p2[0]) / (p1[1] - p1[0])
	theta0 = p2[0] - theta1 * p1[0]

	return theta0, theta1


if __name__ == '__main__':
	# load dataset
	data = pd.read_csv("data.csv")
	# get columns from dataset
	mileage = np.array(data['km'])
	price = np.array(data['price'])
	# normalize mileage
	mileage_norm = (mileage - np.mean(mileage)) / np.std(mileage)
	# learning rate and number of iterations
	lr = 1
	num_iter = 100
	# executing training function
	for i in range(num_iter):
		theta0, theta1 = training(mileage_norm, price, lr)
	# scaling model hyperparameters
	theta0, theta1 = scale(mileage, mileage_norm, theta0, theta1)
	print(theta0, theta1)
	# saving model hyperparameters for use in prediction
	with open('model.csv', 'w+') as file:
		file.write(f'{theta0}, {theta1}')
		file.close()
