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


def normalize_feature(feature):
	min_val = min(feature)
	max_val = max(feature)
	normalized_feature = [(x-min_val)/(max_val - min_val) for x in feature]
	return normalized_feature


def gradient_descent(mileage, price, learning_rate, num_iter):
	m = len(mileage)
	theta0, theta1 = 0, 0

	for _ in range(num_iter):
		tmp_theta0 = (learning_rate / m) * sum([(theta0 + theta1 * mileage[i] - price[i]) for i in range(m)])
		tmp_theta1 = (learning_rate / m) * sum([(theta0 + theta1 * mileage[i] - price[i]) * mileage[i] for i in range(m)])

		theta0 -= tmp_theta0
		theta1 -= tmp_theta1

	print(theta0, theta1)
	return theta0, theta1

def main():
	data_file = 'data.csv'
	learning_rate = 0.001
	num_iter = 1000

	mileage, price = load_dataset(data_file)
	normalized_mileage = normalize_feature(mileage)
	normalized_price = normalize_feature(price)
	theta0, theta1 = gradient_descent(normalized_mileage, normalized_price, learning_rate, num_iter)

	with open('trained_vars.txt', 'w') as file:
		file.write(f'{theta0}\n{theta1}\n')

if __name__ == '__main__':
	main()

