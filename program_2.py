
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Normalize_data, save_theta, trainning_model, average_cost

from typing import Tuple

def main() -> None:
	"""
	this program for implementing simple linear regression 
	with a single feature, which is the mileage of a car. 
	the training and save the theta.
	"""
	data = pd.read_csv("./data.csv")
	max_km, max_price = Normalize_data(data)
	max_iter = 1000
	learning_rate = 0.1
	data = data.values
	features = data[:,0]
	targets = data[:,1]
	errors = []
	theta0, theta1 = (0.0, 0.0)

	for i in range(1, max_iter + 1):
		for j in range(0, data.shape[0], data.shape[0]):
			x = features[j:j + data.shape[0]]
			y = targets[j:j + data.shape[0]]
			theta0, theta1 = trainning_model(x, y, theta0, theta1, learning_rate)
		avg_error = average_cost(data[:, 0], data[:, 1], theta0, theta1)
		errors.append(avg_error)
	plt.plot(np.array(errors))
	plt.show()
	theta0 *= max_price
	theta1 *= (max_price / max_km)
	print("Theta0: {:.4f}".format(theta0))
	print("Theta1: {:.4f}".format(theta1))
	save_theta(theta0, theta1)

if (__name__ == "__main__"):
	main()