
import numpy as np
import pandas as pd
from typing import Tuple


def predict(theta0: float , theta1: float, kilometrage: float) -> float:
	"""
	simple linear regression prediction function
	Given the learned weights theta0 and theta1 and a value of kilometrage (mileage),
	it computes and returns an estimated price p
	"""
	return theta0 + theta1 * kilometrage

def get_thetas() -> Tuple[float, float]:
	"""
	retrieve the values of theta0 and theta1 from a saved file on disk. 
	These values represent the parameters of a trained linear regression model.
	."""
	try:
		thetas = np.load("theta.npy")
		return (thetas[0], thetas[1])
	except:
		save_theta(0, 0)
	return (0, 0)

def save_theta(theta0: float, theta1: float) -> None:
	"""Save theta on the files name  it theta.npy"""
	np.save("theta", np.array([theta0, theta1]))
	print("theta has been saved on this disk.")


def average_cost(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float) -> float:
	"""Return the average error for given weights."""
	predictions = predict(theta0, theta1, features)
	errors = np.abs(predictions - targets)
	return (1 / errors.shape[0]) * np.sum(errors)


def trainning_model(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float, learning_rate: float) -> Tuple[float, float]:
	"""
	error function calculates the error between predictions and actual prices.
	train function performs one step of gradient descent to update the weights 
	(theta0 and theta1) using the given features, targets, and learning rate
	"""
	predictions = predict(theta0, theta1, features)
	errors = predictions - targets
	delta0 = learning_rate * (1 / errors.shape[0]) * np.sum(errors)
	delta1 = learning_rate * (1 / errors.shape[0]) * np.sum(errors * features)
	return (theta0 - delta0, theta1 - delta1)

def Normalize_data(data: pd.DataFrame) -> Tuple[float, float]:
	"""
	function is responsible for normalizing the data 
	function normalizes the "km" and "price" columns of the 
	input dataset by scaling them to a common range, 
	making it easier for the linear regression algorithm to learn from the data and make accurate prediction
	
	"""
	max_km = data["km"].max()
	max_price = data["price"].max()
	data["km"] = data["km"] / max_km
	data["price"] = data["price"] / max_price
	return (max_km, max_price)