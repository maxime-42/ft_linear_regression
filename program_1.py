
from utils import  get_thetas

def predict(theta0: float , theta1: float, kilometrage: float) -> float:
	"""Returns an estimated price prediction for given weights and kilometrage."""
	return theta0 + theta1 * kilometrage

def main() -> None:
	"""
		The program prompt for a mileage, and then give
		you back the estimated price for that mileage
	"""
	kilometrage = input("Enter kilometrage: ")
	try:
		kilometrage = int(kilometrage)
	except:
		print("Cannot cast '{}' to float.".format(kilometrage))
		exit(1)
	theta0, theta1 = get_thetas()
	prediction = predict(theta0, theta1, kilometrage)
	print("Estimated price for {}kms: {:.4f}$".format(kilometrage, prediction))
	if (theta0 == 0 and theta1 == 0):
		print("Note: it seems this the model is not trained yet.")

if (__name__ == "__main__"):
	main()