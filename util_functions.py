
import numpy as np

#active functions
def sigmoid(x):
	return 1/(np.exp(x)+1)

def relu(x):
	if x>0:return x
	return 0
