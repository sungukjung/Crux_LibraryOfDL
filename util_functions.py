
import numpy as np

#active functions
def sigmoid(x):
	return 1/(np.exp(-x)+1)

def relu(x):
	if x>0:return x
	return 0

def cross_entropy_error(y,t):
    if y.ndim ==1:
        y= y.reshape(1,y.size)
        t= t.reshape(1,t.size)
        
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y +1e-7))/batch_size


def cross_entropy_error_one_hot_encoding(y,t):
	if y.ndim==1:
		y= y.reshape(1,y.size)
		t= t.reshape(1,t.size)

	if y.size==t.size:
		t= t.argmax(axis=1)
	batch_size = y.shape[0]
	#you can ignore if it is one-hot-encoding, because answer t is 1, the other is 0.
	#So u can pick only answer node and calculate.
	return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

def diff_sigmoid(x):
	return sigmoid(x)-np.square(sigmoid(x))
	