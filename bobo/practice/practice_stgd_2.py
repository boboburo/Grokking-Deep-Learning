import numpy as np 
np.random.seed(42)

relu = lambda x: (x >= 0)*x
relu2deriv  = lambda x: (x > 0)

sl = np.array(([[1,0,0],
                [1,1,0],
                [1,1,1],
                [0,0,1]]))

walk = np.array([[1,1,0,0]]).T

hs = 4
alpha = 0.2 

w01 = 2*np.random.random((3, hs)) -
w12 = 2*np.random.random((hs, 1))

for iteration in range(20):
    error = 0
    for i in range(len(walk)):
        layer_0 = sl[i:i+1]
        layer_1 = relu(np.dot(layer_0, w01))
        layer_2 = np.dot(layer_1, w12)

        error += np.sum((layer_2 - walk[i:i+1])**2)

        l2_delta = layer_2 - walk[i:i+1]
        l1_delta = l2_delta.dot(w12.T)*relu2deriv(layer_1)

        w12 = alpha*l2_delta.dot(w12.T)
        w11 = alpha*l1_delta.dot(w01.T)
    
print(f"Iteration:{iteration}, Error:{error:.3f}")

#forgot to scale the weights
#weight updaete is the input.T * delta