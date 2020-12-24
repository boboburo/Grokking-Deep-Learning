import numpy as np 
np.random.seed(1)

sl =np.array([[1,0,1],
                [0,1,1],
                [0,0,1],
                [1,1,1]])

walk = np.array([[1,1,0,0]]).T

alpha = 0.01
hidden_state = 4

w01 = 2 * np.random.random((3,hidden_state)) - 1
w12 = 2 * np.random.random((hidden_state, 1)) - 1

relu = lambda x: (x >0)* x
relu2deriv = lambda x: (x>0)

for it in range(60):
    error = 0
    for i in range(len(sl)):
        layer_0 = sl[i:i+1]
        layer_1 = relu(np.dot(layer_0, w01))
        layer_2 = np.dot(layer_1, w12)
        
        error += np.sum((layer_2 - walk[i:i+1])**2)
        
        layer_2_delta = (layer_2 - walk[i:i+1])
        layer_1_delta = layer_2_delta.dot(w12.T) * relu2deriv(layer_1)
        
        w12 -= layer_1.T.dot(layer_2_delta)
        w01 -= layer_0.T.dot(layer_1_delta)
    print(f"Error:{error}")

    #weight updates are wrong - it is the input .dot delta
    #don't forget the update symbols.