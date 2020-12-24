import numpy

np.random.seed(42)

def relu(x):
    return(x>0) * x

def relu2deriv(output):
    return(output>0)

#inputs 
streetlights = np.array([[0,0,1],
                        [0,1,1],
                        [1,1,1],
                        [1,0,1]])

walk = np.array([1,0,1,1])

hidden_size_1 = 4
alpha = 0.2
iterations = 1

weights_0_1 = 2 * np.random.random((3,hidden_size_1)) - 2
weights_1_2 = 2 * np.random.random((hidden_size_1, 1)) - 2


#weights updates 
for i in range(iterations):
    error_layer_2 = 0
    for j in range(len(walk)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu( np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error_layer_2 += np.sum((layer_2 - walk[i:i+1])**2)
        layer_2_delta = layer_2 - walk[i:i+1] 

        layer_1_delta = layer_2_delta * weights_1_2.T.dot(relu2deriv(layer_1))

        weights_1_2 -= layer_2 .dot(layer_2_delta)
        weights_0_1 -= layer_1.T.dot(layer_1_delta)


#missing the print statement at the end 
#think of layer_1 as a) passing the layer_2_delta among the weights and then times/not dot in imput 
#weight updates require a T as coming back down to recreate themselves. 



 