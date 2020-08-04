# NEURAL NETWORKS AND DEEPLEARNING
---
## Introduction to deep learning
---
### What is a Neural Network ?
- Neural Network is referred to as **connectionist learning** due to the connections between units.
- Single neuron is linear regression without applying activation(perceptron).
- Basically a single neuron will calculate **weighted sum** of input(W.T * X) and then we can set a threshold to predict output in a perceptron. If weighted sum of input cross the threshold, perceptron fires and if not then perceptron doesn't predict.
- Perceptron can take real values input or boolean values.
- Actually, when w*x +b = 0 the perceptron outputs 0.
- Disadvantage of perceptron is that it only output binary values and if we try to give small change in weight and bias then perceptron can flip the output. We need some system which can modify the output slightly according to small change in weight and bias. Here comes sigmoid function in picture.
- If we change perceptrong with a sigmoid function, we can make slight change in output.
- e.g. output in perceptrong = 0, you can slightly changed weight and bias, output becomes = 1, but actual output is 0.7. In case of sigmoid, output' = 0, slight change in weight and bias, output = 0.7.
- If we apply sigmoide function then Single neuron will act as ***Logistic Regression***.
- We can understand difference between perceptron and sigmoid function by looking at sigmoid function graph
- Simple Neural Network graph:
    - ![Simple Neural Network](https://miro.medium.com/max/1100/1*YgJ6SYO7byjfCmt5uV0PmA.png)
    - Image taken from medium.com
- RELU ( rectified linear unit) is the most popular activation right now that makes deep Neural Networks train faster now.
- Hidden layers predicts connection between inputs automatically, thats what deep learning is good at.
- Deep Neural Network consists of more hidden layers (Deeper layers)
    - ![Deep NN](https://www.houseofbots.com/images/news/11733/cover.png)
    - Image taken form houseofbots.com
- Each Input will be connected to hidden layer and the Neural Network will decide the connections.
- Supervised learning means we have (X, Y) and we need to get function that maps X to Y.

### Advantage vs Disadvantage
| Advantage                                             | Disadvantage                                                                  |
|-------------------------------------------------------|-------------------------------------------------------------------------------|
| High tolerance of noisy data                          | Involve long training times                                                   |
| Well suited for continuous inputs and outputs         | Require a number of parameters that are typically best determined empirically |
| Inherently parallel, speed up the computation process | Poor interpretability                                                         |
| Perform both classification and regression            |                                                                               |

### Defining a Network Topology
- *How can I design the neural network's topology ?*
    - Specifying the number of units in the input layer (number of attributes).
    - Specifying the number of hidden layers (if more than one), the number of units in each hidden layer, and the number of units in the output layer.

    - Normalizing the input values for each attribute measured in the training tuples → speed up learning phase.
    - Discrete-valued attribites may be enconded.
    - ***No rules as to the "best" number of hidden layer units.*** Network design is a trial-and-error process.
    - Cross-validation techniques for accuracy estimation.

### Supervised learning with neural networks
- Different types of neural networks for supervised learning which includes:
    - CNN (Convolution Neural Networks), which is useful in computer vision.
    - RNN (Recurrent Neural Networks), which is useful for sequential model,e.g. Speech recognition, NLP..
    - Standard Neural Network (Useful for Structured data).
    - Hybrid/custom Neural Network
- Structured data is like the databases and tables.
- Unstructured data is like images, video, audio, and text.
- Structured data gives more money because companies relies on prediction on its big data.

### Why is deep learning taking off?
- Deep learning is taking off for 3 reasons:
    - Data:
        - Using this image we can conclude:
            - ![Data and Performance](https://miro.medium.com/max/3606/1*1ADLHcqXhmtgDBkMF7b2Hw.png)
        - For small amount of data, Neural Network can perform as Linear regression or Support vector machine (SVM)
        - For big amount of data, a small Neural Network is better than SVM
        - For big amount of data, big NN is better than medium one and medium one is better than small one
        - Hopefully we have a lot of data because the world is using the computer a little bit more:
            - Mobiles
            - Internet of things (IOT)
    - Computation:
        - GPUs
        - Powerful CPUs
        - Distributed computing
        - ASICs
    - Algorithm:
        - Creative algorithms has appeared that changed the way NN works
            - e.g Using RELU function is much better than using SIGMOID function in training a Neural Network because it helps with the vanishing gradient problem

## Neural Networks Basics
---

### Binary Classification
- Using Logistic Regression to make a binary classifier
    - ![Binary classifier](https://www.researchgate.net/profile/Francois_Kawala/publication/285653348/figure/fig5/AS:669589956460558@1536654094794/This-illustration-present-a-binary-classification-that-is-performed-on-two-features.png)
    - Image taken from researchgate.net
- e.g. Cat and Dog classification:
    - ![Cat and Dog](https://miro.medium.com/max/843/1*g2TQog59vQDCpvtioJZSxw.png)
- Notations:
    - `M is the number of training vectors`
    - `Nx is the size of input vector`
    - `Ny is the size of the output vector`
    - `X(1) is the first input vector`
    - `Y(1) is the first output vector`
    - `X = [x(1) x(2) .. x(M)]`
    - `Y = (x(1) x(2) .. x(M))`

### Logistic Regression
- Used for binary classification
- Equations:
    - Simple equation: `y = wx +b`
    - If x is a vector: `y = w.T * x +b`
    - If we need y to be in range [0, 1] (probability): `y = sigmoid(w.T * x +b)`
- In binary classification `Y` has to be between `0` and `1`
- `w` is a vector of `Nx` and `b` is a real number

### Logistic Regression cost function
- Loss function would be Square root error: `L(y', y) = 1/2 (y' - y)^2`
    - Not use this notation because it leads us to optimization problem which is non convex, means it contains local optimum points.
- We will use: `L(y', y) = - (y * log(y') + (1 - y) * log(1 - y'))
- Explain:
    - if `y = 1`→`L(y', 1) = - log(y')` →we want `y'` to be the largest→`y'` biggest value is 1
    - if `y = 0`→`L(y', 10 = - log(1-y')` →we want `1-y'` to be the largest→`y'` to be small as possible because it can only has 1 value
- Cost function: `J(w,b) = (1/m) * Sum(L(y'[i],y[i]))`
- The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

### Gradient Descent
- Optimizer
- We eant to predict `w` and `b` that minimize the cost function.
- Our cost function is convex
- Initialize `w` and `b` to 0,0 or randomly in the convex function and then try to improve the values which reach minimum value.
- In logistic regression people always use 0,0 instead of random.
- The gradient decent algorithm repeats: `w = w - alpha * dw` where alpha is the learning rate and `dw` is the derivative of `w` (the change to `w`) or the slope of `w`
- Looks like greedy algorithms, the derivative give us the direction to improve our parameters
- The actual equations we will implement:
    - `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in w direction)
    - `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in d direction)

### Computation Graph
- Organizes the computation from left to right
    - ![Computation graph](https://lrscy.github.io/2018/10/22/DeepLearningNotes-NNandDL/Computation_Graph.png)

### Derivatives with a Computation Graph
- Chain rule: If x→y→z (x effects y and y effects z) then `d(z)/d(x) = d(z)/d(y) * d(y)/d(x)`
- ![DCG](https://lrscy.github.io/2018/10/22/DeepLearningNotes-NNandDL/Computing_Derivatives.png)
- We compute the derivatives on a graph from right to left.
- `dvar` means the derivatives of a final output variable with respect to various intermediate quantities.

### Logistic Regression Gradient Descent
- Gradient descent example for one sample with two features `X` and `Y`
- ![LRD](https://opyate.com/assets/posts/2017-11-22-week-2-of-neural-networks-and-deep-learning/logistic-regression-gradient-descent.png)

### Gradient Descent on m Examples
- Variables:
  ```
  	X1                  Feature
  	X2                  Feature
  	W1                  Weight of the first feature.
  	W2                  Weight of the second feature.
  	B                   Logistic Regression parameter.
  	M                   Number of training examples
  	Y(i)                Expected output of i
  ```
  
- Then we have:
![](https://github.com/mbadry1/DeepLearning.ai-Summary/blob/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/09.png?raw=true)
- From right to left, we will calculate derivations compared to the result:
```
    d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
	d(z)  = d(l)/d(z) = a - y
	d(W1) = X1 * d(z)
	d(W2) = X2 * d(z)
	d(B)  = d(z)
```
- Pseudo code:
```
    J = 0; dw1 = 0; dw2 =0; db = 0;                 # Devs.
	w1 = 0; w2 = 0; b=0;							# Weights
	for i = 1 to m
		# Forward pass
		z(i) = W1*x1(i) + W2*x2(i) + b
		a(i) = Sigmoid(z(i))
		J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

		# Backward pass
		dz(i) = a(i) - Y(i)
		dw1 += dz(i) * x1(i)
		dw2 += dz(i) * x2(i)
		db  += dz(i)
	J /= m
	dw1 /= m
	dw2 /= m
	db /= m

	# Gradient descent
	w1 = w1 - alpha * dw1
	w2 = w2 - alpha * dw2
	b = b - alpha * db
```
- The above code should run for some iterations to minimize error.
- There will be two inner loops to implement the logistic regression
- Vectorization is so important on deep learning to reduce loops.

### Vectorization
- Deep learning shines when the dataset are big. However for loops will make you wait a lot for a result. Thats why we need vectorization to get rid of some of our for loops.
- NumPy library (dot) function is using vectorization by default.
- The vectorization can be done on CPU or GPU thought the **SIMD** operation. But its faster on GPU.
- SIMD is short for Single Instruction/Multiple Data, while the term SIMD operations refers to a computing method that enables processing of multiple data with a single instruction. In contrast, the conventional sequential approach using one instruction to process each individual data is called scalar operations.
- Whenever possible avoid for loops.
- Most of the NumPy library methods are vectorized.

### Vectorizing Logistic Regression
- As an input we have a matrix `X` and its `[Nx, m]` and a matrix `Y` and its `[Ny, m]`.
- Compute at instance `[z1, z2 .. zm] = W' * X + [b, b, .. b]`. Python:
```
    Z = np.dot(W.T, X) + b       # Vectorization, then broadcasting, Z shape is (1, m)
    A = 1 / (1 + np.exp(-Z))     # Vectorization, A shape is (1, m)
```
- Vectorizing Logistic Regression's Gradient Output:
```
  	dz = A - Y                  # Vectorization, dz shape is (1, m)
  	dw = np.dot(X, dz.T) / m    # Vectorization, dw shape is (Nx, 1)
  	db = dz.sum() / m           # Vectorization, dz shape is (1, 1)
```
### Notes on Python and NumPy
- In NumPy, `obj.sum(axis = 0)` sums the columns while `obj.sum(axis = 1)` sums the rows.
- In NumPy, `obj.reshape(1,4)` changes the shape of the matrix by broadcasting the values.
- Reshape is cheap in calculations so put it everywhere your're not sure about the calculations.
- Broadcasting works when you do a matrix operation with matrices that doesn't match for the operation, in this case NumPy automatically makes the shapes ready for operation by broadcasting the values.
- In general principle of broadcasting. If you have an (m, n) matrix and you add (+) or substract (-) or multiply (*) or divide (/) with an (1, n) matrix, then this will copy it m times to an (m, n) matrix. THe same with if you use those operations with a (m, 1) matrix, then this will copy it n times into (m, n) matrix. And then apply the addition, substraction, and multiplication of division element wise.
- Some tricks to eliminate all the strange bugs in the code:
    - If you didn't specify the shape of a vector, it will take a shape of `(m, )` and the transpose operation won't work. You have to reshape it to `(m, 1)`.
    - Try to not use the rank one matrix in ANN.
    - Don't hesitate to use `assert(a.shape == (5, 1))` to check if your matrix shape is the required one.
    - If you've found a rank one matrix try to run reshape on it.
- Jupyter / IPython notebooks are so useful library in python that makes it easy to integrate code and document at the same time. It runs in the browser and doesn't need an IDE to run.
    - To open Jupyter Notebook, open the command line and call: jupyter-notebook It should be installed to work.
- To Compute the derivative of Sigmoid:
```
    s = sigmoid(x)
	ds = s * (1 - s)       # derivative  using calculus
```
- To make an image of (widt, height, depth) be a vector, use this:
```
v = image.reshape(image.shape[0]* image.shape[1]*image.shape[2], 1)
```
- Gradient descent converges faster after normalization of the input matrices.
### General Notes
- The main steps for building a Neural Network are:
    - Define the model structure.
    - Initialize the model's parameters.
    - Loop:
        - Calculate current loss (forward propagation)
        - Calculate current gradient (backward propagation)
        - Update parameters (gradient descent)
- Preprocessing the dataset is important.
- Tuning the learning rate can make a big difference to the algorithm.

## Shallow Neural Network
---
> Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.

### Neural Networks Overview
- In logistic regression we had:
```
X1  \  
X2   ==>  z = XW + B ==> a = Sigmoid(z) ==> l(a,Y)
X3  /
```
- In neural networks with one layer we will have:
```
X1  \  
X2   =>  z1 = XW1 + B1 => a1 = Sigmoid(z1) => z2 = a1W2 + B2 => a2 = Sigmoid(z2) => l(a2,Y)
X3  /
```
- `X` is the input vecor `(X1, X2, X3)` and `Y` is the output variable `(1 x 1)`
- Neural Network is stack of logistic regression objects.

### Neural Network Representation
- We will define the neural networks that has one hidden layer.
- Neural Network contains of input layers, hidden layers, output layers.
- Hidden layer means we cant see that layers in the training set.
- `a0 = x` (the input layer).
- `a1` will represent the activation of the hidden neurons.
- `a2` will represent the output layer.
- We are talking about 2 layers Neuron Network. The input layer isn't counted.

### Computing a Neural Network's Output
- Equations of Hidden layers:
    - ![](https://miro.medium.com/max/705/1*buxOnswsinejx2FVZDuF8w.png)
    - Image taken from medium.com
- Some infomations:
    - `no_of_hiddnen_neurons = 4`
    - `Nx = 3`
    - Shapes of the variables:
        - `W1` is the matrix of the first hidden layer, it has a shape of `(no_of_hidden_neurons, Nx)`
        - `b1` is the matrix of the first hidden layer , it has a shape of (no_of_hidden_neurons, 1)
        - `z1` is the result of the equation ` z1 = W1 * X + b`, it has a shape of (no_hidden_neurons, 1)
        - `a1` is the result of the equation `a1 = sigmoid(z1)`, it has a shape of `(1, no_of_hidden_neurons)`
        - `W2` is the matrix of the second hidden layer, it has a shape of `(1, no_of_hidden_neurons)`
        - `b2` is the matrix of the secon hidden layer, it has a shape of `(1, 1)`
        - `z2` is the result of the equation `z2 = W2 * a1 + b`, it has a shape of `(1, 1)`
        - `a2` is the result of the equation `a2 = sigmoid(z2)`, it has a shape of `(1, 1)`
### Vectorizing across multiple examples
- Pseudo code for forward propagation for the 2 layers NN:
```
for i = 1 to m
  z[1, i] = W1*x[i] + b1      # shape of z[1, i] is (noOfHiddenNeurons,1)
  a[1, i] = sigmoid(z[1, i])  # shape of a[1, i] is (noOfHiddenNeurons,1)
  z[2, i] = W2*a[1, i] + b2   # shape of z[2, i] is (1,1)
  a[2, i] = sigmoid(z[2, i])  # shape of a[2, i] is (1,1)
```
- Lets say we have `X` on shape `(Nx, m)`. Pseudo code:
```
Z1 = W1X + b1     # shape of Z1 (noOfHiddenNeurons,m)
A1 = sigmoid(Z1)  # shape of A1 (noOfHiddenNeurons,m)
Z2 = W2A1 + b2    # shape of Z2 is (1,m)
A2 = sigmoid(Z2)  # shape of A2 is (1,m)
```
- m is the number of columns.
- If `X` = `A0`:
```
Z1 = W1A0 + b1    # shape of Z1 (noOfHiddenNeurons,m)
A1 = sigmoid(Z1)  # shape of A1 (noOfHiddenNeurons,m)
Z2 = W2A1 + b2    # shape of Z2 is (1,m)
A2 = sigmoid(Z2)  # shape of A2 is (1,m)
```

### Activation function
- Sigmoid:
    - So far, we are using sigmoid, but in some cases other functions can be alot better.
    - Sigmoid can lead us to gradient descent problem whre the updates are so low.
    - Sigmoid activation function range is [0, 1]: `A = 1 / (1 + np.exp(-z) # Where z is the input matrix`
    - ![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/1200px-Sigmoid-function-2.svg.png)
    - Derivation of Sigmoid activation function:
    ```
    g(z)  = 1 / (1 + np.exp(-z))
    g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
    g'(z) = g(z) * (1 - g(z))
    ```
    - ![](https://i.stack.imgur.com/inMoa.png)
- Tanh:
    - Range is [-1, 1] (shifted version of sigmoid function)
    - In NumPy we can implement Tanh using one of these methods: 
    ```
    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) # Where z is the input matrix
    ```
    - Or `A = np.tanh(z) # Where z is the input matrix`
    - Tends to make each layer's output more or less normalized(i.e, centered around 0) at the beginning of training. This often helps speed up convergence.
    - ![](https://www.20sim.com/webhelp/tanh_zoom60.jpg)
    - Derivation of Tanh activation function:
    ```
    g(z)  = (e^z - e^-z) / (e^z + e^-z)
    g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
    ```
    - ![](https://lh3.googleusercontent.com/proxy/3DgCVdhgra_kWPXwtpb0OW5Ipkk23JbPC2cwDp7z3zL4sNs6U0El31HV4F0Sz0HEynynBlawja1r6Xq98o1dfRWl3qeQZOkJ25I793sSsT_L4zp00vXsBHj4_QT4nnI)
- **Disadvantage** of tanh and sigmoid is that if the input is too small or too high, the slope will be near zero which will cause us the gradient decent problem.
- RELU:
    - One of the popular activation functions that solved the slow gradient decent.
    - `RELU = max(0, z) # so if z is negative the slope is 0 and if z is positive the slope remains linear.`
    - Continuous but unfortunately not diffrentiable at `z = 0` (slope changes abruptly, which can make Gradient Descent bounce around).
    - Works very well in practice and has the advangtage of being fast to compute.
    - Does not have a maximum output value also helps reduce some issues during Gradient Descent.
    - ![](https://michielstraat.com/talk/mastertalk/featured.png)
    - Derivation of RELU:
    ```
    g(z)  = np.maximum(0,z)
    g'(z) = { 0  if z < 0
              1  if z >= 0  }
    ```
    - ![](https://i.stack.imgur.com/UtuWP.png)
- Leaky RELU:
    - Different of RELU is that if the input is negative the slope will be so small. 
    - It works as RELU but most people uses RELU.
    - `Leaky_RELU = max(0.01z, z)` # the 0.01 can be a parameter for your algorithm
    - ![](https://i0.wp.com/knowhowspot.com/wp-content/uploads/2019/04/IMG_20190406_220045-1.jpg?fit=1024%2C561&ssl=1)
    - Derivation of Leaky Relu:
    ```
    g(z)  = np.maximum(0.01 * z, z)
    g'(z) = { 0.01  if z < 0
              1     if z >= 0   }
    ```
    - ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/10/17161626/leaky-relu-derivative.png)
- So here is some basic rules for choosing activation functions, if your classification is between 0 and 1, use the output activation as sigmoid and the others as RELU.
- In NN you will decide a lot of choices like:
    - No of hidden layers.
    - No of neurodes in each hidden layer.
    - Learning rate. (The most important parameter)
    - Activation functions.
    - And others..
- It turns out there are no guide lines for that. You should try all activation functions for example.

### Why do you need non-linear activation functions?
- If removed the activation function from our algorithm that can be called linear activation function.
- Linear activation function will output linear activations:
    - Whatever hidden layers you add, the activation will be always linear like logistic regression, the it is useless in a lot of complex problems
- You might use linear activation function in one place - in the output layer if the output is real numbers (regression problem). But even in this case if the output is non negative you could use RELU instead.

### Gradient Descent for Neural Networks
- In this section we will have the full back propagation of the neural network.
- Gradient descent algorithm:
    - NN parameters:
        - `n[0] = Nx`
        - `n[1] = NoOfHiddenNeurons`
        - `n[2] = NoOfOutputNeurons = 1`
        - `W1` shape is `(n[1],n[0])`
        - `b1` shape is `(n[1],1)`
        - `W2` shape is `(n[2],n[1])`
        - `b2` shape is `(n[2],1)`
    - Cost function `I = I(W1, b1, W2, b2) = (1/m) * Sum(L(Y,A2))`
    - Then Gradient Descent:
    ```
    Repeat:
		Compute predictions (y'[i], i = 0,...m)
		Get derivatives: dW1, db1, dW2, db2
		Update: W1 = W1 - LearningRate * dW1
				b1 = b1 - LearningRate * db1
				W2 = W2 - LearningRate * dW2
				b2 = b2 - LearningRate * db2
    ```
- Forward propagation:
    ```
    Z1 = W1A0 + b1    # A0 is X
    A1 = g1(Z1)
    Z2 = W2A1 + b2
    A2 = Sigmoid(Z2)      # Sigmoid because the output is between 0 and 1
    ```
- Backpropagation:
    ```
    dZ2 = A2 - Y      # derivative of cost function we used * derivative of the sigmoid function
    dW2 = (dZ2 * A1.T) / m
    db2 = Sum(dZ2) / m
    dZ1 = (W2.T * dZ2) * g'1(Z1)  # element wise product (*)
    dW1 = (dZ1 * A0.T) / m   # A0 = X
    db1 = Sum(dZ1) / m
    # Hint there are transposes with multiplication because to keep dimensions correct
    ```
- How we derived the 6 equations of the backpropagation:
    ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSbUM1bgnlZkh8TqhxlgcWlDHvBgZd6ZoAsjQ&usqp=CAU)

### Random Initialization 
- In logistic regression, it wasn't important to initialize the weights randomly, while in Neural NetWork, we have to initialize randomly.
- If we initialize all the weights with **zeros** in Neural Network, it **won't work** (initializing bias with zero is OK):
    - All hiddent units will be completely identical (symmetric) - compute exactly the same function
    - On eachgradient descent iteration all the hidden units will always update the same.
- TO solve this we initialize the W's with a small random numbers:
    ```
    W1 = np.random.randn((2,2)) * 0.01    # 0.01 to make it small enough
    b1 = np.zeros((2,1))                  # its ok to have b as zero, it won't get us to the symmetry breaking problem
    ```
- We need small values because in sigmoid (or tanh), for example, if the weight is too large, you are more likely to end up even at the very start of training with very large values of Z. Which causes your tand or your sigmoid activation function to be saturated, thus slowing down the learning. If yoy don't have any sigmoid or tanh activation functions throughout your neural network, this is less of an issue.
- Constant 0.01 is alright for 1 hidden layer networks, but if the NN is deep this number can be changed but it will always be a small number.

## Deep Neural Networks
---
> Undetstand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

### Deep L-layer Neural Network
- Shallow Neural Network is a Neural Network with one or two layers.
- Deep Neural Network is a Neural Network with three or more layers.
- We will use the notation `l` to denote the number of layers in a NN.
- `n[l]` is the number of neurons in specific layer `l`.
- `n[0]` denotes the number of neurons input layer. `n[L]` denotes the number of neurons in output layer.
- `g[l]` is the activation function.
- `a[l] = g[l](z[l])`
- `w[l] weights is used for `z[l]`
- `x = a[0]`, `a[l] = y'
- These were the notation we will use for deep neural network.
- So we have:
    - A vector `n` of shape (1, NoOfLayers+1)
    - A vector `g` of shape (1, NoOfLayers)
    - A list of different shapes `w` based on the number of neurons on the previous and the current layer.
    - A list of different shapes `b` based on the number of neurons and the current layer.
### Forward Propagation in Deep Network
- Forward propagation general rule for one input:
    ```
    z[l] = W[l]a[l-1] + b[l]
    a[l] = g[l](a[l])
    ```
- Forward propagation general rule for `m` inputs:
    ```
    Z[l] = W[l]A[l-1] + B[l]
    A[l] = g[l](A[l])
    ```
- We can't compute the whole layers forward propagation without a for loop so its OK to have a for loop here.
- The dimensions of the matrices are so important you need to figure it out.

### Getting your matrix dimensions right
- The best way to debug your matrices dimensions is by a pencil and paper.
- Dimension of `W` is `(n[l], n[l-1]). Can be thought by right to left.
- Dimension of `b` is `(n[l], 1)`.
- `dw` has the same shape as `W`, while `db is the same shape as `b`.
-  Dimension of `Z[l]`, `A[l]`, `dZ[l]`, and `dA[l]` is `(n[l], m)`

### Why deep representations ?
- Why deep NN works well, we will discuss this question in this section.
- Deep NN makes relations with data from simpler to complex. In each layer it tries to make a relation with the previous layer. E.g.:
    - Face recognition application:
    E.g:  Image→Edges→Face parts→Faces →desired face
    -  Audio recognition application:
    E.g: Audio→Low→level sound features (ssb, bb)→Phonemes→Words→Sentences
    - Neural Researchers think that deep neural networks 'think' like brains (simple to complex)
    - Circuit theory and deep learning:
    ![](https://cdn-images-1.medium.com/max/1000/1*F75UluioExBRoNjGIH3Lww.png)
- When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions(e.g. Logistic Regression), then try the shallow neural network and so on.
### Building blocks of deep neural networks
- Forward and back propagation for a layer l:
    ![](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/1-%20Neural%20Networks%20and%20Deep%20Learning/Images/10.png)
- Deep NN blocks:
    ![](https://opyate.com/assets/posts/2017-11-24-week-4-of-neural-networks-and-deep-learning/forward-and-backward-functions.png)

### Forward and Backward Propagation
- Peseudo code for forward propagation for layer l:
    ```
    Input  A[l-1]
    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l])
    ```
- Pseudo code for back propagation for layer l:
    ```
    Input da[l], Caches
    dZ[l] = dA[l] * g'[l](Z[l])
    dW[l] = (dZ[l]A[l-1].T) / m
    db[l] = sum(dZ[l])/m                # Dont forget axis=1, keepdims=True
    dA[l-1] = w[l].T * dZ[l]            # The multiplication here are a dot product.
    Output dA[l-1], dW[l], db[l]
    ```
- If we have used our loss function then:
    ```
    dA[L] = (-(y/a) + ((1-y)/(1-a)))
    ```
### Parameters vs Hyperparameters
- Main parameters of NN is `W` and `b`.
- Hyper parameters (parameters that control the algorithm) are like:
    - Learning rate.
    - Number of iteration
    - Number of hidden layers `L`.
    - Number of hiddent units `n`.
    - Choice of activation functions.
- You have to try values yourself of hyper parameters.
- In the earlier days of DL and ML learning rates was often called a parameterm but it really is ( and now everybody call it) a hyperparameter.

