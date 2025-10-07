import numpy as np

# --- Helper Functions ---

def sigmoid(z):
  """The standard sigmoid activation function."""
  return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z) * (1 - sigmoid(z))

# New function 'f' and its derivative 'f_prime' (using ReLU as example)
def f(z):
  """The new activation function f(z) = ReLU(z) = max(0, z)."""
  return np.maximum(0, z)

def f_prime(z):
  """Derivative of f(z): f'(z) = 1 if z > 0, 0 otherwise."""
  return (z > 0).astype(z.dtype)

# --- Modified Network Class ---

class Network(object):
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    # Define the location of the modified neuron
    # Layer 1 is the first hidden layer (index 1), Neuron 0 is the first neuron.
    self.modified_layer_index = 1
    self.modified_neuron_index = 0 

  def feedforward(self, a):
    """Return the network's output, applying f(z) to the single modified neuron."""
    # layer_index starts at 0 for the first set of weights/biases
    for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
      z = np.dot(w, a) + b
      
      if layer_index + 1 == self.modified_layer_index:
        # Layer is the target for modification (e.g., Layer 1)
        
        # 1. Start with all activations using the standard sigmoid
        a = sigmoid(z)
        
        # 2. Overwrite the activation for the single modified neuron with f(z)
        # Note: z[index, 0] selects the single scalar value
        a[self.modified_neuron_index, 0] = f(z[self.modified_neuron_index, 0])
      else:
        # Standard layer: use sigmoid for all neurons
        a = sigmoid(z)
        
    return a
  
  def backprop(self, x, y):
    """
    Implements backpropagation, applying f_prime(z) only to the 
    derivative calculation of the single modified neuron.
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # --- Forward Pass (Must match feedforward) ---
    activation = x
    activations = [x]
    zs = []
    
    for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
      z = np.dot(w, activation) + b
      zs.append(z)
      
      if layer_index + 1 == self.modified_layer_index:
        activation = sigmoid(z)
        activation[self.modified_neuron_index, 0] = f(z[self.modified_neuron_index, 0])
      else:
        activation = sigmoid(z)
        
      activations.append(activation)
    
    # --- Backward Pass ---
    
    # 1. Output Layer (BP1: Always standard sigmoid if modification is elsewhere)
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # 2. Hidden Layers (BP2: Check for modified neuron's derivative)
    # l is the layer number counting backward (l=2 is the second-to-last layer)
    for l in range(2, self.num_layers):
      z = zs[-l]
      
      # Determine the layer index from the start: layer_index_from_start = self.num_layers - l
      layer_index_from_start = self.num_layers - l
      
      # Initialize derivative vector with standard sigmoid_prime
      sp = sigmoid_prime(z) 
      
      if layer_index_from_start == self.modified_layer_index:
        # The key change: overwrite the derivative for the single neuron with f_prime
        sp[self.modified_neuron_index, 0] = f_prime(z[self.modified_neuron_index, 0])
      
      # Propagate the error using the potentially modified derivative vector (sp)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      
      # Update gradients (BP3 & BP4)
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return (nabla_b, nabla_w)

  def update_mini_batch(self, mini_batch, eta):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb +dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
  
  def cost_derivative(self, output_activations, y):
    return (output_activations - y)

if __name__ == '__main__':
    net = Network([2, 3, 1])
    
    x = np.array([[1.0], [0.5]]) 
    y = np.array([[0.0]])
    
    print(f"Network: {net.sizes}. Modified Neuron: Layer {net.modified_layer_index}, Index {net.modified_neuron_index} (Uses f/ReLU)")
    
    # --- Check Initial Output ---
    initial_output = net.feedforward(x)
    print(f"\nInitial Output: {initial_output.flatten()}")

    # --- Run One Training Step ---
    mini_batch = [(x, y)]
    learning_rate = 3.0
    net.update_mini_batch(mini_batch, learning_rate)
    
    # --- Check Updated Output ---
    updated_output = net.feedforward(x)
    print(f"Updated Output: {updated_output.flatten()}")
