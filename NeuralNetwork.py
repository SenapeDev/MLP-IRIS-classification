import numpy as np

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.5) -> None:
        # neural network attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # neurons attributes
        self.input_neurons = np.zeros(self.input_size + 1)
        self.hidden_neurons = np.zeros(self.hidden_size + 1)
        self.output_neurons = np.zeros(output_size)

        # random-generated weights for input-hidden layer and for hidden-output layer
        self.IH_weights = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.HO_weights = np.random.uniform(-1, 1,(output_size, hidden_size))

        # random-generated biases for input-hidden layer and for hidden-output layer
        self.HL_bias = np.random.uniform(-1, 1, (hidden_size, 1))
        self.OL_bias = np.random.uniform(-1, 1, (output_size, 1))

        # velocity terms for momentum
        self.IH_velocity = np.zeros_like(self.IH_weights)
        self.HO_velocity = np.zeros_like(self.HO_weights)


    def sigmoid(self, x:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    

    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x))


    def feed_forward(self, inputs:np.ndarray) -> np.ndarray:
        # input layer
        self.input_neurons[:self.input_size] = inputs
        self.input_neurons[-1] = -1

        # weighted sum (w and input neurons)
        hidden_input = np.dot(self.IH_weights, self.input_neurons)
        self.hidden_neurons[:-1] = self.sigmoid(hidden_input + self.HL_bias)
        self.hidden_neurons[-1] = -1

        # weighted sum (w and hidden neurons)
        self.output_input = np.dot(self.HO_weights, self.hidden_neurons)
        self.output_neurons = self.softmax(self.output_input)


    def back_propagate(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # Calcolo errore e aggiornamento per il livello di output
        for o in range(self.output_size):
            delta_output = (self.output_neurons[o] - targets[o]) * self.output_neurons[o] * (1 - self.output_neurons[o])
            for h in range(self.hidden_size + 1):  # Include bias
                self.HO_velocity[o][h] = self.momentum * self.HO_velocity[o][h] - self.learning_rate * delta_output * self.hidden_neurons[h]
                self.HO_weights[o][h] += self.HO_velocity[o][h]

        # Calcolo errore per il livello nascosto
        hidden_errors = np.zeros(self.hidden_size)
        for h in range(self.hidden_size):
            hidden_errors[h] = sum(
                self.HO_weights[o][h] * (self.output_neurons[o] - targets[o]) for o in range(self.output_size)
            ) * self.hidden_neurons[h] * (1 - self.hidden_neurons[h])

        # Aggiornamento dei pesi tra input e livello nascosto
        for h in range(self.hidden_size):
            for i in range(self.input_size + 1):  # Include bias
                self.IH_velocity[h][i] = self.momentum * self.IH_velocity[h][i] - self.learning_rate * hidden_errors[h] * self.input_neurons[i]
                self.IH_weights[h][i] += self.IH_velocity[h][i]


    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, decay: float = 0.01) -> None:
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in zip(X, y):
                self.feed_forward(inputs)
                total_loss += -np.sum(targets * np.log(self.output_neurons + 1e-9))  # Cross-entropy loss
                self.back_propagate(inputs, targets)

            # Update learning rate (decay)
            self.learning_rate = self.learning_rate / (1 + decay * epoch)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")
