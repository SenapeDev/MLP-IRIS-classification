import json
import numpy as np
import matplotlib.pyplot as plt

vector = np.ndarray

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, eta: float=.01, momentum: float=.9): 
        # neural network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.eta = eta
        self.momentum = momentum
        
        # initialize neurons
        self.input_neurons = np.zeros((1, self.input_size))
        self.hidden_neurons = np.zeros((1, self.hidden_size))
        self.output_neurons = np.zeros((1, self.output_size))

        # weights for input -> hidden and for hidden -> output
        self.IH_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.HO_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # bias for hidden and output layer
        self.HL_bias = np.zeros((1, self.hidden_size))
        self.OL_bias = np.zeros((1, self.output_size))
    
        # velocity terms for momentum
        self.IH_velocity = np.zeros_like(self.IH_weights)
        self.HO_velocity = np.zeros_like(self.HO_weights)
        self.HL_velocity = np.zeros_like(self.HL_bias)
        self.OL_velocity = np.zeros_like(self.OL_bias)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)


    def feedforward(self, X: vector) -> vector:
        # input -> hidden
        self.input_neurons = X
        self.hidden_input = np.dot(self.input_neurons, self.IH_weights) + self.HL_bias
        self.hidden_neurons = self.sigmoid(self.hidden_input)

        # hidden -> output
        self.output_input = np.dot(self.hidden_neurons, self.HO_weights) + self.OL_bias
        self.output_neurons = self.sigmoid(self.output_input)

        return self.output_neurons


    def backpropagation(self, X: vector, y: vector) -> None:
        # get error at output layer
        output_error = self.output_neurons - y
        output_delta = output_error * self.sigmoid_derivative(self.output_input)

        # get error at hidden layer
        hidden_error = np.dot(output_delta, self.HO_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_input)

        # update weights and biases with momentum
        self.HO_velocity = self.momentum * self.HO_velocity - self.eta * np.dot(self.hidden_neurons.T, output_delta)
        self.HO_weights += self.HO_velocity

        self.OL_velocity = self.momentum * self.OL_velocity - self.eta * np.sum(output_delta, axis=0, keepdims=True)
        self.OL_bias += self.OL_velocity

        self.IH_velocity = self.momentum * self.IH_velocity - self.eta * np.dot(X.T, hidden_delta)
        self.IH_weights += self.IH_velocity

        self.HL_velocity = self.momentum * self.HL_velocity - self.eta * np.sum(hidden_delta, axis=0, keepdims=True)
        self.HL_bias += self.HL_velocity



    def get_loss(self, y_true: vector, y_pred: vector) -> float:
        return np.mean((y_true - y_pred) ** 2)


    def train(self, X: vector, y: vector, epochs: int, debug: bool=False) -> None:
        loss = np.zeros(epochs)

        for i in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y)
            
            loss[i] = self.get_loss(y, self.output_neurons)
            
        if debug:
            for i in range(epochs):
                print(f"Epoch {i+1}/{epochs}, Loss: {loss[i]:.2f}")

            plt.figure(figsize=(8, 5))
            epochs = np.arange(len(loss))

            plt.plot(epochs, loss, label='Loss', marker='o', markersize=4)
            plt.title('Loss in function of epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()


    def test(self, X: vector, y: vector, debug: bool=False) -> float:
        predictions = self.feedforward(X)
        
        # get the index of the highest probability for each row
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(y, axis=1)

        # get the number of correct predictions and calculate the accuracy
        correct_predictions = np.sum(predicted_classes == actual_classes)
        accuracy = (correct_predictions / y.shape[0]) * 100

        if debug:
            species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            
            # print the predictions and the actual classes for each row
            for i, prediction in enumerate(predictions):
                percentage = (np.max(prediction) * 100).round(2)
                predicted = species[predicted_classes[i]]
                actual = species[actual_classes[i]]
                print(f"{percentage}% that it is {predicted}. Actual: {actual}. Result: {predicted == actual}")
            
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy
    

    def train_and_test(self, X_train: vector, y_train: vector, X_test: vector, y_test: vector, epochs: int, debug: bool=False) -> None:
        train_loss = np.zeros(epochs)
        test_loss = np.zeros(epochs)

        for i in range(epochs):
            # Training step
            self.feedforward(X_train)
            self.backpropagation(X_train, y_train)
            train_loss[i] = self.get_loss(y_train, self.output_neurons)

            # Test step
            test_predictions = self.feedforward(X_test)
            test_loss[i] = self.get_loss(y_test, test_predictions)
            
        if debug:
            for i in range(epochs):
                print(f"Epoch {i+1}/{epochs}, Training loss: {train_loss[i]:.4f}, Test loss: {test_loss[i]:.4f}")

            plt.figure(figsize=(10, 6))
            epochs_range = np.arange(epochs)

            plt.plot(epochs_range, train_loss, label='Training loss', marker='o', markersize=4)
            plt.plot(epochs_range, test_loss, label='Test loss', marker='x', markersize=4)
            plt.title('Training and test loss over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()


    def export_model(self, file_path: str) -> None:
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "eta": self.eta,
            "momentum": self.momentum,
            "IH_weights": self.IH_weights.tolist(),
            "HO_weights": self.HO_weights.tolist(),
            "HL_bias": self.HL_bias.tolist(),
            "OL_bias": self.OL_bias.tolist(),
            "IH_velocity": self.IH_velocity.tolist(),
            "HO_velocity": self.HO_velocity.tolist(),
            "HL_velocity": self.HL_velocity.tolist(),
            "OL_velocity": self.OL_velocity.tolist()
        }
        
        with open(file_path, 'w') as file:
            json.dump(model_data, file, indent=4)
        print(f"Model exported to {file_path}")


    def import_model(self, file_path: str) -> None:
        with open(file_path, 'r') as file:
            model_data = json.load(file)

        self.input_size = model_data["input_size"]
        self.hidden_size = model_data["hidden_size"]
        self.output_size = model_data["output_size"]
        self.eta = model_data["eta"]
        self.momentum = model_data["momentum"]

        self.IH_weights = np.array(model_data["IH_weights"])
        self.HO_weights = np.array(model_data["HO_weights"])
        self.HL_bias = np.array(model_data["HL_bias"])
        self.OL_bias = np.array(model_data["OL_bias"])
        self.IH_velocity = np.array(model_data["IH_velocity"])
        self.HO_velocity = np.array(model_data["HO_velocity"])
        self.HL_velocity = np.array(model_data["HL_velocity"])
        self.OL_velocity = np.array(model_data["OL_velocity"])

        print(f"Model imported from {file_path}")
