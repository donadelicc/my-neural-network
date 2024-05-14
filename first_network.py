from torch import nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the 28x28 images to 784-dimensional vectors
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 15),  # First layer from 784 inputs to 15 hidden neurons
            nn.ReLU(),             # Activation function for non-linearity
            nn.Linear(15, 10)      # Output layer from 15 hidden neurons to 10 outputs
        )

    def forward(self, x):
        x = self.flatten(x)        # Flatten the image
        logits = self.linear_relu_stack(x)  # Pass through the network
        return logits
