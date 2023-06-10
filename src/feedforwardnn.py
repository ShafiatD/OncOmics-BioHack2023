import torch
import torch.nn as nn

# Define the neural network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the input size, hidden size, and output size
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the feedforward neural network
model = FeedforwardNN(input_size, hidden_size, output_size)

# Generate random input data
input_data = torch.randn(1, input_size)

# Forward pass
output = model(input_data)

print("Output:", output)
