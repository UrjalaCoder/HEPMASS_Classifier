import torch.nn as nn

class Network(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=5):
        super().__init__()
        # Two layer network
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
        self.activation_hidden = nn.ReLU()
        self.activation_out = nn.Softmax()
        self.lossF = nn.MSELoss()

    def forward(self, input_tensor):
        hidden_zs = self.linear1(input_tensor)
        hidden_output = self.activation_hidden(hidden_zs)
        output_zs = self.linear2(hidden_output)
        output_output = self.activation_out(output_zs)
        return output_output

    def loss(self, predicted, target):
        return self.lossF(predicted, target)
