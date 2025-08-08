import torch
import torch.nn as nn
import sys
# import torch.nn.functional as F

# TODO set number of inputs and outputs
NUM_INPUTS = 33
NUM_OUTPUTS = 4

class e2eNetwork(nn.Module):
    def __init__(self):
        # TODO make sure the layers and activation function match the model you have trained
        super(e2eNetwork, self).__init__()
        self.fc1 = nn.Linear(NUM_INPUTS, 128)  # Input layer
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 48)
        self.fc5 = nn.Linear(48, NUM_OUTPUTS) # Output layer (number of motors)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Make sure to add correct activation functions
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def convert_network():
    # Load the state dictionary.
    state_dict = torch.load("gen_ppo.pth", map_location=torch.device('cpu'))
    
    # Extract the model state dictionary
    model_state_dict = state_dict["model"]

    # Map the keys to match the e2eNetwork structure
    mapped_state_dict = {
        "fc1.weight": model_state_dict["a2c_network.actor_mlp.0.weight"],
        "fc1.bias": model_state_dict["a2c_network.actor_mlp.0.bias"],
        "fc2.weight": model_state_dict["a2c_network.actor_mlp.2.weight"],
        "fc2.bias": model_state_dict["a2c_network.actor_mlp.2.bias"],
        "fc3.weight": model_state_dict["a2c_network.actor_mlp.4.weight"],
        "fc3.bias": model_state_dict["a2c_network.actor_mlp.4.bias"],
        "fc4.weight": model_state_dict["a2c_network.actor_mlp.6.weight"],
        "fc4.bias": model_state_dict["a2c_network.actor_mlp.6.bias"],
        "fc5.weight": model_state_dict["a2c_network.mu.weight"],
        "fc5.bias": model_state_dict["a2c_network.mu.bias"]
    }

    # Initialize the e2eNetwork model
    e2e_model = e2eNetwork()
    e2e_model.load_state_dict(mapped_state_dict)
    e2e_model.eval()

    # Test the model
    sample_input = torch.rand(1, NUM_INPUTS)
    pytorch_output = e2e_model(sample_input)

    torch.onnx.export(
        e2e_model,
        sample_input,
        "gen_ppo.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Done")

if __name__ == "__main__":
    convert_network()