import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

class e2eNetwork(nn.Module):
    def __init__(self):
        # TODO make sure the layers and activation function match the model you have trained
        super(e2eNetwork, self).__init__()
        self.fc1 = nn.Linear(33, 256)  # Input layer (19 inputs)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4) # Output layer (number of motors)

    def forward(self, x):
        x = F.elu(self.fc1(x)) # Make sure to add correct activation functions
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

def convert_network():
    # Load the state dictionary. TODO: Change the model name to match the one you have trained
    state_dict = torch.load(sys.argv[1], map_location=torch.device('cpu'))
    
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
        "fc4.weight": model_state_dict["a2c_network.mu.weight"],
        "fc4.bias": model_state_dict["a2c_network.mu.bias"]
    }

    # Initialize the e2eNetwork model
    e2e_model = e2eNetwork()
    e2e_model.load_state_dict(mapped_state_dict)
    e2e_model.eval()

    # Test the model
    sample_input = torch.rand(1, 33)
    pytorch_output = e2e_model(sample_input)

    torch.onnx.export(
        e2e_model,
        sample_input,
        f"{sys.argv[2]}.onnx",
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