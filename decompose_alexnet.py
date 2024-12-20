import torch
from torchvision import models
from ht2 import ht2
from hosvd1 import hosvd1
from svd_decomposition import svd_decomposition

if __name__ == "__main__":
    energy = 0.8
    # Load the pretrained AlexNet model
    net = models.alexnet(pretrained=True)
    
    # Decompose the first convolutional layer (conv1)
    net.features[0] = hosvd1(net.features[0], energy)

    # Decompose the remaining convolutional layers in the 'features' block
    for i in range(1, len(net.features)):
        layer = net.features[i]
        
        # If the layer is a convolutional layer, apply decomposition
        if isinstance(layer, torch.nn.Conv2d):
            if i % 2 == 0:  # Example: Decompose even-indexed convolutions with hosvd1
                net.features[i] = hosvd1(layer, energy)
            else:  # Decompose odd-indexed convolutions with ht2
                net.features[i] = ht2(layer, energy)
    
    # Decompose the fully connected layers (classifier)
    for i in range(len(net.classifier)):
        layer = net.classifier[i]
        
        # Decompose the fully connected layers with svd_decomposition
        if isinstance(layer, torch.nn.Linear):
            net.classifier[i] = svd_decomposition(layer, energy)
    
    # Print the modified network
    print(net)
    
    # Save the decomposed model
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    torch.save(checkpoint, "alexnet_ht2.pth")
