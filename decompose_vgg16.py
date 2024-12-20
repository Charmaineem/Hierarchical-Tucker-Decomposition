import torch
from torch import nn
from torchvision import models

from ht2 import ht2
from hosvd1 import hosvd1
from svd_decomposition import svd_decomposition

if __name__ == "__main__":
    energy = 0.78

    #load and initialise net
    state_dict = torch.load("vgg16.pth", weights_only=True)
    net = models.vgg16(pretrained=False)
    net.load_state_dict(state_dict)

    for key, layer in net.features._modules.items():
        #applying hosvd decomposition for first layer
        if key == "0":
            decomposed = hosvd1(layer, energy)
            net.features._modules[key] = decomposed
            
        #applying ht2 decomposition for other layers
        if isinstance(layer, nn.modules.conv.Conv2d) and key != "0":
            decomposed = ht2(layer, energy)
            net.features._modules[key] = decomposed

    #svd for fully connected layers
    for key, layer in net.classifier._modules.items():
        if isinstance(layer, nn.modules.linear.Linear):
            decomposed = svd_decomposition(layer, energy)
            net.classifier._modules[key] = decomposed

    print(net)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    torch.save(checkpoint, "vgg16_ht2.pth")