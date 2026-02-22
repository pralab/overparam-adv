
import torch
from src.lipestime import LipUpperBound, LipLowerBound
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model.eval()
estime = LipUpperBound(model)
result = estime.propagate(torch.randn(1, 3, 32, 32))
lip=1.0
for node in estime.graph.nodes:
    lip = getattr(node, 'lip', lip)
    print(f"Node: {node.name}, op: {node.op}, target: {node.target}, lip: {lip:.3e}")
print(result)

print("Starting lower bound estimation")
testset = CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


normalized_model = torch.nn.Sequential(
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    model
)
normalized_model.eval()
normalized_model.to('cuda')
lower_estime = LipLowerBound(normalized_model)
lower_bound= 0.0
for i, (inputs, labels) in enumerate(testloader):
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    if i == 10:
        break
    lip_estimate = lower_estime.estimate(inputs, labels, n_aug=100) # reduce n_aug for faster estimation during testing
    lower_bound = max(lower_bound, lip_estimate)
    print(f"Batch {i}, Lipschitz lower bound estimate: {lower_bound:.3e}")