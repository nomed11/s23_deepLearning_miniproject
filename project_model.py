import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, kernel, skip_kernel, stride=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel[0], stride=stride, padding=kernel[1], bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel[0], stride=1, padding=kernel[1], bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=skip_kernel[0], padding=skip_kernel[1], stride=stride, bias=bias),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, in_planes, num_layers, num_blocks, kernel, skip_kernel, num_classes=10, bias=True):
        if not isinstance(num_blocks, list):
            raise Exception("num_blocks parameter should be a list of integer values")
        if num_layers != len(num_blocks):
            raise Exception("Residual layers should be equal to the length of num_blocks list")
        super(ResNet, self).__init__()
        self.kernel = kernel
        self.skip_kernel = skip_kernel
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=kernel[0], stride=1, padding=kernel[1], bias=bias)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.num_layers = num_layers
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1, bias=bias)
        for i in range(2, num_layers+1):
            setattr(self, "layer"+str(i), self._make_layer(block, 2*self.in_planes, num_blocks[i-1], stride=2, bias=bias))
        finalshape = list(getattr(self, "layer"+str(num_layers))[-1].modules())[-2].num_features
        self.multiplier = 4 if num_layers == 2 else (2 if num_layers == 3 else 1)
        self.linear = nn.Linear(finalshape, num_classes)
        self.path = "./project1_model.pt"

    def _make_layer(self, block, planes, num_blocks, stride, bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        custom_layers = []
        for stride in strides:
            custom_layers.append(block(self.in_planes, planes,self.kernel,self.skip_kernel, stride, bias))
            self.in_planes = planes
        return nn.Sequential(*custom_layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(1, self.num_layers+1):
            out = eval("self.layer" + str(i) + "(out)")
        out = F.avg_pool2d(out, 4*self.multiplier)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def saveToDisk(self):
        torch.save(self.state_dict(), self.path)

    def loadFromDisk(self):
        self.load_state_dict(torch.load(self.path))

def project1_model():
    return ResNet(BasicBlock, 32, 4, [4, 4, 4, 2],kernel=(3,1),skip_kernel=(1,0), num_classes=10, bias=True)

if __name__ == "__main__":
    model = ResNet(BasicBlock, 32, 4, [4, 4, 4, 2],kernel=(3,1),skip_kernel=(1,0), num_classes=10, bias=True)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_parameters)
    x = torch.rand(1, 3, 32, 32)
    model(x)