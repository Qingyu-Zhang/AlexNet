import sys

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
model.load_state_dict(torch.load("models/alexnet_cifar10_DataAugmented.pth", map_location=device))
model.eval()

# 设置 transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# 加载自建的测试集
test_dataset = datasets.ImageFolder(root='data/real_test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预测并评估
correct = 0
total = 0
classes = test_dataset.classes  # 自动读取文件夹名顺序


from torchvision.datasets import CIFAR10
# 不需要下载，只获取类名
dummy_dataset = CIFAR10(root='./data', train=True, download=True)
cifar10_classes = dummy_dataset.classes
# 判断是否一致
if classes == cifar10_classes:
    print("\n顺序一致，可以安全使用！")
else:
    print("\n类别顺序不一致！请重命名或调整文件夹顺序！")
    sys.exit(1)



print("开始预测...\n")

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        pred_label = classes[predicted.item()]
        true_label = classes[labels.item()]
        print(f"Image {i+1}: Predicted: {pred_label} | True: {true_label}")

        if predicted.item() == labels.item():
            correct += 1
        total += 1

acc = 100 * correct / total
print(f"\n总测试图像数: {total}")
print(f"正确分类数: {correct}")
print(f"预测准确率: {acc:.2f}%")
