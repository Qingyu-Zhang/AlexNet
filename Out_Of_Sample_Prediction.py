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

import os
from PIL import ImageDraw, ImageFont

# 创建标注图片保存路径结构
marked_root = "data/real_test_marked"
os.makedirs(marked_root, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(marked_root, class_name), exist_ok=True)

print("开始预测并标注图片...\n")

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)

        # 获取概率前4的类别及其概率
        probs = torch.softmax(outputs, dim=1)
        topk_probs, topk_indices = torch.topk(probs, 4)

        topk_probs = topk_probs[0].cpu().numpy()
        topk_indices = topk_indices[0].cpu().numpy()
        topk_classes = [classes[idx] for idx in topk_indices]

        # 当前图片路径信息
        image_path, _ = test_dataset.samples[i]
        true_label = classes[labels.item()]
        filename = os.path.basename(image_path)
        folder_name = os.path.basename(os.path.dirname(image_path))

        # 打印输出 Top-4 预测
        print(f"Image {i+1}:")
        for rank in range(4):
            print(f"  Top {rank+1}: {topk_classes[rank]} ({topk_probs[rank]*100:.2f}%)")
        print(f"  True Label: {true_label}")
        print("-" * 40)

        # 读取原图（不经过transform）
        original_img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_img)

        # 使用字体（可指定路径或使用默认）
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 写入前4预测类别和概率
        y_offset = 5
        for rank in range(4):
            text = f"{rank+1}. {topk_classes[rank]}: {topk_probs[rank]*100:.1f}%"
            draw.text((5, y_offset), text, fill="red", font=font)
            y_offset += 22

        # 保存到对应路径
        save_path = os.path.join(marked_root, folder_name, filename)
        original_img.save(save_path)

        # 统计准确率
        if topk_indices[0] == labels.item():
            correct += 1
        total += 1


acc = 100 * correct / total
print(f"\n总测试图像数: {total}")
print(f"正确分类数: {correct}")
print(f"预测准确率: {acc:.2f}%")
