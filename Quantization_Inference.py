import os
import time
import torch
import torch_tensorrt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image, ImageDraw, ImageFont
from torch import nn


# AlexNet 定义（保持与训练时一致）
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载原始模型
model_fp32 = AlexNet().to(device)
model_fp32.load_state_dict(torch.load("models/alexnet_cifar10_DataAugmented.pth", map_location=device))
model_fp32.eval()

# 转换为 FP16 量化模型
example_input = torch.randn((1, 3, 224, 224)).to(device)
model_trt = torch_tensorrt.compile(model_fp32,
    inputs=[torch_tensorrt.Input(example_input.shape)],
    enabled_precisions={torch.float16},
    truncation=True
)

print("FP16 量化模型构建完成。")

# CIFAR-10 类别
dummy_dataset = CIFAR10(root='./data', train=True, download=True)
classes = dummy_dataset.classes

# 变换 & 加载数据
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])
test_dataset = datasets.ImageFolder(root='data/real_test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建保存目录
save_root = "data/real_test_quantization"
os.makedirs(save_root, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(save_root, class_name), exist_ok=True)

# 预测并标注
print("开始推理与标注...\n")

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)

    # 原始模型推理时间
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model_fp32(images)
    torch.cuda.synchronize()
    orig_time = (time.time() - t0) * 1000  # ms

    # 量化模型推理 + 时间
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        outputs = model_trt(images.half())  # 注意 FP16 输入
    torch.cuda.synchronize()
    quant_time = (time.time() - t1) * 1000  # ms

    probs = torch.softmax(outputs, dim=1)
    topk_probs, topk_indices = torch.topk(probs, 4)
    topk_probs = topk_probs[0].cpu().numpy()
    topk_indices = topk_indices[0].cpu().numpy()
    topk_classes = [classes[idx] for idx in topk_indices]

    # 读取原图
    image_path, _ = test_dataset.samples[i]
    original_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)

    # 字体
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # 写文字
    y = 5
    draw.text((5, y), "Model: FP16 Quantized", fill="red", font=font); y += 22
    draw.text((5, y), f"Orig Time: {orig_time:.2f} ms", fill="black", font=font); y += 22
    draw.text((5, y), f"Quant Time: {quant_time:.2f} ms", fill="black", font=font); y += 22
    draw.text((5, y), f"Top-4:", fill="blue", font=font); y += 22
    for rank in range(4):
        text = f"{rank+1}. {topk_classes[rank]}  {topk_probs[rank]*100:.1f}%"
        draw.text((5, y), text, fill="blue", font=font)
        y += 22

    # 保存图片
    filename = os.path.basename(image_path)
    folder_name = os.path.basename(os.path.dirname(image_path))
    save_path = os.path.join(save_root, folder_name, filename)
    original_img.save(save_path)

    print(f"[{i+1}/{len(test_dataset)}] 图像已保存至: {save_path}")

print("\n全部完成！量化模型的推理速度和预测结果已成功写入图片。")
