import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
        self.transform = transform
        self.classes = self.get_classes()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
        label_index = self.classes.index(label)

        return image, label_index

    def get_classes(self):
        classes = []
        for image_path in self.image_paths:
            label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
            if label not in classes:
                classes.append(label)
        return classes


# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化图像数据
])

# 设置训练参数
batch_size = 32
epochs = 20
learning_rate = 0.001

# 创建训练集和数据加载器data_dir = picture_class,pictre_spec或picture_class
train_dataset = ImageDataset(data_dir='picture_spec', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
num_classes = len(train_dataset.classes)  # 根据训练时的类别数量进行设定
model.fc = nn.Linear(num_features, num_classes)

# 加载训练好的模型参数
model.load_state_dict(torch.load("model_spec.pth"))
model.eval()

# 使用模型进行预测
test_image = Image.open("spec_test.jpg").convert('RGB')
test_image = transform(test_image).unsqueeze(0)

with torch.no_grad():
    output = model(test_image)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = train_dataset.classes[predicted_idx.item()]

print("Predicted Spec:", predicted_label)
