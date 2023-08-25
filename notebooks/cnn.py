import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# 自定义数据集类
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
epochs = 10
learning_rate = 0.001

# 创建训练集和数据加载器data_dir = picture_brand,pictre_spec或picture_class
train_dataset = ImageDataset(data_dir='picture_brand', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), "model_brand.pth")

# 使用模型进行预测
test_image = Image.open("brand_test.jpg").convert('RGB')
test_image = transform(test_image).unsqueeze(0).to(device)

model.load_state_dict(torch.load("model_brand.pth"))
model.eval()
with torch.no_grad():
    output = model(test_image)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = train_dataset.classes[predicted_idx.item()]

print("Predicted Brand:", predicted_label)
