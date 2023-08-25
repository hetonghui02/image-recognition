import os
import pandas as pd
import requests

# 创建保存图片的文件夹
if not os.path.exists('picture_brand'):
    os.makedirs('picture_brand')

if not os.path.exists('picture_spec'):
    os.makedirs('picture_spec')

if not os.path.exists('picture_class'):
    os.makedirs('picture_class')

# 读取Excel文件
df = pd.read_excel('商品标题与原图.xlsx')


# 创建字典用于记录品牌图标出现的次数
brand_count = {}
spec_count = {}
class_count = {}

# 遍历每一行数据
for index, row in df.iterrows():
    img_url = row['img_head']  # 图片链接
    brand = row['brand']  # 品牌名称
    spec =row['spec'] # 型号名称
    class_of_picture =row['class']
    # 发送请求获取图片数据
    response = requests.get(img_url)
    if response.status_code == 200:
        # 获取品牌图标出现的次数
        count = brand_count.get(brand, 0) + 1
        brand_count[brand] = count
        # 提取图片文件名
        img_name = f"{brand}_{count}.jpg"
        # 拼接保存路径
        save_path = os.path.join('picture_brand', img_name)
        # 保存图片
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"图片保存成功: {save_path}")
    else:
        print(f"图片下载失败: {img_url}")

for index, row in df.iterrows():
    img_url = row['img_head']  # 图片链接
    brand = row['brand']  # 品牌名称
    spec = row['spec']  # 型号名称
    class_of_picture = row['class']
    # 发送请求获取图片数据
    response = requests.get(img_url)
    if response.status_code == 200:
        # 获取品牌图标出现的次数
        count = spec_count.get(spec, 0) + 1
        spec_count[spec] = count
        # 提取图片文件名
        img_name = f"{spec}_{count}.jpg"
        # 拼接保存路径
        save_path = os.path.join('picture_spec', img_name)
        # 保存图片
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"图片保存成功: {save_path}")
    else:
        print(f"图片下载失败: {img_url}")

for index, row in df.iterrows():
    img_url = row['img_head']  # 图片链接
    brand = row['brand']  # 品牌名称
    spec = row['spec']  # 型号名称
    class_of_picture = row['class']
    # 发送请求获取图片数据
    response = requests.get(img_url)
    if response.status_code == 200:
        # 获取品牌图标出现的次数
        count = class_count.get(brand, 0) + 1
        class_count[brand] = count
        # 提取图片文件名
        img_name = f"{class_of_picture}_{count}.jpg"
        # 拼接保存路径
        save_path = os.path.join('picture_class', img_name)
        # 保存图片
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"图片保存成功: {save_path}")
    else:
        print(f"图片下载失败: {img_url}")

print("图片下载完成")
