import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os
'''
# 指定图片文件夹路径
picture_folder = 'picture'

# 获取图片文件夹下所有jpg格式的图片文件路径
img_paths = [os.path.join(picture_folder, file) for file in os.listdir(picture_folder) if file.endswith('.jpg')]

# 打印所有图片文件路径
for img_path in img_paths:
    print(img_path)
'''
universal_matting = pipeline(Tasks.universal_matting,model='damo/cv_unet_universal-matting')
result = universal_matting('https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/1.png')

cv2.imwrite('99还原版本balenciaga巴黎老爹鞋track30复古三代休闲鞋运动鞋_巴黎世家户外概念鞋男女款科技发光鞋增高厚底鞋透气男女鞋.jpg', result[OutputKeys.OUTPUT_IMG])
