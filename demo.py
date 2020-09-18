import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from ssd import build_ssd
from data import VOC_CLASSES as labels
from matplotlib import pyplot as plt


module_path = os.path.abspath(os.path.join('..'))  
if module_path not in sys.path:
    sys.path.append(module_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 构建架构，指定输入图像的大小（300），和要评分的对象类别的数量（X+1类）
net = build_ssd('test', 300, 4)    # 【改1】这里改一下，如果有5类，就改成6
# 将预训练的权重加载到数据集上
net.load_weights('weights/ssd300_VOC_2000.pth')  # 【改2】这里改成你自己的模型文件

# 加载多张图像
# 【改3】改成你自己的文件夹
imgs = 'my_img/'
img_list = os.listdir(imgs)
for img in img_list:
    # 对输入图像进行预处理
    current_img = imgs + img
    image = cv2.imread(current_img)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    # 把图片设为变量
    xx = Variable(x.unsqueeze(0))
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    # 解析 查看结果

    top_k = 10

    plt.figure(figsize=(6, 6))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()

    detections = y.data
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            print(display_txt)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j += 1
    plt.imshow(rgb_image)
    plt.show()
