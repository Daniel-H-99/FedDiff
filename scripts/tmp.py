import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

# *-- VGG model 읽어오기 --*
use_pretrained=True # 학습 된 파라미터 사용
net = models.vgg16(pretrained=use_pretrained)
net.eval()
# 모델 네트워크 구성 출력
print(net)

