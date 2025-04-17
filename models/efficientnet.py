import torch
import torch.nn as nn
from models.components import ConvBNAct, FusedMBConv, MBConv

# EfficientNetV2-S配置
v2_s_cfg = [
    # 类型  重复次数 输出通道 步幅 expand_ratio se_ratio kernel
    ('fused', 2,  24, 1, 1, None, 3),
    ('fused', 4,  48, 2, 4, None, 3),
    ('fused', 4,  64, 2, 4, None, 3),
    ('mbconv', 6, 128, 2, 4, 0.25, 3),
    ('mbconv', 9, 160, 1, 6, 0.25, 3),
    ('mbconv',15, 256, 2, 6, 0.25, 3),
]

def efficientnetv2_s(num_classes=1000, dropout_rate=0.2):
    return EfficientNetV2(v2_s_cfg, num_classes, dropout_rate)

# EfficientNetV2 模型
class EfficientNetV2(nn.Module):
    def __init__(self, cfg, num_classes=1000, dropout_rate=0.2):
        super().__init__()
        self.stem = ConvBNAct(3, 24, kernel=3, stride=2)
        layers = []
        in_c = 24

        # 按配置逐层堆叠
        for t, repeats, out_c, stride, exp_ratio, se_ratio, k in cfg:
            for i in range(repeats):
                layers.append(
                    FusedMBConv(in_c, out_c, stride if i == 0 else 1, exp_ratio, se_ratio, k)
                if t == 'fused'
                    else MBConv(in_c, out_c, stride if i == 0 else 1, exp_ratio, se_ratio, k)
                )
                in_c = out_c

        self.blocks = nn.Sequential(*layers)
         # 修改为单独的组件，便于 CAM 访问最后的特征图
        self.conv_head = ConvBNAct(in_c, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        # 添加初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 权重初始化，提高模型的稳定性
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)  # 这是 CAM 将要使用的特征图
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    # 添加获取最后卷积层的方法，用于 CAM
    def get_cam_layer(self):
        return self.conv_head
