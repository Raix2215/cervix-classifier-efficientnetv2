import torch
import torch.nn as nn

# 1. 基础模块：卷积 + BN + 激活
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.1)  # 调整BN的动量参数
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# SE 模块（通道注意力）
class SqueezeExcite(nn.Module):
    def __init__(self, in_c, se_ratio=0.25):
        super().__init__()
        reduced_c = max(1, int(in_c * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, reduced_c, 1)
        self.fc2 = nn.Conv2d(reduced_c, in_c, 1)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y

# 2. Fused-MBConv 模块
class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio, se_ratio=None, kernel=3):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.use_skip = stride == 1 and in_c == out_c

        # Expand + Fused Conv（当 expand_ratio > 1 时）
        if expand_ratio != 1:
            self.conv1 = ConvBNAct(in_c, mid_c, kernel, stride)
            self.conv2 = ConvBNAct(mid_c, out_c, 1, act=False)
        else:
            self.conv1 = ConvBNAct(in_c, out_c, kernel, stride, act=False)
            self.conv2 = None

        # SE 模块
        self.se = SqueezeExcite(out_c, se_ratio) if se_ratio else nn.Identity()
        
        # 添加随机丢弃模块，提高泛化能力
        self.drop_path = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        if self.conv2:
            x = self.conv2(x)
        x = self.se(x)
        x = self.drop_path(x)
        return x + identity if self.use_skip else x

# 3. MBConv 模块
class MBConv(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio, se_ratio, kernel=3):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.use_skip = stride == 1 and in_c == out_c

        # Expand（当 expand_ratio > 1 时）
        self.expand = ConvBNAct(in_c, mid_c, 1) if expand_ratio != 1 else nn.Identity()

        # Depthwise Conv
        self.dwconv = ConvBNAct(mid_c, mid_c, kernel, stride, groups=mid_c)
        
        # SE 模块
        self.se = SqueezeExcite(mid_c, se_ratio) if se_ratio else nn.Identity()

        # Squeeze
        self.pwconv = ConvBNAct(mid_c, out_c, 1, act=False)
        
        # 添加随机丢弃路径
        self.drop_path = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = self.pwconv(x)
        x = self.drop_path(x)
        return x + identity if self.use_skip else x

# 实现EMA模型更新 - 模型集成技术
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        import copy
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        
        # 锁定参数，不参与梯度计算
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
                
    def to(self, device):
        self.device = device
        self.ema.to(device)
        return self
