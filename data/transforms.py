import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

def get_transforms(is_training=True):
    if is_training:
        # 训练集需要数据增强以提高模型泛化能力
        return A.Compose([
            # 首先进行调整大小，稍大一些以便进行随机裁剪
            A.Resize(height=224, width=224),
            # 随机裁剪到Config.IMG_SIZE×Config.IMG_SIZE，提供位置多样性
            A.RandomResizedCrop(
                height=Config.IMG_SIZE, 
                width=Config.IMG_SIZE,
                scale=(0.8, 1.0)
            ),
            # 水平和垂直翻转
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # 旋转变换
            A.RandomRotate90(p=0.5),
            # 色彩变换
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            # 亮度对比度调整
            A.RandomBrightnessContrast(p=0.5),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 转为tensor
            ToTensorV2(),
        ])
    else:
        # 验证集和测试集只需要调整大小和中心裁剪，无需数据增强
        return A.Compose([
            # 调整为稍大尺寸
            A.Resize(height=212, width=212),
            # 从中心裁剪到目标尺寸
            A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 转为tensor
            ToTensorV2(),
        ])

# 专门为边缘区域问题设计的变换，训练集可以使用
def get_transforms_edge_aware(is_training=True):
    if is_training:
        return A.Compose([
            # 更激进的裁剪，避免边缘干扰
            A.Resize(height=240, width=240),
            # 中心区域更大的随机裁剪，减少对边缘的依赖
            A.RandomResizedCrop(
                height=Config.IMG_SIZE, 
                width=Config.IMG_SIZE,
                scale=(0.7, 0.9),  # 更集中的裁剪比例
                ratio=(0.9, 1.1)   # 更接近正方形的裁剪
            ),
            # 标准数据增强
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            # 更激进的中心裁剪
            A.Resize(height=240, width=240),
            A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# 针对宫颈图像特殊处理的变换
def get_transforms_cervix_specific(is_training=True):
    if is_training:
        return A.Compose([
            # 基础尺寸调整
            A.Resize(height=224, width=224),
            # 裁剪中心区域
            A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            # 在中心裁剪基础上进行随机调整
            A.ShiftScaleRotate(
                shift_limit=0.05,  # 小幅移动
                scale_limit=0.05,  # 小幅缩放
                rotate_limit=15,   # 小角度旋转
                p=0.7
            ),
            # 颜色增强，针对医学图像特点调整
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=1.0
                ),
                A.CLAHE(clip_limit=2, p=1.0),  # 对比度受限自适应直方图均衡化
            ], p=0.7),
            # 可选：添加网格扭曲，模拟采集角度变化
            A.GridDistortion(p=0.3),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            # 调整尺寸并进行中心裁剪
            A.Resize(height=224, width=224),
            A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            # 规范化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])