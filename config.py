import os
import torch

class Config:
    # 基本参数
    BATCH_SIZE = 32
    IMG_SIZE = 224  # 输入图像大小
    EPOCHS = 50
    BASE_LR = 2e-4  
    WEIGHT_DECAY = 1e-2  # 添加权重衰减
    DATA_PATH = "Dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # 不要使用多进程加载数据
    PIN_MEMORY = True  # 加速数据传输
    
    # 学习率调度参数
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    MAX_LR = 1e-3
    
    # 早停参数
    PATIENCE = 20
    
    # 模型保存路径
    MODEL_DIR = "saved_models"
    LOG_DIR = "logs_efficientnetv2"
    
    # 混合精度训练
    USE_AMP = True
    
    # EMA模型参数
    USE_EMA = False
    EMA_DECAY = 0.5
    
    # CAM相关参数
    USE_CAM = True
    CAM_METHOD = "gradcam"  # 可选: "gradcam", "gradcam++", "scorecam", "xgradcam"
    CAM_VISUALIZE_EPOCH_INTERVAL = 5  # 每隔多少个epoch可视化一次CAM
    CAM_SAVE_DIR = "cam_results"

# 创建保存模型的目录
os.makedirs(Config.MODEL_DIR, exist_ok=True)
