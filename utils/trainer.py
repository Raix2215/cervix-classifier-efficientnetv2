import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from utils.cam import CAMVisualizer
from utils.metrics import validate

# 学习率预热和余弦衰减调度器
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 学习率线性预热
            return epoch / warmup_epochs
        else:
            # 余弦衰减
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (1 - min_lr) * (1 + np.cos(np.pi * progress))
            
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用标签平滑的交叉熵损失 - 减少过拟合
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        
        # 创建平滑标签
        with torch.no_grad():
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = -(target_smooth * log_preds).sum(dim=1).mean()
        return loss

def train_epoch(model, loader, optimizer, criterion, scaler, ema_model=None, visualize_cam=False, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用进度条跟踪训练过程
    progress_bar = tqdm(loader, desc="Training")

    # 如果需要可视化 CAM
    if visualize_cam and epoch % Config.CAM_VISUALIZE_EPOCH_INTERVAL == 0:
        cam_visualizer = CAMVisualizer(model, cam_method=Config.CAM_METHOD)
        os.makedirs(os.path.join(Config.MODEL_DIR, f"cam_epoch_{epoch}"), exist_ok=True)
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        # 自动混合精度训练
        with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 梯度缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 更新EMA模型
        if ema_model is not None:
            ema_model.update(model)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': correct / total
        })
        
        # 可视化 CAM
        if visualize_cam and epoch % Config.CAM_VISUALIZE_EPOCH_INTERVAL == 0 and batch_idx % 50 == 0:
            for i in range(min(4, inputs.size(0))):  # 可视化批次中的前4个图像
                img_tensor = inputs[i].detach().cpu()
                label = labels[i].item()
                pred = predicted[i].item()
                
                # 生成CAM并保存
                save_path = os.path.join(
                    Config.MODEL_DIR, 
                    f"cam_epoch_{epoch}", 
                    f"batch_{batch_idx}_img_{i}_true_{label}_pred_{pred}.jpg"
                )
                
                cam_visualizer.visualize(
                    img_tensor, 
                    target_category=pred,
                    save_path=save_path
                )
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, ema_model=None, writer=None):
    best_val_metrics = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0}
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    # 设置早停
    from utils.early_stopping import EarlyStopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        path=os.path.join(Config.MODEL_DIR, 'best_model.pt')
    )
    
    start_time = time.time()
    
    for epoch in range(Config.EPOCHS):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, ema_model,
            visualize_cam=Config.USE_CAM, epoch=epoch
        )
        
        # 验证阶段 - 使用EMA模型如果可用
        val_model = ema_model.ema if ema_model else model
        val_metrics = validate(val_model, val_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['lr'].append(current_lr)
        
        # 记录TensorBoard指标
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/Val', val_metrics['acc'], epoch)
            writer.add_scalar('Precision/Val', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/Val', val_metrics['recall'], epoch)
            writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # 每5个epoch记录模型权重分布
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        
        # 打印详细信息
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # 保存最佳模型 (基于F1分数)
        if val_metrics['f1'] > best_val_metrics['f1']:
            best_val_metrics = val_metrics
            
            # 保存EMA模型如果可用，否则保存普通模型
            best_model = ema_model.ema if ema_model else model
            torch.save(best_model, os.path.join(Config.MODEL_DIR, "Cervix_Classifier_best.pth"))
            print(f"新的最佳模型已保存 (F1={val_metrics['f1']:.4f})")
        
        # 早停检查
        early_stopping(val_metrics['f1'], model)
        if early_stopping.early_stop:
            print("触发早停！")
            break
    
    # 输出总训练时间
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time/60:.2f} 分钟")
    
    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(Config.MODEL_DIR, 'training_history.csv'), index=False)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_DIR, 'training_curves.png'))
    
    return best_val_metrics
