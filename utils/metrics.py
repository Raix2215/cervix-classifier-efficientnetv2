import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 存储概率用于计算AUC
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 获取概率
            probs = F.softmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 存储正类的概率
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 计算其他评估指标
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
