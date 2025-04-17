import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_roc_curve(y_true, y_score, model_name="模型", save_path=None, show=True, class_labels=None):
    """
    绘制ROC曲线
    
    参数:
    y_true: 真实标签 (0/1)
    y_score: 模型对正类的预测概率
    model_name: 模型名称
    save_path: 保存图像的路径，如果为None则不保存
    show: 是否显示图像
    class_labels: 类别标签列表，例如['正常', '病变']
    
    返回:
    auc_value: ROC曲线下面积
    fpr: 假阳性率
    tpr: 真阳性率
    thresholds: 对应的阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return auc_value, fpr, tpr, thresholds

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, show=True):
    """
    绘制混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    class_names: 类别名称列表
    save_path: 保存路径
    show: 是否显示
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm

def plot_comparative_roc(models_data, save_path=None, show=True):
    """
    在同一图中绘制多个模型的ROC曲线比较
    
    参数:
    models_data: 字典，键为模型名称，值为元组(y_true, y_score)
    save_path: 保存图像的路径
    show: 是否显示图像
    
    返回:
    results: 字典，包含每个模型的AUC值
    """
    plt.figure(figsize=(10, 8))
    results = {}
    
    for model_name, (y_true, y_score) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = auc(fpr, tpr)
        results[model_name] = auc_value
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_value:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('模型ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较ROC曲线已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return results

def find_optimal_threshold(fpr, tpr, thresholds):
    """
    找到最佳决策阈值（尤登指数最大值点）
    
    参数:
    fpr: 假阳性率
    tpr: 真阳性率
    thresholds: 阈值数组
    
    返回:
    optimal_threshold: 最佳阈值
    j_value: 最佳尤登指数值
    """
    # 计算尤登指数(Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, j_scores[optimal_idx]