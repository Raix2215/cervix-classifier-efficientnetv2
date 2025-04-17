import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import Config

def visualize_cam_for_test_with_errors(model, test_loader, num_samples=None, apply_segmentation=False):
    """独立的CAM可视化函数，专门输出预测错误的样本"""
    model.eval()
    
    # 设置matplotlib支持中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建保存目录 - 错误预测的文件会放在不同文件夹
    save_dir = os.path.join(Config.MODEL_DIR, "error_predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    # 按照真实标签创建子文件夹
    lesion_errors_dir = os.path.join(save_dir, "true_lesion_errors")  # 真实是病变但预测错误
    normal_errors_dir = os.path.join(save_dir, "true_normal_errors")  # 真实是正常但预测错误
    
    os.makedirs(lesion_errors_dir, exist_ok=True)
    os.makedirs(normal_errors_dir, exist_ok=True)
    
    # 定义类别名称
    class_names = ['Lesion', 'Normal']  # 根据你的实际类别进行修改
    
    # 统计信息
    errors_count = 0
    processed_samples = 0
    results = []
    
    with torch.no_grad():  # 评估模式，无需梯度
        total_samples = len(test_loader.dataset)
        print(f"开始处理所有测试样本，总数: {total_samples}")

        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="处理测试样本")):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # 获取模型预测
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # 找出错误预测
            incorrect_mask = (preds != labels)
            if not incorrect_mask.any():
                processed_samples += inputs.size(0)
                continue
                
            # 仅处理预测错误的样本
            incorrect_indices = torch.nonzero(incorrect_mask).squeeze(1)
            
            for idx in incorrect_indices:
                i = idx.item()
                img = inputs[i].cpu()
                label = labels[i].item()
                pred = preds[i].item()
                prob = probabilities[i][pred].item()
                
                # 获取类别名称
                true_label_name = class_names[label]
                pred_label_name = class_names[pred]
                
                # 选择保存目录
                save_subdir = lesion_errors_dir if label == 0 else normal_errors_dir
                
                # 准备原始图像
                rgb_img = img.numpy().transpose(1, 2, 0)
                rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                rgb_img = np.clip(rgb_img, 0, 1)
                
                # 保存原始图像
                orig_path = os.path.join(save_subdir, f"error_{errors_count}_original.jpg")
                cv2.imwrite(orig_path, cv2.cvtColor((rgb_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # 生成CAM
                input_tensor = img.unsqueeze(0).to(Config.DEVICE)
                target_layer = model.conv_head
                cam = ScoreCAM(model=model, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                # 创建CAM可视化
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_path = os.path.join(save_subdir, f"error_{errors_count}_true_{true_label_name}_pred_{pred_label_name}_{prob:.4f}.jpg")
                cv2.imwrite(cam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                
                # 记录结果
                results.append({
                    'error_idx': errors_count,
                    'true_label': true_label_name,
                    'pred_label': pred_label_name,
                    'probability': prob,
                    'cam_path': cam_path
                })
                
                errors_count += 1
            
            processed_samples += inputs.size(0)
    
    print(f"预测错误样本总数: {errors_count}")
    print(f"错误样本CAM可视化结果已保存到: {save_dir}")
    
    # 打印部分错误案例
    if results:
        print("\n部分错误预测案例:")
        for i, result in enumerate(results[:min(10, len(results))]):
            print(f"样本 {i}: 真实标签={result['true_label']}, 错误预测为={result['pred_label']}, 错误概率={result['probability']:.4f}")
    
    return save_dir, results

def test_model_and_analyze_errors(model, test_loader, criterion):
    """完整的模型测试和错误分析函数"""
    from utils.metrics import validate
    
    print("开始模型测试和错误预测分析...")
    
    # 1. 基本测试评估
    model.eval()
    test_metrics = validate(model, test_loader, criterion)
    
    print("\n测试结果摘要:")
    print(f"准确率: {test_metrics['acc']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # 2. 错误案例分析和可视化
    print("\n开始分析预测错误的样本...")
    error_cam_dir, error_results = visualize_cam_for_test_with_errors(
        model,
        test_loader,
        num_samples=None,  # 处理所有样本
        apply_segmentation=False
    )
    
    # 3. 错误分析统计
    if error_results:
        # 按真实标签统计错误
        true_lesion_errors = sum(1 for r in error_results if r['true_label'] == 'Lesion')
        true_normal_errors = sum(1 for r in error_results if r['true_label'] == 'Normal')
        
        # 计算各类错误率
        total_samples = len(test_loader.dataset)
        total_errors = len(error_results)
        error_rate = total_errors / total_samples
        
        print("\n错误分析统计:")
        print(f"总样本数: {total_samples}")
        print(f"总错误数: {total_errors} (错误率: {error_rate:.2%})")
        print(f"真实为病变但预测为正常的错误: {true_lesion_errors} ({true_lesion_errors/total_errors:.2%} 的错误)")
        print(f"真实为正常但预测为病变的错误: {true_normal_errors} ({true_normal_errors/total_errors:.2%} 的错误)")
    
    return test_metrics, error_results
