import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

from config import Config

# CAM 工具类实现
class CAMVisualizer:
    def __init__(self, model, target_layer=None, cam_method="gradcam"):
        self.model = model
        if target_layer is None:
            # 默认使用模型的最后一个卷积层
            self.target_layer = model.get_cam_layer()
        else:
            self.target_layer = target_layer
        
        methods = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "scorecam": ScoreCAM,  # 较慢但不需要梯度
            "xgradcam": XGradCAM,
            "ablationcam": AblationCAM,  # 非常慢，但理论上更精确
            "eigencam": EigenCAM,  # 无需类别，利用主成分分析
            "hirescam": HiResCAM,  # 更高分辨率的 GradCAM
        }
        
        self.cam = methods[cam_method.lower()](
            model=model, 
            target_layers=[self.target_layer],
        )
    
    def generate_cam(self, input_tensor, target_category=None, aug_smooth=False, eigen_smooth=False):
        if target_category is None:
            # 预测最可能的类别
            with torch.no_grad():
                output = self.model(input_tensor)
                target_category = output.argmax(dim=1).item()
        
        # 这是可选的，使用增强平滑和特征值平滑可以改善 CAM 质量
        targets = [ClassifierOutputTarget(target_category)]
        
        # 生成 CAM
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth
        )
        grayscale_cam = grayscale_cam[0, :]  # 取第一个图像（如果是批处理）
        
        return grayscale_cam, target_category
    
    def visualize(self, img_tensor, target_category=None, alpha=0.5, save_path=None):
        # 确保图像是 CPU 上的 numpy 数组
        if isinstance(img_tensor, torch.Tensor):
            if img_tensor.is_cuda:
                img_tensor = img_tensor.cpu()
            
            # 处理预处理过的图像
            img_np = img_tensor.squeeze(0).numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = img_tensor
            
        # 生成 CAM
        grayscale_cam, predicted_class = self.generate_cam(
            input_tensor=img_tensor.unsqueeze(0).to(Config.DEVICE) if isinstance(img_tensor, torch.Tensor) else img_tensor,
            target_category=target_category
        )
        
        # 可视化
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=alpha)
        
        # 保存或返回
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        return cam_image, grayscale_cam, predicted_class

def visualize_cam_for_test(model, test_loader, num_samples=20):
    """独立的CAM可视化函数，避免梯度计算问题"""
    model.eval()
    
    # 创建保存目录
    save_dir = os.path.join(Config.MODEL_DIR, "cam_results")
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取一些测试样本
    images_to_visualize = []
    labels_to_visualize = []
    preds_to_visualize = []
    probs_to_visualize = []  # 添加存储概率的列表
    # 定义类别名称
    class_names = ['Lesion', 'Normal']  # 根据你的实际类别进行修改

    with torch.no_grad():  # 评估模式，无需梯度
        for inputs, labels in test_loader:
            if len(images_to_visualize) >= num_samples:
                break
                
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 获取概率
            _, preds = torch.max(outputs, 1)
            
            for i in range(min(len(inputs), num_samples - len(images_to_visualize))):
                images_to_visualize.append(inputs[i].cpu())
                labels_to_visualize.append(labels[i].item())
                preds_to_visualize.append(preds[i].item())

                # 获取预测类别的概率值
                pred_prob = probabilities[i][preds[i].item()].item()
                probs_to_visualize.append(round(pred_prob, 7))
    
    # 为选定的样本生成CAM
    target_layer = model.conv_head  # 使用最后的卷积层

    results = []  # 存储结果
    
    for i, (img, label, pred, prob) in enumerate(zip(images_to_visualize, labels_to_visualize, preds_to_visualize, probs_to_visualize)):
        # 获取类别名称
        true_label_name = class_names[label]
        pred_label_name = class_names[pred]

        # 准备输入
        input_tensor = img.unsqueeze(0).to(Config.DEVICE)
        rgb_img = img.numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # 使用ScoreCAM替代GradCAM，避免梯度要求
        cam = ScoreCAM(model=model, target_layers=[target_layer])
        
        # 生成CAM (ScoreCAM不需要梯度)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # 创建可视化
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 保存结果
        save_path = os.path.join(save_dir, f"sample_{i}_true_{label}_pred_{pred}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        # 也保存原始图像作为参考
        orig_path = os.path.join(save_dir, f"sample_{i}_original.jpg")
        cv2.imwrite(orig_path, cv2.cvtColor((rgb_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 添加到结果列表
        results.append({
            'sample_index': i,
            'true_label': true_label_name,
            'pred_label': pred_label_name,
            'probability': prob,
            'cam_path': save_path
        })
    
    print(f"CAM可视化结果已保存到: {save_dir}")

    # 打印部分预测结果作为示例
    for i, result in enumerate(results[:20]):  
        print(f"样本 {i}: 真实标签={result['true_label']}, 预测标签={result['pred_label']}, 概率={result['probability']}")

    return save_dir, results

