import os
import numpy as np
import matplotlib.pyplot as plt
import json

class ModelComparator:
    """模型对比工具类"""
    
    @staticmethod
    def load_test_results(output_dir):
        """加载所有模型的测试结果"""
        results = {}
        models = ['unet3d', 'transformer', 'hybrid']
        
        for model in models:
            result_path = os.path.join(output_dir, model, 'test_results.txt')
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        lines = f.readlines()
                    
                    model_result = {}
                    for line in lines[2:]:  # 跳过前两行标题
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            key = key.strip()
                            value = float(value.strip())
                            model_result[key] = value
                    
                    results[model] = model_result
                except Exception as e:
                    print(f"加载 {model} 测试结果失败: {e}")
                    results[model] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def load_evaluation_results(output_dir):
        """加载所有模型的评估结果"""
        results = {}
        models = ['unet3d', 'transformer', 'hybrid']
        
        for model in models:
            result_path = os.path.join(output_dir, model, 'evaluation_results.json')
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        model_result = json.load(f)
                    results[model] = model_result
                except Exception as e:
                    print(f"加载 {model} 评估结果失败: {e}")
                    results[model] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def plot_test_metrics(test_results, save_path=None):
        """绘制测试指标对比图"""
        models = list(test_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # 准备数据
        data = {metric: [] for metric in metrics}
        for model in models:
            if 'error' not in test_results[model]:
                for metric in metrics:
                    if metric in test_results[model]:
                        data[metric].append(test_results[model][metric])
                    else:
                        data[metric].append(0)
            else:
                for metric in metrics:
                    data[metric].append(0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(models, data[metric])
            ax.set_title(metric)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            
            # 在柱状图上添加数值
            for j, value in enumerate(data[metric]):
                ax.text(j, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle('模型测试指标对比', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"测试指标对比图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_evaluation_metrics(evaluation_results, save_path=None):
        """绘制评估指标对比图"""
        models = list(evaluation_results.keys())
        metrics = ['success_rate', 'avg_path_length', 'avg_smoothness', 'avg_inference_time_ms']
        
        # 准备数据
        data = {metric: [] for metric in metrics}
        for model in models:
            if 'error' not in evaluation_results[model]:
                for metric in metrics:
                    if metric in evaluation_results[model]:
                        data[metric].append(evaluation_results[model][metric])
                    else:
                        data[metric].append(0)
            else:
                for metric in metrics:
                    data[metric].append(0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(models, data[metric])
            ax.set_title(metric)
            
            # 根据指标调整Y轴范围
            if metric == 'success_rate':
                ax.set_ylim(0, 1.1)
            elif metric == 'avg_inference_time_ms':
                ax.set_ylim(0, max(data[metric]) * 1.2 if data[metric] else 100)
            else:
                ax.set_ylim(0, max(data[metric]) * 1.2 if data[metric] else 20)
            
            ax.grid(True)
            
            # 在柱状图上添加数值
            for j, value in enumerate(data[metric]):
                if metric == 'avg_inference_time_ms':
                    ax.text(j, value + 0.5, f'{value:.4f}', ha='center', va='bottom')
                else:
                    ax.text(j, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle('模型评估指标对比', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"评估指标对比图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_comparison_report(test_results, evaluation_results, save_path=None):
        """生成对比报告"""
        report = """# 三维路径规划模型对比报告

## 一、测试指标对比

| 模型类型 | 准确率 | 精确率 | 召回率 | F1分数 |
|---------|-------|-------|-------|-------|
"""
        
        # 添加测试指标
        for model in sorted(test_results.keys()):
            if 'error' not in test_results[model]:
                report += f"| {model:<8} | {test_results[model].get('accuracy', 0):<5.4f} | {test_results[model].get('precision', 0):<5.4f} | {test_results[model].get('recall', 0):<5.4f} | {test_results[model].get('f1', 0):<5.4f} |\n"
            else:
                report += f"| {model:<8} | {'错误':<5} | {'错误':<5} | {'错误':<5} | {'错误':<5} |\n"
        
        report += """

## 二、评估指标对比

| 模型类型 | 成功率 | 平均路径长度 | 平均平滑度 | 平均推理时间(ms) |
|---------|-------|-------------|-----------|----------------|
"""
        
        # 添加评估指标
        for model in sorted(evaluation_results.keys()):
            if 'error' not in evaluation_results[model]:
                report += f"| {model:<8} | {evaluation_results[model].get('success_rate', 0):<5.4f} | {evaluation_results[model].get('avg_path_length', 0):<11.4f} | {evaluation_results[model].get('avg_smoothness', 0):<9.4f} | {evaluation_results[model].get('avg_inference_time_ms', 0):<16.4f} |\n"
            else:
                report += f"| {model:<8} | {'错误':<5} | {'错误':<11} | {'错误':<9} | {'错误':<16} |\n"
        
        report += """

## 三、模型比较分析

### 3.1 测试指标分析

- **准确率**：
- **精确率**：
- **召回率**：
- **F1分数**：

### 3.2 评估指标分析

- **成功率**：
- **平均路径长度**：
- **平均平滑度**：
- **平均推理时间**：

### 3.3 综合评价

"""
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"对比报告已保存至: {save_path}")
        else:
            print(report)
    
    @staticmethod
    def compare_all_models(output_dir='./results', save_plots=True):
        """对比所有模型"""
        print("开始对比所有模型...")
        
        # 加载测试结果
        test_results = ModelComparator.load_test_results(output_dir)
        
        # 加载评估结果
        evaluation_results = ModelComparator.load_evaluation_results(output_dir)
        
        # 绘制测试指标对比图
        if save_plots:
            ModelComparator.plot_test_metrics(
                test_results, 
                save_path=os.path.join(output_dir, 'test_metrics_comparison.png')
            )
        else:
            ModelComparator.plot_test_metrics(test_results)
        
        # 绘制评估指标对比图
        if save_plots:
            ModelComparator.plot_evaluation_metrics(
                evaluation_results, 
                save_path=os.path.join(output_dir, 'evaluation_metrics_comparison.png')
            )
        else:
            ModelComparator.plot_evaluation_metrics(evaluation_results)
        
        # 生成对比报告
        ModelComparator.generate_comparison_report(
            test_results, 
            evaluation_results, 
            save_path=os.path.join(output_dir, 'models_comparison_report.md')
        )
        
        print("\n模型对比完成！")

if __name__ == "__main__":
    ModelComparator.compare_all_models()