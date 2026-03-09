# src/core/experiment_manager.py
import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ExperimentManager:
    """实验管理器 - 支持断点续传"""
    
    def __init__(self, experiments_root: str = "output/experiments"):
        self.experiments_root = os.path.abspath(experiments_root)
        os.makedirs(self.experiments_root, exist_ok=True)
        self.current_experiment_id: Optional[str] = None
        self.current_experiment_dir: Optional[str] = None
    
    def create_experiment(self, task: str, domain: str, config: Any, 
                         resume: Optional[str] = None) -> str:
        """创建新实验或恢复现有实验"""
        
        # 处理恢复训练
        if resume:
            exp_id = self._resolve_resume_experiment(task, domain, resume)
            if exp_id:
                self.current_experiment_id = exp_id
                self.current_experiment_dir = self.get_experiment_path(exp_id)
                logger.info(f"恢复实验: {exp_id}")
                return exp_id
        
        # 创建新实验
        exp_id = self._generate_experiment_id(task, domain)
        exp_dir = os.path.join(self.experiments_root, exp_id)
        self.current_experiment_id = exp_id
        self.current_experiment_dir = exp_dir
        
        # 创建实验目录结构
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "evaluation"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        
        # 转换config为可序列化的字典
        config_dict = self._config_to_dict(config)
        
        # 保存实验配置
        config_path = os.path.join(exp_dir, "experiment_config.json")
        experiment_config = {
            "experiment_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "task": task,
            "domain": domain,
            "config": config_dict,
            "status": "created"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        # 保存训练参数到args.json方便恢复
        args_path = os.path.join(exp_dir, "training_args.json")
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"创建实验: {exp_id}")
        logger.info(f"实验目录: {exp_dir}")
        return exp_id
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """将配置对象转换为可JSON序列化的字典"""
        if config is None:
            return {}
        
        # 如果是argparse.Namespace，转换为字典
        if hasattr(config, '__dict__'):
            config_dict = vars(config).copy()
        elif isinstance(config, dict):
            config_dict = config.copy()
        else:
            config_dict = {"value": str(config)}
        
        # 递归处理嵌套对象
        result = {}
        for key, value in config_dict.items():
            # 跳过不可序列化的对象（如模型、tokenizer等）
            if key in ['model', 'tokenizer', 'dataset', 'trainer']:
                continue
            
            # 处理特殊类型
            if hasattr(value, '__dict__'):
                # 嵌套对象
                result[key] = self._config_to_dict(value)
            elif isinstance(value, (list, tuple)):
                # 列表或元组
                result[key] = [
                    self._config_to_dict(item) if hasattr(item, '__dict__') else 
                    str(item) if not self._is_json_serializable(item) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                # 嵌套字典
                result[key] = {
                    k: self._config_to_dict(v) if hasattr(v, '__dict__') else
                    str(v) if not self._is_json_serializable(v) else v
                    for k, v in value.items()
                }
            elif self._is_json_serializable(value):
                # 可直接序列化的值
                result[key] = value
            else:
                # 其他类型转为字符串
                result[key] = str(value)
        
        return result
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """检查对象是否可直接JSON序列化"""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    def _resolve_resume_experiment(self, task: str, domain: str, 
                                   resume: str) -> Optional[str]:
        """解析恢复参数，返回实验ID"""
        if resume == "auto":
            return self.find_latest_experiment(task, domain)
        else:
            # 恢复指定ID的实验
            exp_dir = self.get_experiment_path(resume)
            if not os.path.exists(exp_dir):
                logger.warning(f"指定实验 {resume} 不存在")
                return None
            
            # 验证是否有有效检查点
            latest_checkpoint = self.get_latest_checkpoint(resume)
            if latest_checkpoint:
                logger.info(f"恢复指定实验: {resume}")
                return resume
            else:
                logger.warning(f"指定实验 {resume} 没有有效检查点")
                return None
    
    def find_latest_experiment(self, task: str, domain: str) -> Optional[str]:
        """查找最新的实验（按创建时间）"""
        prefix = f"{task}_{domain}_"
        
        if not os.path.exists(self.experiments_root):
            return None
        
        experiments = []
        for d in os.listdir(self.experiments_root):
            if d.startswith(prefix) and os.path.isdir(os.path.join(self.experiments_root, d)):
                exp_path = os.path.join(self.experiments_root, d)
                # 按修改时间排序（更准确地反映最后活动）
                mtime = os.path.getmtime(exp_path)
                experiments.append((mtime, d))
        
        if not experiments:
            return None
        
        # 按时间降序排序，返回最新的
        experiments.sort(reverse=True)
        
        # 检查是否有有效检查点
        for mtime, exp_id in experiments:
            latest_checkpoint = self.get_latest_checkpoint(exp_id)
            if latest_checkpoint:
                logger.info(f"找到最新可恢复实验: {exp_id}")
                return exp_id
        
        logger.info("未找到有可恢复检查点的实验")
        return None
    
    def _generate_experiment_id(self, task: str, domain: str) -> str:
        """生成新的实验ID"""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        existing = self._get_existing_experiments(task, domain)
        exp_num = len(existing) + 1
        
        return f"{task}_{domain}_{exp_num:03d}_{timestamp}"
    
    def _get_existing_experiments(self, task: str, domain: str) -> List[str]:
        """获取现有实验列表"""
        prefix = f"{task}_{domain}_"
        
        if not os.path.exists(self.experiments_root):
            return []
        
        return [d for d in os.listdir(self.experiments_root) 
                if d.startswith(prefix) and os.path.isdir(os.path.join(self.experiments_root, d))]
    
    def get_experiment_path(self, exp_id: str) -> str:
        """获取实验路径"""
        return os.path.join(self.experiments_root, exp_id)
    
    def get_adapter_path(self, exp_id: Optional[str] = None) -> str:
        """获取适配器路径"""
        if exp_id is None:
            exp_id = self.current_experiment_id
        return os.path.join(self.experiments_root, exp_id, "final_adapter")
    
    def get_merged_model_path(self, exp_id: Optional[str] = None) -> str:
        """获取合并模型路径"""
        if exp_id is None:
            exp_id = self.current_experiment_id
        return os.path.join(self.experiments_root, exp_id, "merged_model")
    
    def get_latest_checkpoint(self, exp_id: Optional[str] = None) -> Optional[str]:
        """获取最新的检查点路径"""
        if exp_id is None:
            exp_id = self.current_experiment_id
        
        exp_dir = self.get_experiment_path(exp_id)
        
        if not os.path.exists(exp_dir):
            return None
        
        # 在实验目录下查找checkpoint-*目录
        checkpoints = []
        for d in os.listdir(exp_dir):
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(exp_dir, d)):
                checkpoints.append(d)
        
        if not checkpoints:
            return None
        
        # 按步数排序
        try:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        except (ValueError, IndexError) as e:
            logger.warning(f"解析checkpoint名称失败: {e}")
            return None
        
        latest_checkpoint = os.path.join(exp_dir, checkpoints[-1])
        
        # 验证检查点完整性
        required_files = ["trainer_state.json", "pytorch_model.bin", "adapter_config.json"]
        missing_files = [f for f in required_files 
                        if not os.path.exists(os.path.join(latest_checkpoint, f))]
        
        if missing_files:
            # 尝试safetensors格式
            if not os.path.exists(os.path.join(latest_checkpoint, "adapter_model.safetensors")):
                logger.warning(f"检查点不完整，缺少文件: {missing_files}")
                return None
        
        logger.info(f"找到有效检查点: {latest_checkpoint}")
        return latest_checkpoint
    
    def get_resume_checkpoint_path(self, resume: str, task: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
        """获取恢复训练的检查点路径和实验ID"""
        exp_id = self.create_experiment(task, domain, {}, resume=resume)
        if not exp_id:
            return None, None
        
        checkpoint_path = self.get_latest_checkpoint(exp_id)
        return checkpoint_path, exp_id
    
    def update_experiment_status(self, status: str, metadata: Dict[str, Any] = None):
        """更新实验状态"""
        if not self.current_experiment_id:
            return
        
        config_path = os.path.join(self.current_experiment_dir, "experiment_config.json")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config["status"] = status
            config["updated_at"] = datetime.now().isoformat()
            
            if metadata:
                if "metadata" not in config:
                    config["metadata"] = {}
                config["metadata"].update(metadata)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"更新实验状态失败: {e}")
    
    def save_checkpoint_info(self, checkpoint_path: str, step: int, loss: float):
        """保存检查点信息"""
        if not self.current_experiment_dir:
            return
        
        info_path = os.path.join(self.current_experiment_dir, "checkpoint_history.json")
        
        history = []
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass
        
        history.append({
            "step": step,
            "path": checkpoint_path,
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def list_experiments(self, task: Optional[str] = None, 
                        domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有实验"""
        if not os.path.exists(self.experiments_root):
            return []
        
        experiments = []
        for exp_id in os.listdir(self.experiments_root):
            exp_path = os.path.join(self.experiments_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            
            # 过滤
            if task and not exp_id.startswith(f"{task}_"):
                continue
            if domain and f"_{domain}_" not in exp_id:
                continue
            
            config_path = os.path.join(exp_path, "experiment_config.json")
            exp_info = {
                "experiment_id": exp_id,
                "path": exp_path,
                "created_at": None,
                "status": "unknown"
            }
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    exp_info["created_at"] = config.get("created_at")
                    exp_info["status"] = config.get("status", "unknown")
                    exp_info["task"] = config.get("task")
                    exp_info["domain"] = config.get("domain")
                except:
                    pass
            
            # 检查是否有最终适配器
            adapter_path = os.path.join(exp_path, "final_adapter")
            exp_info["has_adapter"] = os.path.exists(adapter_path)
            
            # 检查是否有合并模型
            merged_path = os.path.join(exp_path, "merged_model")
            exp_info["has_merged_model"] = os.path.exists(merged_path)
            
            experiments.append(exp_info)
        
        # 按创建时间排序
        experiments.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return experiments
    
    def cleanup_old_experiments(self, keep_count: int = 10):
        """清理旧实验，只保留最近的N个"""
        experiments = self.list_experiments()
        
        if len(experiments) <= keep_count:
            return
        
        to_remove = experiments[keep_count:]
        for exp in to_remove:
            exp_path = exp["path"]
            try:
                shutil.rmtree(exp_path)
                logger.info(f"清理旧实验: {exp['experiment_id']}")
            except Exception as e:
                logger.error(f"清理实验失败 {exp['experiment_id']}: {e}")


# 全局实验管理器实例
_experiment_manager = None

def get_experiment_manager(experiments_root: str = "output/experiments") -> ExperimentManager:
    """获取全局实验管理器实例"""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager(experiments_root)
    return _experiment_manager

def reset_experiment_manager():
    """重置实验管理器（用于测试）"""
    global _experiment_manager
    _experiment_manager = None