"""
MODULE: SCALING POLICIES & ANOMALY DETECTION
------------------------------------------------
Mô tả: Chứa các thuật toán ra quyết định mở rộng (Reactive/Predictive).
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 1. ANOMALY DETECTOR
class AnomalyDetector:
    def __init__(self, threshold: float = 3.0, window_size: int = 24):
        self.threshold = threshold
        self.window_size = window_size
        self.errors: List[float] = []

    def check(self, actual: float, predicted: float) -> bool:
        """Kiểm tra bất thường dựa trên Z-score của phần dư."""
        error = actual - predicted
        self.errors.append(error)
        
        if len(self.errors) < self.window_size:
            return False
            
        recent_errors = self.errors[-self.window_size:]
        std_dev = np.std(recent_errors)
        mean_error = np.mean(recent_errors)
        
        if std_dev == 0: return False
            
        z_score = abs(error - mean_error) / std_dev
        
        # Chỉ báo động khi lệch dương (Thực tế > Dự báo) quá ngưỡng
        return z_score > self.threshold and error > 0


# 2. SCALING STRATEGIES
class BaseScalingStrategy(ABC):
    def __init__(self, name: str, specs: Dict[str, Any]):
        self.name = name
        self.capacity = specs['server_capacity']
        self.min_rep = specs['min_replicas']
        self.max_rep = specs['max_replicas']

    @abstractmethod
    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        pass


class ReactiveStrategy(BaseScalingStrategy):
    """Chiến lược Phản ứng (Traditional)."""
    def __init__(self, specs: Dict[str, Any]):
        super().__init__("Reactive (Traditional)", specs)

    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        # Buffer 25% cố định
        target = np.ceil((current_demand / self.capacity) * 1.2)
        return int(np.clip(target, self.min_rep, self.max_rep))


class PredictiveStrategy(BaseScalingStrategy):
    """Chiến lược Dự báo."""
    def __init__(self, specs: Dict[str, Any]):
        super().__init__("Predictive", specs)
        self.buffer_ratio = specs['buffer_ratio']

    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        # Buffer động từ config
        target = np.ceil((predicted_demand / self.capacity) * (1 + self.buffer_ratio))
        return int(np.clip(target, self.min_rep, self.max_rep))