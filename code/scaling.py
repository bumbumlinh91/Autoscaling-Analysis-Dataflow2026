"""
MODULE: AUTOSCALING SIMULATION ENGINE 
------------------------------------------------------------------------
Mô tả:
    Thực hiện mô phỏng các chiến lược tự động mở rộng (Reactive vs. Predictive)
    dựa trên tham số từ file cấu hình, đánh giá hiệu quả tài chính và độ ổn định.
    
    Các tính năng chính:
    1. Load tham số động từ config/config.yaml.
    2. Phát hiện bất thường.
    3. Cơ chế trễ & Mở rộng khẩn cấp.

"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from pathlib import Path

# CONFIG LOADER HELPER

def load_simulation_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Tải và kiểm tra cấu hình từ file YAML."""
    path = Path(config_path)
    if not path.exists():
        # Fallback đường dẫn nếu chạy từ thư mục khác
        path = Path("../config/config.yaml")
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    
    # Lấy section scaling, nếu không có thì báo lỗi
    if 'scaling_simulation' not in full_config:
        raise KeyError("Thiếu section 'scaling_simulation' trong file config.yaml")
        
    return full_config['scaling_simulation']

# CORE LOGIC CLASSES

class AnomalyDetector:
    """
    Lớp chịu trách nhiệm phát hiện các điểm dữ liệu bất thường.
    """
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


class BaseScalingStrategy(ABC):
    """
    Lớp trừu tượng cho chiến lược Scaling.
    Nhận tham số cấu hình (specs) từ bên ngoài vào.
    """
    def __init__(self, name: str, specs: Dict[str, Any]):
        self.name = name
        self.specs = specs
        
        # Thông số từ config
        self.capacity = specs['server_capacity']
        self.cost = specs['server_cost']
        self.sla_penalty = specs['sla_penalty']
        self.min_rep = specs['min_replicas']
        self.max_rep = specs['max_replicas']

    @abstractmethod
    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        pass

    def compute_financials(self, replicas: int, actual_demand: float) -> Tuple[float, float, float, float]:
        """Tính toán tài chính dựa trên tham số config."""
        capacity = replicas * self.capacity
        dropped_requests = max(0, actual_demand - capacity)
        
        infra_cost = replicas * self.cost
        sla_penalty = dropped_requests * self.sla_penalty
        total_cost = infra_cost + sla_penalty
        
        return infra_cost, sla_penalty, total_cost, dropped_requests


class ReactiveStrategy(BaseScalingStrategy):
    """Chiến lược Phản ứng (Reactive)."""
    def __init__(self, specs: Dict[str, Any]):
        super().__init__("Reactive (Traditional)", specs)

    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        # Buffer 20% cố định cho Reactive để bù trễ
        target = np.ceil((current_demand / self.capacity) * 1.2)
        return int(np.clip(target, self.min_rep, self.max_rep))


class PredictiveStrategy(BaseScalingStrategy):
    """Chiến lược Dự báo (AI Driven)."""
    def __init__(self, specs: Dict[str, Any]):
        super().__init__("Predictive (AI Driven)", specs)
        self.buffer_ratio = specs['buffer_ratio']

    def calculate_target_replicas(self, current_demand: float, predicted_demand: float) -> int:
        # Sử dụng Buffer từ Config
        target = np.ceil((predicted_demand / self.capacity) * (1 + self.buffer_ratio))
        return int(np.clip(target, self.min_rep, self.max_rep))


class SimulationEngine:
    """
    Động cơ mô phỏng chính.
    """
    def __init__(self, data_path: str, config_path: str):
        self.data_path = Path(data_path)
        
        # Tải cấu hình
        print(f"⚙️ Đang tải cấu hình từ: {config_path}")
        self.specs = load_simulation_config(config_path)
        print(f"   - Sức chịu tải: {self.specs['server_capacity']:,} reqs")
        print(f"   - Buffer: {self.specs['buffer_ratio']*100}%")
        print(f"   - Số lượng máy chủ tối đa: {self.specs['max_replicas']}")

        # Khởi tạo chiến lược và bộ phát hiện bất thường
        self.strategies: List[BaseScalingStrategy] = [
            ReactiveStrategy(self.specs),
            PredictiveStrategy(self.specs)
        ]
        self.anomaly_detector = AnomalyDetector(threshold=self.specs['anomaly_threshold'])
        self.results: pd.DataFrame = pd.DataFrame()

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['ds'] = pd.to_datetime(df['ds'])
        return df.dropna().reset_index(drop=True)

    def run(self):
        print(f"\n{'='*60}")
        print("ĐANG CHẠY MÔ PHỎNG TỰ ĐỘNG MỞ RỘNG ỨNG DỤNG")
        print(f"{'='*60}")
        
        df = self._load_data()
        simulation_log = []
        
        cooldown_period = self.specs['cooldown_period']
        max_replicas = self.specs['max_replicas']
        min_replicas = self.specs['min_replicas']

        # Quản lý trạng thái cho mỗi chiến lược
        strategy_states = {
            s.name: {'current_replicas': min_replicas, 'cooldown_counter': 0} 
            for s in self.strategies
        }
        
        prev_actual_demand = df['Actual'].iloc[0]

        for _, row in df.iterrows():
            actual = row['Actual']
            predicted = row['Prophet']
            timestamp = row['ds']
            
            # Check bất thường
            is_anomaly = self.anomaly_detector.check(actual, predicted)
            
            row_result = {'ds': timestamp, 'Actual': actual, 'Is_Anomaly': is_anomaly}

            for strategy in self.strategies:
                state = strategy_states[strategy.name]
                
                # Tính toán mục tiêu mở rộng
                if is_anomaly and isinstance(strategy, PredictiveStrategy):
                    target = max_replicas # Emergency Scale
                elif isinstance(strategy, ReactiveStrategy):
                    target = strategy.calculate_target_replicas(prev_actual_demand, predicted)
                else:
                    target = strategy.calculate_target_replicas(actual, predicted)
                
                # 3. Logic trễ & cooldown
                current = state['current_replicas']
                final_replicas = current

                if target > current:
                    final_replicas = target
                    state['cooldown_counter'] = cooldown_period
                elif target < current:
                    if state['cooldown_counter'] <= 0:
                        final_replicas = target
                    else:
                        final_replicas = current
                        state['cooldown_counter'] -= 1
                else:
                    state['cooldown_counter'] = max(0, state['cooldown_counter'] - 1)
                
                state['current_replicas'] = final_replicas
                
                # 4. Tính toán tài chính
                infra, penalty, total, dropped = strategy.compute_financials(final_replicas, actual)
                
                # 5. Log
                prefix = strategy.name.split()[0]
                row_result[f'{prefix}_Replicas'] = final_replicas
                row_result[f'{prefix}_Cost'] = total
                row_result[f'{prefix}_Dropped'] = dropped
            
            simulation_log.append(row_result)
            prev_actual_demand = actual

        self.results = pd.DataFrame(simulation_log)
        self._generate_report()
        self._visualize_results()

    def _generate_report(self):
        print(f"\n{'='*60}")
        print("BÁO CÁO CHI PHÍ VÀ HIỆU QUẢ MÔ PHỎNG")
        print(f"{'='*60}")
        
        summary_data = []
        for strategy in self.strategies:
            prefix = strategy.name.split()[0]
            total_cost = self.results[f'{prefix}_Cost'].sum()
            total_dropped = self.results[f'{prefix}_Dropped'].sum()
            
            summary_data.append({
                'Chiến lược': strategy.name,
                'Tổng chi phí ($)': total_cost,
                'Requests bị bỏ qua': int(total_dropped)
            })
            
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Tính ROI
        base_cost = summary_df.loc[0, 'Tổng chi phí ($)']
        ai_cost = summary_df.loc[1, 'Tổng chi phí ($)']
        savings = base_cost - ai_cost
        savings_pct = (savings / base_cost) * 100
        
        print(f"\n>>> Phân tích ROI: Chính sách scaling dự báo tiết kiệm ${savings:,.2f} ({savings_pct:.2f}%)")
        
        Path("results").mkdir(exist_ok=True)
        summary_df.to_csv("results/scaling_financial_report.csv", index=False)

    def _visualize_results(self):
        # Biểu đồ kết quả mô phỏng
        plt.style.use('seaborn-v0_8-whitegrid')
        chart_dir = Path("results/charts")
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        zoom_df = self.results.head(288) # Zoom 3 ngày
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        # Chart 1
        capacity = self.specs['server_capacity'] 
        ax1.plot(zoom_df['ds'], zoom_df['Actual'] / capacity, 
                 color='gray', alpha=0.3, label='Normalized Demand')
        
        anomalies = zoom_df[zoom_df['Is_Anomaly']]
        ax1.scatter(anomalies['ds'], anomalies['Actual'] / capacity, 
                    color='red', s=40, label='Anomaly Detected', zorder=5)
        
        ax1.step(zoom_df['ds'], zoom_df['Reactive_Replicas'], 
                 color='#d62728', label='Reactive', where='post', linestyle='--')
        ax1.step(zoom_df['ds'], zoom_df['Predictive_Replicas'], 
                 color='#2ca02c', label='Predictive (AI)', where='post', linewidth=2)
        
        ax1.set_ylabel("Replicas")
        ax1.set_title("Scaling Behavior (3-Day Zoom)", fontweight='bold')
        ax1.legend()
        
        # Chart 2
        self.results['Reactive_CumCost'] = self.results['Reactive_Cost'].cumsum()
        self.results['Predictive_CumCost'] = self.results['Predictive_Cost'].cumsum()
        
        ax2.plot(self.results['ds'], self.results['Reactive_CumCost'], color='#d62728', label='Reactive Cost')
        ax2.plot(self.results['ds'], self.results['Predictive_CumCost'], color='#2ca02c', label='Predictive Cost')
        ax2.fill_between(self.results['ds'], self.results['Reactive_CumCost'], self.results['Predictive_CumCost'], 
                         color='green', alpha=0.1, label='Cost Savings')
        
        ax2.set_ylabel("Cumulative Cost ($)")
        ax2.set_title("Financial Trajectory", fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(chart_dir / "scaling_simulation_final.png", dpi=150)
        print(f"\n[+] Visualization saved: {chart_dir / 'scaling_simulation_final.png'}")

if __name__ == "__main__":
    DATA_FILE = "results/predictions_15min.csv"
    CONFIG_FILE = "config/config.yaml"
    
    try:
        sim = SimulationEngine(DATA_FILE, CONFIG_FILE)
        sim.run()
    except Exception as e:
        print(f"❌ Error: {e}")