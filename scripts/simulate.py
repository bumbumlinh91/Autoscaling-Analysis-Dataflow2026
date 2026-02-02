"""
SCRIPT: SIMULATION RUNNER
------------------------------------------------
Mô tả: Script thực thi mô phỏng, kết hợp Scaling Policy và Cost Model.
"""
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

# --- SETUP IMPORT TỪ SRC ---
# Thêm đường dẫn để import được package src
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.scaling_policy import AnomalyDetector, ReactiveStrategy, PredictiveStrategy
from src.costs import CostModel


# --- HELPER ---
def load_simulation_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        path = root_dir / config_path # Fallback tìm từ root
        
    with open(path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    return full_config['scaling_simulation']

# --- MAIN ENGINE ---
class SimulationEngine:
    def __init__(self, data_path: str, config_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
             self.data_path = root_dir / data_path

        print(f"⚙️ Đang tải cấu hình từ: {config_path}")
        self.specs = load_simulation_config(config_path)
        
        # 1. Khởi tạo Modules
        self.cost_model = CostModel(self.specs) # Chi phí
        self.anomaly_detector = AnomalyDetector(threshold=self.specs['anomaly_threshold']) # Kỹ thuật
        
        self.strategies = [
            ReactiveStrategy(self.specs),
            PredictiveStrategy(self.specs)
        ]
        self.results = pd.DataFrame()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df['ds'] = pd.to_datetime(df['ds'])
        return df.dropna().reset_index(drop=True)

    def run(self):
        print(f"\n{'='*60}\n🚀 ĐANG CHẠY MÔ PHỎNG \n{'='*60}")
        df = self._load_data()
        simulation_log = []
        
        cooldown_period = self.specs['cooldown_period']
        max_replicas = self.specs['max_replicas']
        min_replicas = self.specs['min_replicas']

        # Quản lý trạng thái (Cooldown)
        strategy_states = {
            s.name: {'current_replicas': min_replicas, 'cooldown_counter': 0} 
            for s in self.strategies
        }
        prev_actual_demand = df['Actual'].iloc[0]

        # --- LOOP MÔ PHỎNG ---
        for _, row in df.iterrows():
            actual = row['Actual']
            predicted = row['Prophet']
            timestamp = row['ds']
            
            # 1. Check Bất thường
            is_anomaly = self.anomaly_detector.check(actual, predicted)
            row_result = {'ds': timestamp, 'Actual': actual, 'Is_Anomaly': is_anomaly}

            for strategy in self.strategies:
                state = strategy_states[strategy.name]
                
                # 2. Tính Target Replicas 
                if is_anomaly and isinstance(strategy, PredictiveStrategy):
                    emergency_scale = int(state['current_replicas'] * 1.5) 
                    target = min(self.specs['max_replicas'], emergency_scale)
                elif isinstance(strategy, ReactiveStrategy):
                    target = strategy.calculate_target_replicas(prev_actual_demand, predicted)
                else:
                    target = strategy.calculate_target_replicas(actual, predicted)
                
                # 3. Áp dụng Cooldown 
                current = state['current_replicas']
                final_replicas = current

                if target > current:
                    final_replicas = target
                    state['cooldown_counter'] = cooldown_period
                elif target < current:
                    if state['cooldown_counter'] <= 0:
                        final_replicas = target
                    else:
                        state['cooldown_counter'] -= 1
                else:
                    state['cooldown_counter'] = max(0, state['cooldown_counter'] - 1)
                
                state['current_replicas'] = final_replicas
                
                # 4. Tính Tiền 
                infra, penalty, total, dropped = self.cost_model.calculate_step_cost(final_replicas, actual)
                
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
        """
        Tính toán ROI và xuất báo cáo tài chính chi tiết.
        """
        print(f"\n{'='*60}\n📊 BÁO CÁO HIỆU QUẢ TÀI CHÍNH & VẬN HÀNH\n{'='*60}")
        
        # 1. Tổng hợp số liệu
        summary_data = []
        financials = {}
        
        for strategy in self.strategies:
            name = strategy.name
            prefix = name.split()[0] # Reactive / Predictive
            
            # Tính tổng
            total_cost = self.results[f'{prefix}_Cost'].sum()
            total_dropped = self.results[f'{prefix}_Dropped'].sum()
            total_requests = self.results['Actual'].sum()
            
            # Tính % SLA Violation
            sla_fail_rate = (total_dropped / total_requests) * 100 if total_requests > 0 else 0
            
            # Lưu vào dict để so sánh sau
            financials[prefix] = total_cost
            
            summary_data.append({
                'Chiến lược': name, 
                'Tổng Chi Phí ($)': f"${total_cost:,.2f}", 
                'Request bị rớt': f"{int(total_dropped):,}",
                'Tỉ lệ lỗi SLA': f"{sla_fail_rate:.4f}%"
            })
            
        # 2. In bảng so sánh
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # 3. Tính ROI / Tiết kiệm (Phần mày đang thiếu)
        print(f"\n{'-'*60}")
        if 'Reactive' in financials and 'Predictive' in financials:
            baseline = financials['Reactive']
            optimized = financials['Predictive']
            
            savings = baseline - optimized
            savings_pct = (savings / baseline) * 100 if baseline > 0 else 0
            
            print(f"💰 PHÂN TÍCH ROI (SO VỚI TRUYỀN THỐNG):")
            print(f"   + Chi phí gốc (Reactive):   ${baseline:,.2f}")
            print(f"   + Chi phí mới (Predictive): ${optimized:,.2f}")
            print(f"   -----------------------------------------")
            print(f"   ✅ TIỀN TIẾT KIỆM ĐƯỢC:     ${savings:,.2f}")
            print(f"   🚀 HIỆU SUẤT TỐI ƯU (ROI):  {savings_pct:.2f}%")
            
            # Lưu kết quả ROI ra file text để làm bằng chứng
            with open("results/final_roi_report.txt", "w", encoding="utf-8") as f:
                f.write(f"ROI REPORT\n")
                f.write(f"Savings: ${savings:,.2f}\n")
                f.write(f"Percentage: {savings_pct:.2f}%\n")
        else:
            print("⚠️ Không đủ dữ liệu để so sánh ROI (Cần cả Reactive và Predictive).")
            
        print(f"{'='*60}\n")
        
        # Lưu file CSV chi tiết
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
    # Chạy trực tiếp
    sim = SimulationEngine("results/predictions_15min.csv", "config/config.yaml")
    sim.run()