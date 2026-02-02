"""
MODULE: FINANCIAL & COST MODELING
------------------------------------------------
Mô tả: Chịu trách nhiệm tính toán chi phí hạ tầng và phạt SLA.
"""
from typing import Tuple, Dict, Any

class CostModel:
    def __init__(self, specs: Dict[str, Any]):
        self.server_capacity = specs['server_capacity']
        self.server_cost = specs['server_cost']
        self.sla_penalty = specs['sla_penalty']

    def calculate_step_cost(self, replicas: int, actual_demand: float) -> Tuple[float, float, float, float]:
        """
        Tính toán tài chính cho một bước thời gian.
        Output: (Infra Cost, Penalty Cost, Total Cost, Dropped Requests)
        """
        # 1. Tính khả năng phục vụ
        total_capacity = replicas * self.server_capacity
        
        # 2. Tính request bị rớt (SLA Violation)
        dropped_requests = max(0, actual_demand - total_capacity)
        
        # 3. Tính tiền
        infra_cost = replicas * self.server_cost
        penalty_cost = dropped_requests * self.sla_penalty
        total_cost = infra_cost + penalty_cost
        
        return infra_cost, penalty_cost, total_cost, dropped_requests