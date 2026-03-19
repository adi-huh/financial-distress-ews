"""
Day 28: Production Optimization & Scaling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class ScalingStrategy(Enum):
    """Scaling strategies"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AUTO = "auto"


@dataclass
class ResourceMetrics:
    """Resource metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    timestamp: str = field(default_factory=lambda: str(time.time()))


class LoadBalancer:
    """Load balancer"""
    
    def __init__(self):
        self.servers = []
        self.current_index = 0
    
    def add_server(self, server_id: str):
        """Add server"""
        self.servers.append({"id": server_id, "load": 0})
    
    def route_request(self) -> str:
        """Route request to server"""
        if not self.servers:
            return None
        
        # Round-robin load balancing
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        server["load"] += 1
        return server["id"]
    
    def get_least_loaded_server(self) -> Optional[str]:
        """Get least loaded server"""
        if not self.servers:
            return None
        
        return min(self.servers, key=lambda s: s["load"])["id"]


class CacheCluster:
    """Distributed cache cluster"""
    
    def __init__(self):
        self.nodes = {}
        self.replication_factor = 3
    
    def add_node(self, node_id: str):
        """Add cache node"""
        self.nodes[node_id] = {}
    
    def put(self, key: str, value: Any):
        """Store in cache"""
        if not self.nodes:
            return
        
        # Store in primary and replicas
        node_ids = list(self.nodes.keys())
        for i in range(min(self.replication_factor, len(node_ids))):
            self.nodes[node_ids[i]][key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache"""
        for node in self.nodes.values():
            if key in node:
                return node[key]
        return None
    
    def get_cluster_stats(self) -> Dict[str, int]:
        """Get cluster statistics"""
        return {
            "nodes": len(self.nodes),
            "total_keys": sum(len(node) for node in self.nodes.values()),
            "replication_factor": self.replication_factor
        }


class DatabaseSharding:
    """Database sharding"""
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards = [[] for _ in range(num_shards)]
    
    def get_shard(self, key: str) -> int:
        """Get shard for key"""
        return hash(key) % self.num_shards
    
    def put(self, key: str, value: Any):
        """Store in shard"""
        shard_id = self.get_shard(key)
        self.shards[shard_id].append({key: value})
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from shard"""
        shard_id = self.get_shard(key)
        for item in self.shards[shard_id]:
            if key in item:
                return item[key]
        return None


class AutoScaler:
    """Auto-scaling engine"""
    
    def __init__(self):
        self.scaling_policy = ScalingStrategy.AUTO
        self.min_instances = 1
        self.max_instances = 10
        self.cpu_threshold = 70
        self.memory_threshold = 80
    
    def set_scaling_policy(self, policy: ScalingStrategy):
        """Set scaling policy"""
        self.scaling_policy = policy
    
    def set_thresholds(self, cpu: float, memory: float):
        """Set scaling thresholds"""
        self.cpu_threshold = cpu
        self.memory_threshold = memory
    
    def decide_scaling(self, metrics: ResourceMetrics) -> Optional[str]:
        """Decide scaling action"""
        if metrics.cpu_usage > self.cpu_threshold or metrics.memory_usage > self.memory_threshold:
            return "scale_up"
        elif metrics.cpu_usage < 20 and metrics.memory_usage < 30:
            return "scale_down"
        return None
    
    def get_policy(self) -> Dict[str, Any]:
        """Get scaling policy"""
        return {
            "strategy": self.scaling_policy.value,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold
        }


class PerformanceTuner:
    """Performance tuning engine"""
    
    def __init__(self):
        self.optimizations = []
        self.metrics_baseline = {}
    
    def enable_query_optimization(self):
        """Enable query optimization"""
        self.optimizations.append("query_optimization")
    
    def enable_connection_pooling(self):
        """Enable connection pooling"""
        self.optimizations.append("connection_pooling")
    
    def enable_result_caching(self):
        """Enable result caching"""
        self.optimizations.append("result_caching")
    
    def set_baseline(self, metric: str, value: float):
        """Set baseline metric"""
        self.metrics_baseline[metric] = value
    
    def get_improvement(self, metric: str, current_value: float) -> float:
        """Calculate improvement"""
        if metric not in self.metrics_baseline:
            return 0
        baseline = self.metrics_baseline[metric]
        if baseline == 0:
            return 0
        return ((baseline - current_value) / baseline) * 100


class CDNConfiguration:
    """CDN configuration"""
    
    def __init__(self):
        self.edge_locations = []
        self.cache_policies = {}
    
    def add_edge_location(self, location: str):
        """Add CDN edge location"""
        self.edge_locations.append(location)
    
    def set_cache_policy(self, path: str, ttl: int):
        """Set cache policy"""
        self.cache_policies[path] = ttl
    
    def get_cdn_config(self) -> Dict[str, Any]:
        """Get CDN configuration"""
        return {
            "edge_locations": self.edge_locations,
            "cache_policies": self.cache_policies,
            "coverage": len(self.edge_locations)
        }


class DRPlan:
    """Disaster recovery plan"""
    
    def __init__(self):
        self.backup_strategies = []
        self.recovery_points = []
        self.failover_time_sla = 300  # 5 minutes
    
    def add_backup_strategy(self, name: str, frequency: str):
        """Add backup strategy"""
        self.backup_strategies.append({"name": name, "frequency": frequency})
    
    def add_recovery_point(self, timestamp: str, data_size: int):
        """Add recovery point"""
        self.recovery_points.append({"timestamp": timestamp, "size": data_size})
    
    def get_recovery_plan(self) -> Dict[str, Any]:
        """Get recovery plan"""
        return {
            "backup_strategies": self.backup_strategies,
            "recovery_points": len(self.recovery_points),
            "failover_sla_seconds": self.failover_time_sla,
            "rpo_hours": 1,
            "rto_minutes": 5
        }


class ProductionOptimizationEngine:
    """Production optimization engine"""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.cache_cluster = CacheCluster()
        self.sharding = DatabaseSharding()
        self.auto_scaler = AutoScaler()
        self.performance_tuner = PerformanceTuner()
        self.cdn = CDNConfiguration()
        self.dr_plan = DRPlan()
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization"""
        return {
            "load_balancer_ready": len(self.load_balancer.servers) > 0,
            "cache_nodes": self.cache_cluster.get_cluster_stats()["nodes"],
            "shards": self.sharding.num_shards,
            "auto_scaling_enabled": True,
            "optimizations_applied": len(self.performance_tuner.optimizations)
        }
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get production status"""
        return {
            "load_balancer": len(self.load_balancer.servers),
            "cache_cluster": self.cache_cluster.get_cluster_stats(),
            "scaling_policy": self.auto_scaler.get_policy(),
            "cdn_coverage": len(self.cdn.edge_locations),
            "dr_configured": len(self.dr_plan.backup_strategies) > 0
        }
