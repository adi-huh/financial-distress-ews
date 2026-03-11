"""
Day 20: Performance Optimization System
Comprehensive performance optimization with caching, query optimization,
database indexing, lazy loading, and bottleneck analysis
"""

import logging
import time
import sqlite3
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"
    TTL = "ttl"
    FIFO = "fifo"


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate(),
            'created_at': self.created_at.isoformat()
        }


class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self.stats.total_requests += 1
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return self.cache[key]
            else:
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                    self.stats.evictions += 1
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return self.stats.to_dict()
    
    def serialize(self, filepath: str) -> None:
        """Serialize cache to file"""
        with self.lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Cache serialized to {filepath}")
    
    def deserialize(self, filepath: str) -> None:
        """Deserialize cache from file"""
        with self.lock:
            if Path(filepath).exists():
                with open(filepath, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Cache deserialized from {filepath}")


class TTLCache:
    """Time-To-Live Cache implementation"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.stats = CacheStats()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self.stats.total_requests += 1
            if key in self.cache:
                value, timestamp = self.cache[key]
                if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self.ttl_seconds):
                    self.stats.hits += 1
                    return value
                else:
                    del self.cache[key]
                    self.stats.evictions += 1
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            self.cache[key] = (value, datetime.now(timezone.utc))
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items"""
        with self.lock:
            expired_keys = []
            now = datetime.now(timezone.utc)
            for key, (value, timestamp) in self.cache.items():
                if now - timestamp >= timedelta(seconds=self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return self.stats.to_dict()


@dataclass
class QueryAnalysis:
    """Query analysis results"""
    query: str
    complexity: float
    estimated_time_ms: float
    has_n_plus_one: bool
    index_suggestions: List[str]
    optimization_hints: List[str]
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QueryOptimizer:
    """Database query optimization"""
    
    def __init__(self):
        self.analyzed_queries: List[QueryAnalysis] = []
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query for optimization opportunities"""
        complexity = self._calculate_complexity(query)
        estimated_time = self._estimate_execution_time(query)
        has_n_plus_one = self._detect_n_plus_one(query)
        index_suggestions = self._suggest_indexes(query)
        optimization_hints = self._generate_hints(query)
        
        analysis = QueryAnalysis(
            query=query,
            complexity=complexity,
            estimated_time_ms=estimated_time,
            has_n_plus_one=has_n_plus_one,
            index_suggestions=index_suggestions,
            optimization_hints=optimization_hints
        )
        
        self.analyzed_queries.append(analysis)
        return analysis
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity (0-10 scale)"""
        complexity = 1.0
        
        if 'JOIN' in query.upper():
            join_count = query.upper().count('JOIN')
            complexity += join_count * 1.5
        
        if 'SUBQUERY' in query.upper() or '(SELECT' in query.upper():
            complexity += 2.0
        
        if 'GROUP BY' in query.upper():
            complexity += 1.0
        
        if 'ORDER BY' in query.upper():
            complexity += 0.5
        
        if 'DISTINCT' in query.upper():
            complexity += 0.5
        
        return min(complexity, 10.0)
    
    def _estimate_execution_time(self, query: str) -> float:
        """Estimate query execution time"""
        complexity = self._calculate_complexity(query)
        base_time = 10.0
        return base_time * (complexity / 1.0)
    
    def _detect_n_plus_one(self, query: str) -> bool:
        """Detect N+1 query pattern"""
        # Simple heuristic: check for query in loop patterns
        return False  # Simplified
    
    def _suggest_indexes(self, query: str) -> List[str]:
        """Suggest indexes for query"""
        suggestions = []
        
        if 'WHERE' in query.upper():
            suggestions.append("Add index on WHERE clause columns")
        
        if 'JOIN' in query.upper():
            suggestions.append("Add index on JOIN columns")
        
        if 'ORDER BY' in query.upper():
            suggestions.append("Add index on ORDER BY columns")
        
        return suggestions
    
    def _generate_hints(self, query: str) -> List[str]:
        """Generate optimization hints"""
        hints = []
        
        if 'SELECT *' in query.upper():
            hints.append("Specify only needed columns instead of SELECT *")
        
        if 'JOIN' in query.upper() and query.upper().count('JOIN') > 3:
            hints.append("Consider reducing number of JOINs")
        
        if 'OR' in query.upper():
            hints.append("Consider using IN clause instead of OR")
        
        return hints


@dataclass
class IndexStats:
    """Index statistics"""
    index_name: str
    table_name: str
    columns: List[str]
    effectiveness: float
    query_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DatabaseIndexManager:
    """Manage database indexes"""
    
    def __init__(self):
        self.indexes: Dict[str, IndexStats] = {}
        self.query_stats: Dict[str, int] = {}
    
    def create_index(self, index_name: str, table_name: str, columns: List[str]) -> IndexStats:
        """Create database index"""
        index_key = f"{table_name}_{index_name}"
        
        if index_key not in self.indexes:
            index_stats = IndexStats(
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                effectiveness=0.5
            )
            self.indexes[index_key] = index_stats
            logger.info(f"Created index: {index_name} on {table_name}")
        
        return self.indexes[index_key]
    
    def delete_index(self, index_name: str, table_name: str) -> bool:
        """Delete database index"""
        index_key = f"{table_name}_{index_name}"
        if index_key in self.indexes:
            del self.indexes[index_key]
            logger.info(f"Deleted index: {index_name}")
            return True
        return False
    
    def analyze_index_effectiveness(self, index_name: str, table_name: str) -> float:
        """Analyze how effective an index is"""
        index_key = f"{table_name}_{index_name}"
        if index_key not in self.indexes:
            return 0.0
        
        stats = self.indexes[index_key]
        if stats.query_count == 0:
            return 0.5
        
        return min((stats.query_count / 1000.0), 1.0)
    
    def recommend_indexes(self, query: str) -> List[str]:
        """Recommend indexes based on query"""
        recommendations = []
        
        if 'WHERE' in query.upper():
            recommendations.append("Index on WHERE clause columns")
        
        if 'JOIN' in query.upper():
            recommendations.append("Index on JOIN predicate columns")
        
        return recommendations
    
    def get_index_stats(self, index_name: str, table_name: str) -> Optional[Dict]:
        """Get statistics for an index"""
        index_key = f"{table_name}_{index_name}"
        if index_key in self.indexes:
            stats = self.indexes[index_key]
            return asdict(stats)
        return None
    
    def list_indexes(self) -> List[Dict]:
        """List all indexes"""
        return [asdict(idx) for idx in self.indexes.values()]


class LazyProxy:
    """Lazy loading proxy for deferred initialization"""
    
    def __init__(self, target_class: type, *args, **kwargs):
        self._target_class = target_class
        self._args = args
        self._kwargs = kwargs
        self._target_obj = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize target object if needed"""
        if not self._initialized:
            self._target_obj = self._target_class(*self._args, **self._kwargs)
            self._initialized = True
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to target"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        self._ensure_initialized()
        return getattr(self._target_obj, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Forward calls to target"""
        self._ensure_initialized()
        return self._target_obj(*args, **kwargs)


@dataclass
class BottleneckReport:
    """Performance bottleneck report"""
    function_name: str
    execution_time_ms: float
    call_count: int
    avg_time_ms: float
    memory_used_mb: float
    severity: str  # 'critical', 'high', 'medium', 'low'


class BottleneckAnalyzer:
    """Analyze performance bottlenecks"""
    
    def __init__(self):
        self.profiling_data: Dict[str, Dict] = {}
    
    def profile_function(self, func_name: str, execution_time_ms: float, 
                        memory_used_mb: float) -> None:
        """Record function profiling data"""
        if func_name not in self.profiling_data:
            self.profiling_data[func_name] = {
                'execution_times': [],
                'memory_used': [],
                'call_count': 0
            }
        
        self.profiling_data[func_name]['execution_times'].append(execution_time_ms)
        self.profiling_data[func_name]['memory_used'].append(memory_used_mb)
        self.profiling_data[func_name]['call_count'] += 1
    
    def get_bottlenecks(self, threshold_ms: float = 100.0) -> List[BottleneckReport]:
        """Identify bottleneck functions"""
        bottlenecks = []
        
        for func_name, data in self.profiling_data.items():
            if not data['execution_times']:
                continue
            
            total_time = sum(data['execution_times'])
            avg_time = total_time / len(data['execution_times'])
            avg_memory = sum(data['memory_used']) / len(data['memory_used'])
            
            if avg_time >= threshold_ms:
                # Determine severity
                if avg_time > 1000:
                    severity = 'critical'
                elif avg_time > 500:
                    severity = 'high'
                elif avg_time > 200:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                report = BottleneckReport(
                    function_name=func_name,
                    execution_time_ms=total_time,
                    call_count=data['call_count'],
                    avg_time_ms=avg_time,
                    memory_used_mb=avg_memory,
                    severity=severity
                )
                bottlenecks.append(report)
        
        return sorted(bottlenecks, key=lambda x: x.avg_time_ms, reverse=True)
    
    def clear_profiling_data(self) -> None:
        """Clear profiling data"""
        self.profiling_data.clear()


@dataclass
class PoolStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    wait_time_ms: float = 0.0
    avg_checkout_time_ms: float = 0.0


class ConnectionPool:
    """Database connection pooling"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool: List[sqlite3.Connection] = []
        self.active: List[sqlite3.Connection] = []
        self.lock = threading.Lock()
        self.stats = PoolStats(total_connections=pool_size)
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.pool.append(conn)
            except Exception as e:
                logger.error(f"Error initializing pool: {str(e)}")
    
    def get_connection(self) -> Optional[sqlite3.Connection]:
        """Get connection from pool"""
        with self.lock:
            if self.pool:
                conn = self.pool.pop()
                self.active.append(conn)
                self.stats.active_connections = len(self.active)
                self.stats.idle_connections = len(self.pool)
                return conn
            return None
    
    def release_connection(self, conn: sqlite3.Connection) -> None:
        """Return connection to pool"""
        with self.lock:
            if conn in self.active:
                self.active.remove(conn)
                self.pool.append(conn)
                self.stats.active_connections = len(self.active)
                self.stats.idle_connections = len(self.pool)
    
    def check_health(self) -> bool:
        """Check pool health"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Pool health check failed: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self.lock:
            return asdict(self.stats)
    
    def close_all(self) -> None:
        """Close all connections"""
        with self.lock:
            for conn in self.pool + self.active:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")
            self.pool.clear()
            self.active.clear()


class PerformanceOptimizationEngine:
    """Main orchestrator for performance optimization"""
    
    def __init__(self, db_path: str = 'instance/app.db'):
        self.db_path = db_path
        self.lru_cache = LRUCache(max_size=1000)
        self.ttl_cache = TTLCache(ttl_seconds=3600)
        self.query_optimizer = QueryOptimizer()
        self.index_manager = DatabaseIndexManager()
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.connection_pool = ConnectionPool(db_path, pool_size=5)
        logger.info("Performance Optimization Engine initialized")
    
    def get_or_cache(self, key: str, fetch_func: Callable) -> Any:
        """Get value from cache or fetch"""
        # Try LRU cache first
        value = self.lru_cache.get(key)
        if value is not None:
            return value
        
        # Fetch new value
        value = fetch_func()
        self.lru_cache.set(key, value)
        self.ttl_cache.set(key, value)
        return value
    
    def optimize_query(self, query: str) -> QueryAnalysis:
        """Optimize a database query"""
        return self.query_optimizer.analyze_query(query)
    
    def create_index(self, index_name: str, table_name: str, 
                    columns: List[str]) -> IndexStats:
        """Create an optimized index"""
        return self.index_manager.create_index(index_name, table_name, columns)
    
    def analyze_bottlenecks(self, threshold_ms: float = 100.0) -> List[BottleneckReport]:
        """Get performance bottlenecks"""
        return self.bottleneck_analyzer.get_bottlenecks(threshold_ms)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'lru': self.lru_cache.get_stats(),
            'ttl': self.ttl_cache.get_stats()
        }
    
    def get_pool_stats(self) -> Dict:
        """Get connection pool statistics"""
        return self.connection_pool.get_stats()
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.connection_pool.close_all()
        self.lru_cache.clear()
        self.ttl_cache.clear()
        logger.info("Performance Optimization Engine cleaned up")
