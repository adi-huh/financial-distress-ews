"""
Day 20: Performance Optimization Tests
Comprehensive tests for caching, query optimization, indexing, and bottleneck analysis
"""

import pytest
import tempfile
import time
import sqlite3
from pathlib import Path

from core.performance_optimization import (
    LRUCache, TTLCache, QueryOptimizer, DatabaseIndexManager,
    LazyProxy, BottleneckAnalyzer, ConnectionPool,
    PerformanceOptimizationEngine, CacheStats, CacheStrategy
)


class TestLRUCache:
    """Test LRU Cache implementation"""
    
    def test_lru_cache_creation(self):
        """Test creating LRU cache"""
        cache = LRUCache(max_size=10)
        assert cache.max_size == 10
        assert cache.size() == 0
    
    def test_lru_cache_set_get(self):
        """Test setting and getting values"""
        cache = LRUCache(max_size=5)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.size() == 2
    
    def test_lru_cache_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=3)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')
        
        # Access key1 to make it recently used
        cache.get('key1')
        
        # Add new item - key2 should be evicted
        cache.set('key4', 'value4')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') is None  # Evicted
        assert cache.get('key3') == 'value3'
    
    def test_lru_cache_delete(self):
        """Test deleting from cache"""
        cache = LRUCache(max_size=5)
        cache.set('key1', 'value1')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None
    
    def test_lru_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=5)
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 50.0
    
    def test_lru_cache_clear(self):
        """Test clearing cache"""
        cache = LRUCache(max_size=5)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get('key1') is None


class TestTTLCache:
    """Test TTL Cache implementation"""
    
    def test_ttl_cache_creation(self):
        """Test creating TTL cache"""
        cache = TTLCache(ttl_seconds=60)
        assert cache.ttl_seconds == 60
    
    def test_ttl_cache_set_get(self):
        """Test setting and getting values"""
        cache = TTLCache(ttl_seconds=3600)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
    
    def test_ttl_cache_expiration(self):
        """Test TTL expiration"""
        cache = TTLCache(ttl_seconds=1)
        cache.set('key1', 'value1')
        
        assert cache.get('key1') == 'value1'
        time.sleep(1.1)
        assert cache.get('key1') is None
    
    def test_ttl_cache_cleanup(self):
        """Test cleanup of expired items"""
        cache = TTLCache(ttl_seconds=1)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        time.sleep(1.1)
        removed = cache.cleanup_expired()
        
        assert removed == 2
    
    def test_ttl_cache_stats(self):
        """Test TTL cache statistics"""
        cache = TTLCache(ttl_seconds=3600)
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_ttl_cache_delete(self):
        """Test deleting from TTL cache"""
        cache = TTLCache(ttl_seconds=3600)
        cache.set('key1', 'value1')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None


class TestQueryOptimizer:
    """Test Query Optimizer"""
    
    def test_query_analyzer_creation(self):
        """Test creating query analyzer"""
        analyzer = QueryOptimizer()
        assert len(analyzer.analyzed_queries) == 0
    
    def test_query_complexity_calculation(self):
        """Test query complexity calculation"""
        analyzer = QueryOptimizer()
        
        simple_query = "SELECT * FROM users WHERE id = 1"
        analysis = analyzer.analyze_query(simple_query)
        
        assert analysis.complexity > 0
        assert analysis.complexity <= 10
    
    def test_query_with_joins(self):
        """Test query with JOINs"""
        analyzer = QueryOptimizer()
        
        join_query = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        analysis = analyzer.analyze_query(join_query)
        
        assert analysis.complexity > 1.0
        assert len(analysis.index_suggestions) > 0
    
    def test_query_optimization_hints(self):
        """Test query optimization hints"""
        analyzer = QueryOptimizer()
        
        query = "SELECT * FROM users"
        analysis = analyzer.analyze_query(query)
        
        assert len(analysis.optimization_hints) > 0
    
    def test_query_index_suggestions(self):
        """Test index suggestions"""
        analyzer = QueryOptimizer()
        
        query = "SELECT * FROM users WHERE status = 'active'"
        analysis = analyzer.analyze_query(query)
        
        assert len(analysis.index_suggestions) > 0
    
    def test_multiple_query_analysis(self):
        """Test analyzing multiple queries"""
        analyzer = QueryOptimizer()
        
        queries = [
            "SELECT * FROM users",
            "SELECT * FROM orders JOIN users",
            "SELECT * FROM products WHERE price > 100"
        ]
        
        for query in queries:
            analyzer.analyze_query(query)
        
        assert len(analyzer.analyzed_queries) == 3
    
    def test_query_execution_time_estimation(self):
        """Test query execution time estimation"""
        analyzer = QueryOptimizer()
        
        simple = "SELECT * FROM users WHERE id = 1"
        complex_query = "SELECT * FROM users JOIN orders JOIN products"
        
        simple_analysis = analyzer.analyze_query(simple)
        complex_analysis = analyzer.analyze_query(complex_query)
        
        assert complex_analysis.estimated_time_ms > simple_analysis.estimated_time_ms
    
    def test_n_plus_one_detection(self):
        """Test N+1 query detection"""
        analyzer = QueryOptimizer()
        query = "SELECT * FROM users"
        analysis = analyzer.analyze_query(query)
        
        assert isinstance(analysis.has_n_plus_one, bool)


class TestDatabaseIndexManager:
    """Test Database Index Manager"""
    
    def test_index_manager_creation(self):
        """Test creating index manager"""
        manager = DatabaseIndexManager()
        assert len(manager.indexes) == 0
    
    def test_create_index(self):
        """Test creating index"""
        manager = DatabaseIndexManager()
        index = manager.create_index('idx_user_id', 'users', ['id'])
        
        assert index.index_name == 'idx_user_id'
        assert index.table_name == 'users'
        assert 'id' in index.columns
    
    def test_delete_index(self):
        """Test deleting index"""
        manager = DatabaseIndexManager()
        manager.create_index('idx_user_id', 'users', ['id'])
        
        deleted = manager.delete_index('idx_user_id', 'users')
        assert deleted is True
    
    def test_list_indexes(self):
        """Test listing indexes"""
        manager = DatabaseIndexManager()
        manager.create_index('idx_user_id', 'users', ['id'])
        manager.create_index('idx_user_email', 'users', ['email'])
        
        indexes = manager.list_indexes()
        assert len(indexes) == 2
    
    def test_index_effectiveness_analysis(self):
        """Test index effectiveness analysis"""
        manager = DatabaseIndexManager()
        manager.create_index('idx_user_id', 'users', ['id'])
        
        effectiveness = manager.analyze_index_effectiveness('idx_user_id', 'users')
        assert 0 <= effectiveness <= 1.0
    
    def test_recommend_indexes(self):
        """Test index recommendations"""
        manager = DatabaseIndexManager()
        
        query = "SELECT * FROM users WHERE status = 'active'"
        recommendations = manager.recommend_indexes(query)
        
        assert len(recommendations) > 0
    
    def test_get_index_stats(self):
        """Test getting index statistics"""
        manager = DatabaseIndexManager()
        manager.create_index('idx_user_id', 'users', ['id'])
        
        stats = manager.get_index_stats('idx_user_id', 'users')
        assert stats is not None
        assert stats['index_name'] == 'idx_user_id'


class TestLazyProxy:
    """Test Lazy Loading Proxy"""
    
    def test_lazy_proxy_creation(self):
        """Test creating lazy proxy"""
        class ExpensiveObject:
            def __init__(self, value):
                self.value = value
                self.initialized = True
        
        proxy = LazyProxy(ExpensiveObject, 42)
        assert not proxy._initialized
    
    def test_lazy_proxy_initialization(self):
        """Test lazy initialization"""
        class ExpensiveObject:
            def __init__(self, value):
                self.value = value
        
        proxy = LazyProxy(ExpensiveObject, 42)
        assert not proxy._initialized
        
        # Access triggers initialization
        value = proxy.value
        assert proxy._initialized
        assert value == 42
    
    def test_lazy_proxy_method_call(self):
        """Test method calls through proxy"""
        class ExpensiveObject:
            def get_double(self, x):
                return x * 2
        
        proxy = LazyProxy(ExpensiveObject)
        result = proxy.get_double(21)
        
        assert result == 42


class TestBottleneckAnalyzer:
    """Test Bottleneck Analyzer"""
    
    def test_bottleneck_analyzer_creation(self):
        """Test creating bottleneck analyzer"""
        analyzer = BottleneckAnalyzer()
        assert len(analyzer.profiling_data) == 0
    
    def test_profile_function(self):
        """Test profiling a function"""
        analyzer = BottleneckAnalyzer()
        analyzer.profile_function('slow_func', 150.0, 100.0)
        
        assert 'slow_func' in analyzer.profiling_data
    
    def test_get_bottlenecks(self):
        """Test identifying bottlenecks"""
        analyzer = BottleneckAnalyzer()
        analyzer.profile_function('slow_func', 500.0, 200.0)
        analyzer.profile_function('fast_func', 10.0, 5.0)
        
        bottlenecks = analyzer.get_bottlenecks(threshold_ms=100.0)
        
        assert len(bottlenecks) >= 1
        assert bottlenecks[0].function_name == 'slow_func'
    
    def test_bottleneck_severity(self):
        """Test bottleneck severity classification"""
        analyzer = BottleneckAnalyzer()
        analyzer.profile_function('critical_func', 2000.0, 500.0)
        
        bottlenecks = analyzer.get_bottlenecks(threshold_ms=100.0)
        
        assert bottlenecks[0].severity == 'critical'
    
    def test_clear_profiling_data(self):
        """Test clearing profiling data"""
        analyzer = BottleneckAnalyzer()
        analyzer.profile_function('func1', 100.0, 50.0)
        analyzer.clear_profiling_data()
        
        assert len(analyzer.profiling_data) == 0


class TestConnectionPool:
    """Test Connection Pool"""
    
    def test_connection_pool_creation(self):
        """Test creating connection pool"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            pool = ConnectionPool(db_path, pool_size=3)
            
            assert pool.pool_size == 3
            pool.close_all()
    
    def test_get_connection(self):
        """Test getting connection from pool"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            pool = ConnectionPool(db_path, pool_size=2)
            
            conn = pool.get_connection()
            assert conn is not None
            
            pool.release_connection(conn)
            pool.close_all()
    
    def test_connection_pool_stats(self):
        """Test pool statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            pool = ConnectionPool(db_path, pool_size=3)
            
            conn1 = pool.get_connection()
            conn2 = pool.get_connection()
            
            stats = pool.get_stats()
            assert stats['active_connections'] == 2
            assert stats['idle_connections'] == 1
            
            if conn1:
                pool.release_connection(conn1)
            if conn2:
                pool.release_connection(conn2)
            pool.close_all()
    
    def test_pool_health_check(self):
        """Test pool health check"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            pool = ConnectionPool(db_path, pool_size=2)
            
            health = pool.check_health()
            assert health is True
            
            pool.close_all()


class TestPerformanceOptimizationEngine:
    """Test Performance Optimization Engine"""
    
    def test_engine_creation(self):
        """Test creating optimization engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            assert engine.lru_cache is not None
            assert engine.ttl_cache is not None
            assert engine.query_optimizer is not None
            
            engine.cleanup()
    
    def test_get_or_cache(self):
        """Test get or cache functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            def expensive_fetch():
                return 'expensive_result'
            
            result1 = engine.get_or_cache('key1', expensive_fetch)
            result2 = engine.get_or_cache('key1', expensive_fetch)
            
            assert result1 == result2
            
            engine.cleanup()
    
    def test_optimize_query(self):
        """Test query optimization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            analysis = engine.optimize_query("SELECT * FROM users WHERE id = 1")
            
            assert analysis.complexity > 0
            
            engine.cleanup()
    
    def test_create_index(self):
        """Test creating index through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            index = engine.create_index('idx_user_id', 'users', ['id'])
            
            assert index.index_name == 'idx_user_id'
            
            engine.cleanup()
    
    def test_analyze_bottlenecks(self):
        """Test bottleneck analysis"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            engine.bottleneck_analyzer.profile_function('slow_func', 500.0, 200.0)
            
            bottlenecks = engine.analyze_bottlenecks(threshold_ms=100.0)
            
            assert len(bottlenecks) >= 1
            
            engine.cleanup()
    
    def test_get_cache_stats(self):
        """Test getting cache statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            stats = engine.get_cache_stats()
            
            assert 'lru' in stats
            assert 'ttl' in stats
            
            engine.cleanup()
    
    def test_get_pool_stats(self):
        """Test getting pool statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = PerformanceOptimizationEngine(db_path)
            
            stats = engine.get_pool_stats()
            
            assert 'total_connections' in stats
            
            engine.cleanup()


class TestCacheStats:
    """Test Cache Statistics"""
    
    def test_cache_stats_creation(self):
        """Test creating cache stats"""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = CacheStats()
        stats.hits = 80
        stats.misses = 20
        stats.total_requests = 100
        
        assert stats.hit_rate() == 80.0
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.total_requests = 15
        
        stats_dict = stats.to_dict()
        
        assert stats_dict['hits'] == 10
        assert stats_dict['misses'] == 5


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_cache_get(self):
        """Test getting from empty cache"""
        cache = LRUCache(max_size=5)
        assert cache.get('nonexistent') is None
    
    def test_cache_with_none_values(self):
        """Test caching None values"""
        cache = LRUCache(max_size=5)
        cache.set('key_none', None)
        assert cache.get('key_none') is None
    
    def test_large_cache_size(self):
        """Test cache with large size"""
        cache = LRUCache(max_size=10000)
        for i in range(1000):
            cache.set(f'key{i}', f'value{i}')
        
        assert cache.size() == 1000
    
    def test_bottleneck_with_no_data(self):
        """Test bottleneck analysis with no profiling data"""
        analyzer = BottleneckAnalyzer()
        bottlenecks = analyzer.get_bottlenecks()
        
        assert len(bottlenecks) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
