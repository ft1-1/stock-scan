"""Memory performance and leak detection tests."""

import pytest
import gc
import psutil
import os
import time
import threading
from typing import List, Dict, Tuple
import weakref
from dataclasses import dataclass

# Import modules for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria
from providers.cache import CacheManager, TTLCache
from ai_analysis.data_packager import DataPackager, PackagingConfig
from screener.workflow_engine import WorkflowEngine, StepResult
from models.workflow_models import WorkflowConfig


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    num_objects: int


class MemoryTestSuite:
    """Memory performance and leak detection test suite."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[MemorySnapshot] = []
    
    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory usage snapshot."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Count Python objects
        gc.collect()  # Force garbage collection
        num_objects = len(gc.get_objects())
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            num_objects=num_objects
        )
        
        self.snapshots.append(snapshot)
        
        if label:
            print(f"Memory snapshot ({label}): {snapshot.rss_mb:.1f}MB RSS, {num_objects:,} objects")
        
        return snapshot
    
    @pytest.mark.performance
    def test_technical_indicators_memory_usage(self, test_data_generator):
        """Test memory usage of technical indicators with large datasets."""
        print("\n=== Technical Indicators Memory Test ===")
        
        self.take_memory_snapshot("initial")
        
        # Test with increasingly large datasets
        data_sizes = [252, 500, 1000, 2000]  # Days of data
        
        for days in data_sizes:
            print(f"Testing with {days} days of data...")
            
            # Generate data
            before_data = self.take_memory_snapshot(f"before_data_{days}")
            large_dataset = test_data_generator.generate_historical_data("MEMORY_TEST", days=days)
            after_data = self.take_memory_snapshot(f"after_data_{days}")
            
            # Calculate indicators
            before_calc = self.take_memory_snapshot(f"before_calc_{days}")
            indicators = TechnicalIndicators.calculate_all_indicators(large_dataset)
            after_calc = self.take_memory_snapshot(f"after_calc_{days}")
            
            # Calculate memory usage
            data_memory = after_data.rss_mb - before_data.rss_mb
            calc_memory = after_calc.rss_mb - before_calc.rss_mb
            
            print(f"  Data: {data_memory:.1f}MB, Calculation: {calc_memory:.1f}MB")
            
            # Memory usage should be reasonable
            assert data_memory < days * 0.01, f"Data memory too high: {data_memory:.1f}MB for {days} days"
            assert calc_memory < 50, f"Calculation memory too high: {calc_memory:.1f}MB"
            
            # Clean up
            del large_dataset, indicators
            gc.collect()
            
            after_cleanup = self.take_memory_snapshot(f"after_cleanup_{days}")
            
            # Memory should be mostly recovered
            memory_leak = after_cleanup.rss_mb - before_data.rss_mb
            assert memory_leak < 10, f"Possible memory leak: {memory_leak:.1f}MB not recovered"
    
    @pytest.mark.performance
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency and limits."""
        print("\n=== Cache Memory Efficiency Test ===")
        
        self.take_memory_snapshot("cache_initial")
        
        # Create cache with size limit
        cache = TTLCache(max_size=1000, default_ttl=300)
        
        after_cache_creation = self.take_memory_snapshot("cache_created")
        
        # Fill cache with data
        large_data = "X" * 1024  # 1KB strings
        
        for i in range(2000):  # Try to exceed cache limit
            cache.set(f"key_{i}", large_data)
            
            if i % 500 == 0:
                self.take_memory_snapshot(f"cache_fill_{i}")
        
        after_fill = self.take_memory_snapshot("cache_filled")
        
        # Check cache size enforcement
        assert len(cache._data) <= 1000, f"Cache exceeded size limit: {len(cache._data)} items"
        
        # Memory usage should be bounded
        cache_memory = after_fill.rss_mb - after_cache_creation.rss_mb
        
        # Should not use more than ~2MB for 1000 x 1KB items (with overhead)
        assert cache_memory < 5, f"Cache memory usage too high: {cache_memory:.1f}MB"
        
        # Clear cache
        cache.clear()
        gc.collect()
        
        after_clear = self.take_memory_snapshot("cache_cleared")
        
        # Most memory should be recovered
        memory_leak = after_clear.rss_mb - after_cache_creation.rss_mb
        assert memory_leak < 1, f"Cache memory leak: {memory_leak:.1f}MB not recovered"
    
    @pytest.mark.performance
    def test_options_selector_memory_scaling(self, test_data_generator):
        """Test options selector memory usage with large option chains."""
        print("\n=== Options Selector Memory Scaling Test ===")
        
        criteria = CallOptionCriteria(
            min_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45
        )
        selector = OptionsSelector(criteria)
        
        self.take_memory_snapshot("selector_initial")
        
        # Test with increasing chain sizes
        chain_sizes = [50, 100, 250, 500, 1000]
        
        for size in chain_sizes:
            print(f"Testing with {size} options...")
            
            before_generation = self.take_memory_snapshot(f"before_gen_{size}")
            
            # Generate large options chain
            options_chain = []
            for i in range(size):
                option = test_data_generator.generate_options_chain("MEMORY", 100 + i*0.1, 30)[0]
                options_chain.append(option)
            
            after_generation = self.take_memory_snapshot(f"after_gen_{size}")
            
            # Process options
            before_processing = self.take_memory_snapshot(f"before_proc_{size}")
            best_calls = selector.select_best_calls(options_chain, top_n=min(10, size//10))
            after_processing = self.take_memory_snapshot(f"after_proc_{size}")
            
            # Calculate memory usage
            generation_memory = after_generation.rss_mb - before_generation.rss_mb
            processing_memory = after_processing.rss_mb - before_processing.rss_mb
            
            print(f"  Generation: {generation_memory:.1f}MB, Processing: {processing_memory:.1f}MB")
            
            # Memory should scale reasonably
            expected_gen_memory = size * 0.001  # ~1KB per option
            expected_proc_memory = size * 0.0005  # ~0.5KB per option processing
            
            assert generation_memory < max(5, expected_gen_memory * 2), \
                f"Options generation memory too high: {generation_memory:.1f}MB"
            assert processing_memory < max(2, expected_proc_memory * 2), \
                f"Options processing memory too high: {processing_memory:.1f}MB"
            
            # Clean up
            del options_chain, best_calls
            gc.collect()
    
    @pytest.mark.performance
    def test_workflow_engine_memory_usage(self, test_data_generator):
        """Test workflow engine memory usage during execution."""
        print("\n=== Workflow Engine Memory Test ===")
        
        config = WorkflowConfig(
            max_concurrent_stocks=5,
            max_retry_attempts=1,
            step_timeout_seconds=30,
            save_intermediate_results=True
        )
        
        engine = WorkflowEngine(config)
        
        # Register memory-intensive step
        async def memory_intensive_step(context, symbols):
            # Allocate some memory for each symbol
            results = {}
            for symbol in symbols:
                # Generate data that uses memory
                data = test_data_generator.generate_historical_data(symbol, days=100)
                indicators = TechnicalIndicators.calculate_all_indicators(data)
                results[symbol] = {
                    "data": data,
                    "indicators": indicators
                }
            
            return StepResult(
                step_id="memory_step",
                success=True,
                results=results,
                execution_time_seconds=0.1
            )
        
        engine.register_step("memory_step", memory_intensive_step, order=1)
        
        self.take_memory_snapshot("workflow_initial")
        
        # Test with different numbers of symbols
        symbol_counts = [5, 10, 20, 30]
        
        for count in symbol_counts:
            print(f"Testing workflow with {count} symbols...")
            
            symbols = test_data_generator.generate_stock_symbols(count)
            
            before_workflow = self.take_memory_snapshot(f"before_workflow_{count}")
            
            # Execute workflow
            import asyncio
            context = asyncio.run(engine.execute_workflow(symbols))
            
            after_workflow = self.take_memory_snapshot(f"after_workflow_{count}")
            
            # Memory usage
            workflow_memory = after_workflow.rss_mb - before_workflow.rss_mb
            memory_per_symbol = workflow_memory / count if count > 0 else 0
            
            print(f"  Total: {workflow_memory:.1f}MB, Per symbol: {memory_per_symbol:.1f}MB")
            
            # Memory per symbol should be reasonable
            assert memory_per_symbol < 5, f"Memory per symbol too high: {memory_per_symbol:.1f}MB"
            
            # Clean up workflow context
            del context
            gc.collect()
            
            after_cleanup = self.take_memory_snapshot(f"after_cleanup_{count}")
            
            # Check for memory leaks
            memory_leak = after_cleanup.rss_mb - before_workflow.rss_mb
            assert memory_leak < count * 0.5, f"Workflow memory leak: {memory_leak:.1f}MB"
    
    @pytest.mark.performance
    def test_data_packager_memory_efficiency(self, test_data_generator):
        """Test data packager memory efficiency with large datasets."""
        print("\n=== Data Packager Memory Test ===")
        
        config = PackagingConfig(
            max_payload_size_mb=10,
            compression_enabled=True
        )
        packager = DataPackager(config)
        
        self.take_memory_snapshot("packager_initial")
        
        # Generate large dataset
        large_stock_data = {}
        for i in range(50):
            symbol = f"LARGE_{i:03d}"
            large_stock_data[symbol] = {
                "quote": test_data_generator.generate_stock_quote(symbol),
                "historical": test_data_generator.generate_historical_data(symbol, days=252),
                "options": test_data_generator.generate_options_chain(symbol, 100 + i)
            }
        
        after_data_gen = self.take_memory_snapshot("after_data_generation")
        
        # Package data
        before_packaging = self.take_memory_snapshot("before_packaging")
        
        packaged_data = packager.package_for_analysis(large_stock_data)
        
        after_packaging = self.take_memory_snapshot("after_packaging")
        
        # Compress data
        before_compression = self.take_memory_snapshot("before_compression")
        
        compressed_data = packager.compress_data(packaged_data)
        
        after_compression = self.take_memory_snapshot("after_compression")
        
        # Calculate memory usage
        packaging_memory = after_packaging.rss_mb - before_packaging.rss_mb
        compression_memory = after_compression.rss_mb - before_compression.rss_mb
        
        print(f"Packaging memory: {packaging_memory:.1f}MB")
        print(f"Compression memory: {compression_memory:.1f}MB")
        
        # Memory usage should be reasonable
        assert packaging_memory < 50, f"Packaging memory too high: {packaging_memory:.1f}MB"
        assert compression_memory < 20, f"Compression memory too high: {compression_memory:.1f}MB"
        
        # Verify compression effectiveness
        import json
        original_size = len(json.dumps(packaged_data, default=str))
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        
        print(f"Compression ratio: {compression_ratio:.2f}x")
        assert compression_ratio > 1.1, f"Poor compression ratio: {compression_ratio:.2f}x"
        
        # Clean up
        del large_stock_data, packaged_data, compressed_data
        gc.collect()
        
        after_cleanup = self.take_memory_snapshot("after_cleanup")
        
        # Check for memory leaks
        total_leak = after_cleanup.rss_mb - self.snapshots[0].rss_mb
        assert total_leak < 20, f"Total memory leak: {total_leak:.1f}MB"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_memory_stability(self, test_data_generator):
        """Test memory stability over long-running operations."""
        print("\n=== Long Running Memory Stability Test ===")
        
        self.take_memory_snapshot("stability_initial")
        
        # Simulate long-running operation
        iterations = 100
        memory_samples = []
        
        for i in range(iterations):
            # Perform various operations
            symbol = f"STABLE_{i % 10:03d}"
            
            # Generate and process data
            data = test_data_generator.generate_historical_data(symbol, days=50)
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            # Momentum analysis
            analyzer = MomentumAnalyzer()
            momentum = analyzer.calculate_momentum_score(data)
            
            # Options processing
            options = test_data_generator.generate_options_chain(symbol, 100 + i)
            criteria = CallOptionCriteria()
            selector = OptionsSelector(criteria)
            best_calls = selector.select_best_calls(options, top_n=3)
            
            # Clean up iteration data
            del data, indicators, momentum, options, best_calls, analyzer, selector
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                gc.collect()
                snapshot = self.take_memory_snapshot(f"iteration_{i}")
                memory_samples.append(snapshot.rss_mb)
                
                if i > 0:
                    print(f"Iteration {i}: {snapshot.rss_mb:.1f}MB RSS")
        
        # Analyze memory trend
        if len(memory_samples) >= 3:
            # Calculate memory trend (should be stable, not increasing)
            start_memory = memory_samples[1]  # Skip first sample (may have initialization)
            end_memory = memory_samples[-1]
            memory_growth = end_memory - start_memory
            
            print(f"Memory growth over {iterations} iterations: {memory_growth:.1f}MB")
            
            # Should not grow by more than 10MB over long run
            assert memory_growth < 10, f"Memory leak detected: {memory_growth:.1f}MB growth"
            
            # Memory should be relatively stable
            memory_variance = max(memory_samples) - min(memory_samples)
            assert memory_variance < 20, f"Memory usage too variable: {memory_variance:.1f}MB range"
    
    @pytest.mark.performance
    def test_concurrent_memory_usage(self, test_data_generator):
        """Test memory usage under concurrent operations."""
        print("\n=== Concurrent Memory Usage Test ===")
        
        self.take_memory_snapshot("concurrent_initial")
        
        import threading
        import queue
        
        # Shared result queue
        results_queue = queue.Queue()
        
        def worker_function(worker_id: int, num_operations: int):
            """Worker function for concurrent testing."""
            try:
                for i in range(num_operations):
                    symbol = f"WORKER{worker_id}_{i:03d}"
                    
                    # Perform operations
                    data = test_data_generator.generate_historical_data(symbol, days=30)
                    indicators = TechnicalIndicators.calculate_all_indicators(data)
                    
                    # Simulate some processing delay
                    time.sleep(0.01)
                    
                    # Clean up
                    del data, indicators
                
                results_queue.put(("success", worker_id))
                
            except Exception as e:
                results_queue.put(("error", worker_id, str(e)))
        
        # Start concurrent workers
        num_workers = 4
        operations_per_worker = 25
        
        before_concurrent = self.take_memory_snapshot("before_concurrent")
        
        threads = []
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=worker_function,
                args=(worker_id, operations_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Monitor memory during execution
        memory_during_execution = []
        while any(t.is_alive() for t in threads):
            memory_during_execution.append(self.process.memory_info().rss / 1024 / 1024)
            time.sleep(0.1)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        after_concurrent = self.take_memory_snapshot("after_concurrent")
        
        # Check results
        successful_workers = 0
        failed_workers = 0
        
        while not results_queue.empty():
            result = results_queue.get()
            if result[0] == "success":
                successful_workers += 1
            else:
                failed_workers += 1
                print(f"Worker {result[1]} failed: {result[2]}")
        
        # All workers should succeed
        assert successful_workers == num_workers, f"Only {successful_workers}/{num_workers} workers succeeded"
        assert failed_workers == 0, f"{failed_workers} workers failed"
        
        # Memory usage analysis
        concurrent_memory = after_concurrent.rss_mb - before_concurrent.rss_mb
        peak_memory = max(memory_during_execution) - before_concurrent.rss_mb
        
        print(f"Concurrent memory usage: {concurrent_memory:.1f}MB final, {peak_memory:.1f}MB peak")
        
        # Memory usage should be reasonable for concurrent operations
        expected_max_memory = num_workers * operations_per_worker * 0.1  # ~0.1MB per operation
        assert peak_memory < max(50, expected_max_memory), \
            f"Peak memory too high: {peak_memory:.1f}MB"
        
        # Clean up and check for leaks
        gc.collect()
        after_cleanup = self.take_memory_snapshot("concurrent_cleanup")
        
        memory_leak = after_cleanup.rss_mb - before_concurrent.rss_mb
        assert memory_leak < 5, f"Concurrent memory leak: {memory_leak:.1f}MB"
    
    def test_object_leak_detection(self, test_data_generator):
        """Test for object leaks using weak references."""
        print("\n=== Object Leak Detection Test ===")
        
        self.take_memory_snapshot("leak_detection_initial")
        
        # Create weak references to track object cleanup
        weak_refs = []
        
        def create_and_process_data():
            """Create objects and add weak references."""
            data = test_data_generator.generate_historical_data("LEAK_TEST", days=100)
            weak_refs.append(weakref.ref(data))
            
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            weak_refs.append(weakref.ref(indicators))
            
            analyzer = MomentumAnalyzer()
            weak_refs.append(weakref.ref(analyzer))
            
            momentum = analyzer.calculate_momentum_score(data)
            weak_refs.append(weakref.ref(momentum))
            
            return len(weak_refs)
        
        # Create objects multiple times
        for i in range(10):
            create_and_process_data()
        
        before_gc = self.take_memory_snapshot("before_gc")
        
        # Force garbage collection
        gc.collect()
        
        after_gc = self.take_memory_snapshot("after_gc")
        
        # Check weak references
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        total_objects = len(weak_refs)
        
        print(f"Objects still alive after GC: {alive_objects}/{total_objects}")
        
        # Most objects should be garbage collected
        max_alive = total_objects * 0.1  # Allow 10% to remain (e.g., cached objects)
        assert alive_objects <= max_alive, \
            f"Too many objects still alive: {alive_objects}/{total_objects}"
        
        # Memory should decrease after GC
        memory_freed = before_gc.rss_mb - after_gc.rss_mb
        print(f"Memory freed by GC: {memory_freed:.1f}MB")
    
    def generate_memory_report(self) -> str:
        """Generate memory performance report."""
        if not self.snapshots:
            return "No memory snapshots available."
        
        report_lines = [
            "Memory Performance Report",
            "=" * 40,
            ""
        ]
        
        # Memory usage summary
        initial_memory = self.snapshots[0].rss_mb
        final_memory = self.snapshots[-1].rss_mb
        peak_memory = max(snapshot.rss_mb for snapshot in self.snapshots)
        
        report_lines.extend([
            f"Memory Usage Summary:",
            f"  Initial: {initial_memory:.1f}MB",
            f"  Final: {final_memory:.1f}MB", 
            f"  Peak: {peak_memory:.1f}MB",
            f"  Net Change: {final_memory - initial_memory:+.1f}MB",
            ""
        ])
        
        # Object count summary
        initial_objects = self.snapshots[0].num_objects
        final_objects = self.snapshots[-1].num_objects
        peak_objects = max(snapshot.num_objects for snapshot in self.snapshots)
        
        report_lines.extend([
            f"Object Count Summary:",
            f"  Initial: {initial_objects:,}",
            f"  Final: {final_objects:,}",
            f"  Peak: {peak_objects:,}",
            f"  Net Change: {final_objects - initial_objects:+,}",
            ""
        ])
        
        # Memory efficiency metrics
        if len(self.snapshots) > 1:
            memory_growth_rate = (final_memory - initial_memory) / len(self.snapshots)
            object_growth_rate = (final_objects - initial_objects) / len(self.snapshots)
            
            report_lines.extend([
                f"Growth Rates:",
                f"  Memory: {memory_growth_rate:+.2f}MB per snapshot",
                f"  Objects: {object_growth_rate:+.0f} per snapshot",
                ""
            ])
        
        return "\n".join(report_lines)