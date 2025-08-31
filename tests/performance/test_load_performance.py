"""Load testing and performance benchmarks for the options screening system."""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any
import psutil
import os

# Import modules for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tests.utils.performance_helpers import PerformanceMonitor, LoadTestHelper, BenchmarkRunner
from screener.workflow_engine import WorkflowEngine, StepResult
from models.workflow_models import WorkflowConfig
from providers.provider_manager import ProviderManager
from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria


@dataclass
class LoadTestResult:
    """Results from load testing."""
    test_name: str
    total_operations: int
    concurrent_workers: int
    total_time_seconds: float
    throughput_ops_per_sec: float
    success_rate: float
    average_response_time: float
    error_count: int
    memory_usage_mb: float
    cpu_usage_percent: float


class LoadTestSuite:
    """Comprehensive load testing suite."""
    
    def __init__(self):
        self.results: List[LoadTestResult] = []
        self.benchmark_runner = BenchmarkRunner()
        self.load_helper = LoadTestHelper()
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_technical_indicators_load(self, test_data_generator, performance_timer):
        """Load test technical indicators calculation."""
        print("\n=== Technical Indicators Load Test ===")
        
        # Generate test data
        test_datasets = []
        for i in range(50):
            data = test_data_generator.generate_historical_data(f"TEST{i:03d}", days=100)
            test_datasets.append(data)
        
        def calculate_indicators(dataset):
            """Calculate all technical indicators for a dataset."""
            return TechnicalIndicators.calculate_all_indicators(dataset)
        
        # Perform load test
        result = self.load_helper.run_concurrent_test(
            calculate_indicators,
            num_concurrent=10,
            num_iterations=50,
            test_datasets[0]  # Use first dataset as template
        )
        
        # Verify performance
        assert result["success_rate"] > 95  # Should have high success rate
        assert result["throughput_ops_per_sec"] > 5  # At least 5 calculations per second
        
        load_result = LoadTestResult(
            test_name="technical_indicators_load",
            total_operations=result["total_operations"],
            concurrent_workers=result["concurrent_workers"],
            total_time_seconds=result["total_time"],
            throughput_ops_per_sec=result["throughput_ops_per_sec"],
            success_rate=result["success_rate"],
            average_response_time=result["avg_response_time"],
            error_count=result["errors"],
            memory_usage_mb=result["performance_metrics"].memory_usage_mb,
            cpu_usage_percent=result["performance_metrics"].cpu_usage_percent
        )
        
        self.results.append(load_result)
        
        print(f"Throughput: {load_result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"Success Rate: {load_result.success_rate:.1f}%")
        print(f"Memory Usage: {load_result.memory_usage_mb:.1f} MB")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_options_selector_load(self, test_data_generator):
        """Load test options selection functionality."""
        print("\n=== Options Selector Load Test ===")
        
        # Set up options selector
        criteria = CallOptionCriteria(
            min_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45,
            min_delta=0.3,
            max_delta=0.8
        )
        selector = OptionsSelector(criteria)
        
        # Generate test options chains
        test_chains = []
        for i in range(30):
            chain = test_data_generator.generate_options_chain(f"TEST{i:03d}", 100 + i*5)
            test_chains.append(chain)
        
        def select_best_options(options_chain):
            """Select best options from chain."""
            return selector.select_best_calls(options_chain, top_n=5)
        
        # Perform load test
        result = self.load_helper.run_concurrent_test(
            select_best_options,
            num_concurrent=5,
            num_iterations=30,
            test_chains[0]  # Use first chain as template
        )
        
        # Verify performance
        assert result["success_rate"] > 90  # Should have high success rate
        assert result["throughput_ops_per_sec"] > 2  # At least 2 selections per second
        
        load_result = LoadTestResult(
            test_name="options_selector_load",
            total_operations=result["total_operations"],
            concurrent_workers=result["concurrent_workers"],
            total_time_seconds=result["total_time"],
            throughput_ops_per_sec=result["throughput_ops_per_sec"],
            success_rate=result["success_rate"],
            average_response_time=result["avg_response_time"],
            error_count=result["errors"],
            memory_usage_mb=result["performance_metrics"].memory_usage_mb,
            cpu_usage_percent=result["performance_metrics"].cpu_usage_percent
        )
        
        self.results.append(load_result)
        
        print(f"Throughput: {load_result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"Success Rate: {load_result.success_rate:.1f}%")
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_workflow_engine_load(self, test_data_generator):
        """Load test workflow engine with multiple concurrent workflows."""
        print("\n=== Workflow Engine Load Test ===")
        
        # Set up workflow engine
        config = WorkflowConfig(
            max_concurrent_stocks=5,
            max_retry_attempts=1,
            step_timeout_seconds=10,
            continue_on_errors=True,
            enable_ai_analysis=False  # Disable for performance testing
        )
        engine = WorkflowEngine(config)
        
        # Register simple test step
        async def mock_processing_step(context, symbols):
            await asyncio.sleep(0.1)  # Simulate processing time
            return StepResult(
                step_id="mock_step",
                success=True,
                results={"processed": len(symbols)},
                execution_time_seconds=0.1
            )
        
        engine.register_step("mock_step", mock_processing_step, order=1)
        
        # Define async test function
        async def run_workflow():
            symbols = test_data_generator.generate_stock_symbols(5)
            context = await engine.execute_workflow(symbols)
            return context.state.value
        
        # Perform async load test
        result = await self.load_helper.run_async_concurrent_test(
            run_workflow,
            num_concurrent=3,
            num_iterations=15
        )
        
        # Verify performance
        assert result["success_rate"] > 90
        assert result["throughput_ops_per_sec"] > 1  # At least 1 workflow per second
        
        load_result = LoadTestResult(
            test_name="workflow_engine_load",
            total_operations=result["total_operations"],
            concurrent_workers=result["concurrent_workers"],
            total_time_seconds=result["total_time"],
            throughput_ops_per_sec=result["throughput_ops_per_sec"],
            success_rate=result["success_rate"],
            average_response_time=result["avg_response_time"],
            error_count=result["errors"],
            memory_usage_mb=result["performance_metrics"].memory_usage_mb,
            cpu_usage_percent=result["performance_metrics"].cpu_usage_percent
        )
        
        self.results.append(load_result)
        
        print(f"Throughput: {load_result.throughput_ops_per_sec:.2f} workflows/sec")
        print(f"Success Rate: {load_result.success_rate:.1f}%")
    
    @pytest.mark.performance
    def test_momentum_analysis_scale(self, test_data_generator):
        """Test momentum analysis scaling with different data sizes."""
        print("\n=== Momentum Analysis Scaling Test ===")
        
        analyzer = MomentumAnalyzer()
        
        # Test with different data sizes
        data_sizes = [30, 60, 120, 252]  # Days of data
        scaling_results = []
        
        for days in data_sizes:
            # Generate data
            price_data = test_data_generator.generate_historical_data("SCALE_TEST", days=days)
            
            # Benchmark momentum calculation
            result = self.benchmark_runner.benchmark_function(
                f"momentum_analysis_{days}d",
                analyzer.calculate_momentum_score,
                price_data,
                iterations=20
            )
            
            scaling_results.append({
                "data_size_days": days,
                "avg_time_seconds": result["timing"]["average"],
                "throughput": result["throughput"]
            })
        
        # Verify scaling is reasonable (should be roughly linear or better)
        for i in range(1, len(scaling_results)):
            prev = scaling_results[i-1]
            curr = scaling_results[i]
            
            size_ratio = curr["data_size_days"] / prev["data_size_days"]
            time_ratio = curr["avg_time_seconds"] / prev["avg_time_seconds"]
            
            # Time should not increase faster than quadratic with data size
            assert time_ratio < size_ratio ** 2, f"Poor scaling: {size_ratio}x data -> {time_ratio}x time"
        
        print("Scaling Results:")
        for result in scaling_results:
            print(f"  {result['data_size_days']} days: {result['avg_time_seconds']:.4f}s avg")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_symbol_processing(self, test_data_generator):
        """Test concurrent processing of multiple symbols."""
        print("\n=== Concurrent Symbol Processing Test ===")
        
        # Generate large number of symbols
        symbols = test_data_generator.generate_stock_symbols(100)
        
        def process_symbol(symbol):
            """Simulate symbol processing."""
            # Generate and process data for symbol
            price_data = test_data_generator.generate_historical_data(symbol, days=60)
            indicators = TechnicalIndicators.calculate_all_indicators(price_data)
            
            # Simulate some processing time
            time.sleep(0.01)
            
            return {
                "symbol": symbol,
                "rsi": indicators.get("rsi_14", 50),
                "processed_at": time.time()
            }
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        concurrency_results = []
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(process_symbol, symbol) for symbol in symbols[:50]]
                results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            execution_time = end_time - start_time
            throughput = len(results) / execution_time
            
            concurrency_results.append({
                "concurrency": concurrency,
                "execution_time": execution_time,
                "throughput": throughput,
                "symbols_processed": len(results)
            })
        
        # Verify concurrency improves performance
        sequential_time = concurrency_results[0]["execution_time"]
        best_concurrent_time = min(r["execution_time"] for r in concurrency_results[1:])
        
        improvement_ratio = sequential_time / best_concurrent_time
        assert improvement_ratio > 1.5, f"Concurrency should improve performance, got {improvement_ratio}x"
        
        print("Concurrency Results:")
        for result in concurrency_results:
            print(f"  {result['concurrency']} workers: {result['execution_time']:.2f}s, "
                  f"{result['throughput']:.1f} symbols/sec")
    
    @pytest.mark.performance
    def test_memory_efficiency_large_datasets(self, test_data_generator, memory_monitor):
        """Test memory efficiency with large datasets."""
        print("\n=== Memory Efficiency Test ===")
        
        initial_memory = memory_monitor()
        
        # Process increasingly large datasets
        dataset_sizes = [100, 500, 1000, 2000]
        memory_usage = []
        
        for size in dataset_sizes:
            # Generate large dataset
            large_data = test_data_generator.generate_historical_data("MEMORY_TEST", days=size)
            
            # Process data
            indicators = TechnicalIndicators.calculate_all_indicators(large_data)
            
            current_memory = memory_monitor()
            memory_increase = current_memory - initial_memory
            memory_usage.append({
                "dataset_size": size,
                "memory_mb": memory_increase,
                "memory_per_datapoint": memory_increase / size
            })
            
            # Clean up
            del large_data, indicators
        
        # Verify memory usage is reasonable
        final_memory = memory_monitor()
        total_increase = final_memory - initial_memory
        
        # Should not use more than 500MB for test
        assert total_increase < 500, f"Memory usage too high: {total_increase:.1f}MB"
        
        # Memory per datapoint should be relatively stable
        memory_per_point = [m["memory_per_datapoint"] for m in memory_usage]
        memory_variation = statistics.stdev(memory_per_point) if len(memory_per_point) > 1 else 0
        
        print("Memory Usage Results:")
        for usage in memory_usage:
            print(f"  {usage['dataset_size']} points: {usage['memory_mb']:.1f}MB "
                  f"({usage['memory_per_datapoint']:.4f}MB/point)")
        print(f"Memory per datapoint variation: {memory_variation:.4f}MB")
    
    def generate_load_test_report(self) -> str:
        """Generate comprehensive load test report."""
        if not self.results:
            return "No load test results available."
        
        report_lines = [
            "Load Test Performance Report",
            "=" * 40,
            ""
        ]
        
        # Summary statistics
        total_operations = sum(r.total_operations for r in self.results)
        avg_throughput = statistics.mean(r.throughput_ops_per_sec for r in self.results)
        avg_success_rate = statistics.mean(r.success_rate for r in self.results)
        
        report_lines.extend([
            f"Summary:",
            f"  Total Operations: {total_operations:,}",
            f"  Average Throughput: {avg_throughput:.2f} ops/sec",
            f"  Average Success Rate: {avg_success_rate:.1f}%",
            ""
        ])
        
        # Individual test results
        report_lines.append("Individual Test Results:")
        for result in self.results:
            report_lines.extend([
                f"  {result.test_name}:",
                f"    Throughput: {result.throughput_ops_per_sec:.2f} ops/sec",
                f"    Success Rate: {result.success_rate:.1f}%",
                f"    Avg Response Time: {result.average_response_time:.3f}s",
                f"    Memory Usage: {result.memory_usage_mb:.1f} MB",
                f"    CPU Usage: {result.cpu_usage_percent:.1f}%",
                ""
            ])
        
        # Performance benchmarks
        benchmark_data = self.benchmark_runner.benchmarks
        if benchmark_data:
            report_lines.extend([
                "Benchmark Results:",
                ""
            ])
            
            for name, benchmark in benchmark_data.items():
                report_lines.extend([
                    f"  {name}:",
                    f"    Average Time: {benchmark['timing']['average']:.4f}s",
                    f"    Throughput: {benchmark['throughput']:.2f} ops/sec",
                    f"    Success Rate: {benchmark['success_rate']:.1f}%",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    @pytest.mark.performance
    def test_stress_test_resource_limits(self, test_data_generator):
        """Stress test to find resource limits."""
        print("\n=== Stress Test - Resource Limits ===")
        
        # Monitor system resources
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Gradually increase load until we hit limits
        max_symbols = 1000
        batch_size = 50
        
        successful_batches = 0
        failed_batches = 0
        
        for batch_start in range(0, max_symbols, batch_size):
            try:
                batch_symbols = test_data_generator.generate_stock_symbols(batch_size)
                
                # Process batch
                start_time = time.time()
                
                batch_results = []
                for symbol in batch_symbols:
                    data = test_data_generator.generate_historical_data(symbol, days=30)
                    indicators = TechnicalIndicators.calculate_all_indicators(data)
                    batch_results.append(indicators)
                
                batch_time = time.time() - start_time
                current_memory = process.memory_info().rss / 1024 / 1024
                
                # Check resource limits
                memory_usage = current_memory - initial_memory
                
                if memory_usage > 1000:  # 1GB limit
                    print(f"Memory limit reached: {memory_usage:.1f}MB")
                    break
                
                if batch_time > 60:  # 60 second limit per batch
                    print(f"Time limit reached: {batch_time:.1f}s per batch")
                    break
                
                successful_batches += 1
                
                if successful_batches % 5 == 0:  # Progress report every 5 batches
                    print(f"Processed {successful_batches * batch_size} symbols, "
                          f"Memory: {memory_usage:.1f}MB")
                
            except Exception as e:
                failed_batches += 1
                print(f"Batch failed: {e}")
                
                if failed_batches > 3:  # Stop after 3 consecutive failures
                    break
        
        total_processed = successful_batches * batch_size
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Stress Test Results:")
        print(f"  Successfully processed: {total_processed} symbols")
        print(f"  Failed batches: {failed_batches}")
        print(f"  Final memory usage: {final_memory - initial_memory:.1f}MB")
        
        # Should handle at least 100 symbols without issues
        assert total_processed >= 100, f"Should handle at least 100 symbols, got {total_processed}"


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression testing."""
    
    def test_technical_indicators_performance_baseline(self, test_data_generator):
        """Establish baseline performance for technical indicators."""
        # This test establishes performance baselines
        # Future changes should not significantly degrade performance
        
        data = test_data_generator.generate_historical_data("BASELINE", days=252)  # 1 year
        
        start_time = time.time()
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Baseline: Should complete 1 year of data in under 1 second
        assert execution_time < 1.0, f"Performance regression: {execution_time:.3f}s > 1.0s baseline"
        
        # Verify all indicators calculated
        expected_indicators = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr_14', 'volume_sma_20', 'volume_ratio'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators, f"Missing indicator: {indicator}"
        
        print(f"Technical indicators baseline: {execution_time:.3f}s for 252 days")
    
    def test_options_selector_performance_baseline(self, test_data_generator):
        """Establish baseline performance for options selector."""
        criteria = CallOptionCriteria(
            min_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45
        )
        selector = OptionsSelector(criteria)
        
        # Large options chain (21 strikes across multiple expirations)
        options_chain = test_data_generator.generate_options_chain("BASELINE", 150.0, 30)
        
        start_time = time.time()
        best_calls = selector.select_best_calls(options_chain, top_n=5)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Baseline: Should complete options selection in under 100ms
        assert execution_time < 0.1, f"Performance regression: {execution_time:.3f}s > 0.1s baseline"
        
        # Should return reasonable results
        assert len(best_calls) > 0
        assert len(best_calls) <= 5
        
        print(f"Options selector baseline: {execution_time:.3f}s for {len(options_chain)} options")
    
    def test_momentum_analysis_performance_baseline(self, test_data_generator):
        """Establish baseline performance for momentum analysis."""
        analyzer = MomentumAnalyzer()
        
        # 3 months of data
        price_data = test_data_generator.generate_historical_data("BASELINE", days=90)
        
        start_time = time.time()
        momentum_score = analyzer.calculate_momentum_score(price_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Baseline: Should complete momentum analysis in under 500ms
        assert execution_time < 0.5, f"Performance regression: {execution_time:.3f}s > 0.5s baseline"
        
        # Should return valid momentum score
        assert 'momentum_score' in momentum_score
        assert 0 <= momentum_score['momentum_score'] <= 100
        
        print(f"Momentum analysis baseline: {execution_time:.3f}s for 90 days")