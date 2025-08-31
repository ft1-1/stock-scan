"""Performance testing utilities for load testing and benchmarking."""

import time
import asyncio
import psutil
import os
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    throughput_ops_per_sec: float
    error_count: int
    success_count: int


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self.measurements: List[Dict] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous monitoring of system resources."""
        self.start_time = time.perf_counter()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self._monitoring = True
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return performance metrics."""
        self.end_time = time.perf_counter()
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        execution_time = self.end_time - (self.start_time or 0)
        final_memory = self._get_memory_usage()
        
        # Calculate average CPU usage
        avg_cpu = statistics.mean([m["cpu_percent"] for m in self.measurements]) if self.measurements else 0.0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=final_memory - (self.start_memory or 0),
            cpu_usage_percent=avg_cpu,
            peak_memory_mb=self.peak_memory,
            throughput_ops_per_sec=0.0,  # To be calculated by caller
            error_count=0,  # To be set by caller
            success_count=0  # To be set by caller
        )
    
    def _monitor_loop(self, interval: float):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                current_memory = self._get_memory_usage()
                cpu_percent = self.process.cpu_percent()
                
                self.peak_memory = max(self.peak_memory, current_memory)
                
                self.measurements.append({
                    "timestamp": time.perf_counter(),
                    "memory_mb": current_memory,
                    "cpu_percent": cpu_percent
                })
                
                time.sleep(interval)
            except psutil.NoSuchProcess:
                break
            except Exception:
                # Ignore monitoring errors
                pass
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    @contextmanager
    def measure_block(self):
        """Context manager for measuring a code block."""
        self.start_monitoring()
        try:
            yield self
        finally:
            metrics = self.stop_monitoring()
            self.last_metrics = metrics


class LoadTestHelper:
    """Helper for conducting load tests with multiple concurrent operations."""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def run_concurrent_test(
        self,
        test_function: Callable,
        num_concurrent: int,
        num_iterations: int,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Run concurrent load test with given function."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.perf_counter()
        errors = []
        successes = 0
        
        try:
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                # Submit all tasks
                futures = []
                for _ in range(num_iterations):
                    future = executor.submit(test_function, *args, **kwargs)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        successes += 1
                        self.results.append({"success": True, "result": result})
                    except Exception as e:
                        errors.append(str(e))
                        self.results.append({"success": False, "error": str(e)})
        
        except Exception as e:
            errors.append(f"Load test setup error: {e}")
        
        end_time = time.perf_counter()
        metrics = monitor.stop_monitoring()
        
        total_time = end_time - start_time
        throughput = successes / total_time if total_time > 0 else 0
        
        return {
            "total_operations": num_iterations,
            "concurrent_workers": num_concurrent,
            "successes": successes,
            "errors": len(errors),
            "error_details": errors[:10],  # Limit error details
            "total_time": total_time,
            "throughput_ops_per_sec": throughput,
            "avg_response_time": total_time / num_iterations if num_iterations > 0 else 0,
            "performance_metrics": metrics,
            "success_rate": (successes / num_iterations) * 100 if num_iterations > 0 else 0
        }
    
    async def run_async_concurrent_test(
        self,
        async_test_function: Callable,
        num_concurrent: int,
        num_iterations: int,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Run concurrent load test with async function."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.perf_counter()
        errors = []
        successes = 0
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(num_concurrent)
        
        async def bounded_test():
            async with semaphore:
                return await async_test_function(*args, **kwargs)
        
        try:
            # Run all tasks concurrently
            tasks = [bounded_test() for _ in range(num_iterations)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                    self.results.append({"success": False, "error": str(result)})
                else:
                    successes += 1
                    self.results.append({"success": True, "result": result})
        
        except Exception as e:
            errors.append(f"Async load test error: {e}")
        
        end_time = time.perf_counter()
        metrics = monitor.stop_monitoring()
        
        total_time = end_time - start_time
        throughput = successes / total_time if total_time > 0 else 0
        
        return {
            "total_operations": num_iterations,
            "concurrent_workers": num_concurrent,
            "successes": successes,
            "errors": len(errors),
            "error_details": errors[:10],
            "total_time": total_time,
            "throughput_ops_per_sec": throughput,
            "avg_response_time": total_time / num_iterations if num_iterations > 0 else 0,
            "performance_metrics": metrics,
            "success_rate": (successes / num_iterations) * 100 if num_iterations > 0 else 0
        }
    
    def generate_load_profile(
        self,
        test_function: Callable,
        max_concurrent: int = 50,
        step_size: int = 5,
        iterations_per_step: int = 20
    ) -> List[Dict]:
        """Generate performance profile across different load levels."""
        profile_results = []
        
        for concurrent_users in range(step_size, max_concurrent + 1, step_size):
            print(f"Testing with {concurrent_users} concurrent users...")
            
            result = self.run_concurrent_test(
                test_function,
                concurrent_users,
                iterations_per_step
            )
            
            result["concurrent_users"] = concurrent_users
            profile_results.append(result)
            
            # Brief pause between load levels
            time.sleep(1)
        
        return profile_results
    
    def analyze_performance_profile(self, profile_results: List[Dict]) -> Dict:
        """Analyze performance profile results to find bottlenecks."""
        if not profile_results:
            return {}
        
        # Find optimal concurrency level
        best_throughput = 0
        optimal_concurrency = 0
        
        for result in profile_results:
            if result["throughput_ops_per_sec"] > best_throughput:
                best_throughput = result["throughput_ops_per_sec"]
                optimal_concurrency = result["concurrent_users"]
        
        # Calculate degradation points
        degradation_points = []
        prev_throughput = 0
        
        for result in profile_results:
            current_throughput = result["throughput_ops_per_sec"]
            if prev_throughput > 0 and current_throughput < prev_throughput * 0.9:  # 10% drop
                degradation_points.append(result["concurrent_users"])
            prev_throughput = current_throughput
        
        return {
            "optimal_concurrency": optimal_concurrency,
            "max_throughput": best_throughput,
            "degradation_points": degradation_points,
            "total_tests": len(profile_results),
            "performance_summary": {
                "min_throughput": min(r["throughput_ops_per_sec"] for r in profile_results),
                "max_throughput": max(r["throughput_ops_per_sec"] for r in profile_results),
                "avg_throughput": statistics.mean(r["throughput_ops_per_sec"] for r in profile_results)
            }
        }


class BenchmarkRunner:
    """Run benchmark tests and compare performance."""
    
    def __init__(self):
        self.benchmarks: Dict[str, Dict] = {}
    
    def benchmark_function(
        self,
        name: str,
        func: Callable,
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> Dict:
        """Benchmark a function with multiple iterations."""
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark
        execution_times = []
        errors = 0
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                errors += 1
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)  # Include failed attempts
        
        metrics = monitor.stop_monitoring()
        
        # Calculate statistics
        if execution_times:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        else:
            avg_time = median_time = min_time = max_time = std_dev = 0
        
        benchmark_result = {
            "name": name,
            "iterations": iterations,
            "errors": errors,
            "success_rate": ((iterations - errors) / iterations) * 100,
            "timing": {
                "average": avg_time,
                "median": median_time,
                "min": min_time,
                "max": max_time,
                "std_dev": std_dev
            },
            "performance_metrics": metrics,
            "throughput": iterations / metrics.execution_time if metrics.execution_time > 0 else 0
        }
        
        self.benchmarks[name] = benchmark_result
        return benchmark_result
    
    async def benchmark_async_function(
        self,
        name: str,
        async_func: Callable,
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> Dict:
        """Benchmark an async function."""
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                await async_func(*args, **kwargs)
            except Exception:
                pass
        
        # Actual benchmark
        execution_times = []
        errors = 0
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                result = await async_func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                errors += 1
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
        
        metrics = monitor.stop_monitoring()
        
        # Calculate statistics
        if execution_times:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        else:
            avg_time = median_time = min_time = max_time = std_dev = 0
        
        benchmark_result = {
            "name": name,
            "iterations": iterations,
            "errors": errors,
            "success_rate": ((iterations - errors) / iterations) * 100,
            "timing": {
                "average": avg_time,
                "median": median_time,
                "min": min_time,
                "max": max_time,
                "std_dev": std_dev
            },
            "performance_metrics": metrics,
            "throughput": iterations / metrics.execution_time if metrics.execution_time > 0 else 0
        }
        
        self.benchmarks[name] = benchmark_result
        return benchmark_result
    
    def compare_benchmarks(self, baseline_name: str, comparison_names: List[str]) -> Dict:
        """Compare benchmark results against a baseline."""
        if baseline_name not in self.benchmarks:
            raise ValueError(f"Baseline benchmark '{baseline_name}' not found")
        
        baseline = self.benchmarks[baseline_name]
        comparisons = {}
        
        for name in comparison_names:
            if name not in self.benchmarks:
                continue
            
            comparison = self.benchmarks[name]
            
            # Calculate performance ratios
            timing_ratio = comparison["timing"]["average"] / baseline["timing"]["average"]
            throughput_ratio = comparison["throughput"] / baseline["throughput"] if baseline["throughput"] > 0 else 0
            
            comparisons[name] = {
                "timing_ratio": timing_ratio,  # >1 means slower, <1 means faster
                "throughput_ratio": throughput_ratio,  # >1 means faster, <1 means slower
                "timing_change_percent": (timing_ratio - 1) * 100,
                "throughput_change_percent": (throughput_ratio - 1) * 100,
                "faster": timing_ratio < 1,
                "benchmark_data": comparison
            }
        
        return {
            "baseline": baseline_name,
            "comparisons": comparisons,
            "summary": {
                "fastest": min(comparisons.keys(), key=lambda k: comparisons[k]["timing_ratio"]) if comparisons else None,
                "highest_throughput": max(comparisons.keys(), key=lambda k: comparisons[k]["throughput_ratio"]) if comparisons else None
            }
        }
    
    def generate_performance_report(self) -> str:
        """Generate a formatted performance report."""
        if not self.benchmarks:
            return "No benchmarks to report."
        
        report = ["Performance Benchmark Report", "=" * 30, ""]
        
        for name, benchmark in self.benchmarks.items():
            report.extend([
                f"Benchmark: {name}",
                f"  Iterations: {benchmark['iterations']}",
                f"  Success Rate: {benchmark['success_rate']:.1f}%",
                f"  Average Time: {benchmark['timing']['average']:.4f}s",
                f"  Median Time: {benchmark['timing']['median']:.4f}s",
                f"  Throughput: {benchmark['throughput']:.2f} ops/sec",
                f"  Memory Usage: {benchmark['performance_metrics'].memory_usage_mb:.2f} MB",
                ""
            ])
        
        return "\n".join(report)