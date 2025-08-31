"""Benchmarking tests for analytics components performance."""

import pytest
import time
import numpy as np
import pandas as pd
import statistics
from typing import Dict, List, Tuple
import gc
import psutil
import os

# Import analytics modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria
from analytics.scoring_models import TechnicalScorer, MomentumScorer, CombinedScorer, ScoreWeights
from analytics.squeeze_detector import SqueezeDetector
from analytics.greeks_calculator import GreeksCalculator
from tests.utils.performance_helpers import BenchmarkRunner, PerformanceMonitor


class AnalyticsBenchmarks:
    """Comprehensive benchmarking suite for analytics components."""
    
    def __init__(self):
        self.benchmark_runner = BenchmarkRunner()
        self.results: Dict[str, Dict] = {}
    
    @pytest.mark.performance
    def test_technical_indicators_benchmarks(self, test_data_generator):
        """Benchmark all technical indicators with various data sizes."""
        print("\n=== Technical Indicators Benchmarks ===")
        
        data_sizes = [50, 100, 252, 500, 1000]  # Days of data
        benchmark_results = {}
        
        for days in data_sizes:
            print(f"Benchmarking with {days} days of data...")
            
            # Generate test data
            test_data = test_data_generator.generate_historical_data("BENCH", days=days)
            
            # Benchmark individual indicators
            individual_benchmarks = {}
            
            # SMA benchmarks
            individual_benchmarks['sma_20'] = self.benchmark_runner.benchmark_function(
                f"sma_20_{days}d",
                TechnicalIndicators.sma,
                test_data['close'], 20,
                iterations=100
            )
            
            # EMA benchmarks
            individual_benchmarks['ema_12'] = self.benchmark_runner.benchmark_function(
                f"ema_12_{days}d",
                TechnicalIndicators.ema,
                test_data['close'], 12,
                iterations=100
            )
            
            # RSI benchmarks
            individual_benchmarks['rsi_14'] = self.benchmark_runner.benchmark_function(
                f"rsi_14_{days}d",
                TechnicalIndicators.rsi,
                test_data['close'], 14,
                iterations=50
            )
            
            # MACD benchmarks
            individual_benchmarks['macd'] = self.benchmark_runner.benchmark_function(
                f"macd_{days}d",
                TechnicalIndicators.macd,
                test_data['close'],
                iterations=50
            )
            
            # Bollinger Bands benchmarks
            individual_benchmarks['bollinger'] = self.benchmark_runner.benchmark_function(
                f"bollinger_{days}d",
                TechnicalIndicators.bollinger_bands,
                test_data['close'], 20, 2,
                iterations=50
            )
            
            # ATR benchmarks
            individual_benchmarks['atr'] = self.benchmark_runner.benchmark_function(
                f"atr_{days}d",
                TechnicalIndicators.atr,
                test_data['high'], test_data['low'], test_data['close'], 14,
                iterations=50
            )
            
            # Complete calculation benchmark
            individual_benchmarks['all_indicators'] = self.benchmark_runner.benchmark_function(
                f"all_indicators_{days}d",
                TechnicalIndicators.calculate_all_indicators,
                test_data,
                iterations=20
            )
            
            benchmark_results[days] = individual_benchmarks
        
        # Analyze scaling characteristics
        self._analyze_scaling_performance(benchmark_results, "Technical Indicators")
        
        # Performance assertions
        for days, benchmarks in benchmark_results.items():
            # All indicators together should complete in reasonable time
            all_indicators_time = benchmarks['all_indicators']['timing']['average']
            
            # Should scale roughly linearly with data size
            expected_max_time = days * 0.00005  # 0.05ms per data point
            assert all_indicators_time < max(0.1, expected_max_time), \
                f"Technical indicators too slow: {all_indicators_time:.4f}s for {days} days"
        
        self.results['technical_indicators'] = benchmark_results
        
        print("Technical Indicators Benchmark Summary:")
        for days in data_sizes:
            all_time = benchmark_results[days]['all_indicators']['timing']['average']
            throughput = benchmark_results[days]['all_indicators']['throughput']
            print(f"  {days} days: {all_time:.4f}s avg, {throughput:.1f} ops/sec")
    
    @pytest.mark.performance
    def test_momentum_analysis_benchmarks(self, test_data_generator):
        """Benchmark momentum analysis components."""
        print("\n=== Momentum Analysis Benchmarks ===")
        
        analyzer = MomentumAnalyzer()
        data_sizes = [30, 60, 90, 120, 252]
        benchmark_results = {}
        
        for days in data_sizes:
            print(f"Benchmarking momentum analysis with {days} days...")
            
            test_data = test_data_generator.generate_historical_data("MOMENTUM", days=days)
            
            # Benchmark momentum score calculation
            momentum_benchmark = self.benchmark_runner.benchmark_function(
                f"momentum_score_{days}d",
                analyzer.calculate_momentum_score,
                test_data,
                iterations=50
            )
            
            # Benchmark price momentum analysis
            price_momentum_benchmark = self.benchmark_runner.benchmark_function(
                f"price_momentum_{days}d",
                analyzer.analyze_price_momentum,
                test_data['close'],
                [5, 10, 20, 30],
                iterations=50
            )
            
            # Benchmark volume momentum analysis
            volume_momentum_benchmark = self.benchmark_runner.benchmark_function(
                f"volume_momentum_{days}d",
                analyzer.analyze_volume_momentum,
                test_data['close'], test_data['volume'],
                iterations=50
            )
            
            benchmark_results[days] = {
                'momentum_score': momentum_benchmark,
                'price_momentum': price_momentum_benchmark,
                'volume_momentum': volume_momentum_benchmark
            }
        
        # Performance assertions
        for days, benchmarks in benchmark_results.items():
            momentum_time = benchmarks['momentum_score']['timing']['average']
            
            # Should complete momentum analysis quickly
            expected_max_time = days * 0.0001  # 0.1ms per data point
            assert momentum_time < max(0.2, expected_max_time), \
                f"Momentum analysis too slow: {momentum_time:.4f}s for {days} days"
        
        self.results['momentum_analysis'] = benchmark_results
        
        print("Momentum Analysis Benchmark Summary:")
        for days in data_sizes:
            momentum_time = benchmark_results[days]['momentum_score']['timing']['average']
            throughput = benchmark_results[days]['momentum_score']['throughput']
            print(f"  {days} days: {momentum_time:.4f}s avg, {throughput:.1f} ops/sec")
    
    @pytest.mark.performance
    def test_options_selector_benchmarks(self, test_data_generator):
        """Benchmark options selection algorithms."""
        print("\n=== Options Selector Benchmarks ===")
        
        criteria = CallOptionCriteria(
            min_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45,
            min_delta=0.2,
            max_delta=0.8
        )
        selector = OptionsSelector(criteria)
        
        # Test with different options chain sizes
        chain_sizes = [10, 21, 50, 100, 200]  # Number of options
        benchmark_results = {}
        
        for size in chain_sizes:
            print(f"Benchmarking options selection with {size} options...")
            
            # Generate options chain
            options_chain = []
            for i in range(size):
                strike_price = 100 + (i - size//2) * 5  # Range of strikes
                option_data = test_data_generator.generate_options_chain("BENCH", strike_price, 30)[0]
                options_chain.append(option_data)
            
            # Benchmark volume filtering
            volume_filter_benchmark = self.benchmark_runner.benchmark_function(
                f"volume_filter_{size}",
                selector.filter_by_volume,
                options_chain,
                iterations=100
            )
            
            # Benchmark delta filtering
            delta_filter_benchmark = self.benchmark_runner.benchmark_function(
                f"delta_filter_{size}",
                selector.filter_by_delta,
                options_chain,
                iterations=100
            )
            
            # Benchmark liquidity scoring
            def score_all_options(chain):
                return [selector.calculate_liquidity_score(opt) for opt in chain]
            
            liquidity_score_benchmark = self.benchmark_runner.benchmark_function(
                f"liquidity_score_{size}",
                score_all_options,
                options_chain,
                iterations=50
            )
            
            # Benchmark complete selection
            best_calls_benchmark = self.benchmark_runner.benchmark_function(
                f"best_calls_{size}",
                selector.select_best_calls,
                options_chain, 5,
                iterations=50
            )
            
            benchmark_results[size] = {
                'volume_filter': volume_filter_benchmark,
                'delta_filter': delta_filter_benchmark,
                'liquidity_score': liquidity_score_benchmark,
                'best_calls': best_calls_benchmark
            }
        
        # Performance assertions
        for size, benchmarks in benchmark_results.items():
            best_calls_time = benchmarks['best_calls']['timing']['average']
            
            # Should scale reasonably with chain size
            expected_max_time = size * 0.0001  # 0.1ms per option
            assert best_calls_time < max(0.05, expected_max_time), \
                f"Options selection too slow: {best_calls_time:.4f}s for {size} options"
        
        self.results['options_selector'] = benchmark_results
        
        print("Options Selector Benchmark Summary:")
        for size in chain_sizes:
            best_calls_time = benchmark_results[size]['best_calls']['timing']['average']
            throughput = benchmark_results[size]['best_calls']['throughput']
            print(f"  {size} options: {best_calls_time:.4f}s avg, {throughput:.1f} ops/sec")
    
    @pytest.mark.performance
    def test_scoring_models_benchmarks(self, test_data_generator):
        """Benchmark scoring model performance."""
        print("\n=== Scoring Models Benchmarks ===")
        
        # Set up scorers
        technical_scorer = TechnicalScorer()
        momentum_scorer = MomentumScorer()
        weights = ScoreWeights(technical_weight=0.4, momentum_weight=0.4, options_weight=0.2)
        combined_scorer = CombinedScorer(weights)
        
        # Generate test data
        technical_data = {
            'rsi_14': 65.0,
            'macd': 1.5,
            'macd_signal': 1.2,
            'sma_20': 148.5,
            'sma_50': 145.2,
            'current_price': 150.0,
            'bollinger_upper': 155.0,
            'bollinger_lower': 142.0,
            'atr_14': 3.2,
            'volume_ratio': 1.3
        }
        
        momentum_data = {
            'price_momentum_5d': 3.2,
            'price_momentum_10d': 5.8,
            'price_momentum_20d': 8.5,
            'volume_momentum': 1.4,
            'trend_consistency': 0.75
        }
        
        combined_scores = {
            'technical_score': 75,
            'momentum_score': 82,
            'options_score': 68
        }
        
        # Benchmark technical scoring
        technical_benchmark = self.benchmark_runner.benchmark_function(
            "technical_scoring",
            technical_scorer.calculate_score,
            technical_data,
            iterations=1000
        )
        
        # Benchmark momentum scoring
        momentum_benchmark = self.benchmark_runner.benchmark_function(
            "momentum_scoring",
            momentum_scorer.calculate_score,
            momentum_data,
            iterations=1000
        )
        
        # Benchmark combined scoring
        combined_benchmark = self.benchmark_runner.benchmark_function(
            "combined_scoring",
            combined_scorer.calculate_combined_score,
            combined_scores,
            iterations=1000
        )
        
        # Performance assertions
        assert technical_benchmark['timing']['average'] < 0.001, "Technical scoring too slow"
        assert momentum_benchmark['timing']['average'] < 0.001, "Momentum scoring too slow"
        assert combined_benchmark['timing']['average'] < 0.0001, "Combined scoring too slow"
        
        self.results['scoring_models'] = {
            'technical': technical_benchmark,
            'momentum': momentum_benchmark,
            'combined': combined_benchmark
        }
        
        print("Scoring Models Benchmark Summary:")
        print(f"  Technical: {technical_benchmark['timing']['average']:.6f}s avg")
        print(f"  Momentum: {momentum_benchmark['timing']['average']:.6f}s avg")
        print(f"  Combined: {combined_benchmark['timing']['average']:.6f}s avg")
    
    @pytest.mark.performance
    def test_squeeze_detector_benchmarks(self, test_data_generator):
        """Benchmark squeeze detection algorithms."""
        print("\n=== Squeeze Detector Benchmarks ===")
        
        detector = SqueezeDetector()
        data_sizes = [60, 120, 252, 500]
        
        benchmark_results = {}
        
        for days in data_sizes:
            print(f"Benchmarking squeeze detection with {days} days...")
            
            # Generate consolidating price data
            test_data = test_data_generator.generate_historical_data("SQUEEZE", days=days)
            
            # Benchmark squeeze detection
            squeeze_benchmark = self.benchmark_runner.benchmark_function(
                f"squeeze_detection_{days}d",
                detector.detect_squeeze,
                test_data,
                iterations=20
            )
            
            # Benchmark momentum calculation
            momentum_benchmark = self.benchmark_runner.benchmark_function(
                f"squeeze_momentum_{days}d",
                detector.calculate_squeeze_momentum,
                test_data['close'],
                iterations=50
            )
            
            benchmark_results[days] = {
                'squeeze_detection': squeeze_benchmark,
                'momentum_calculation': momentum_benchmark
            }
        
        # Performance assertions
        for days, benchmarks in benchmark_results.items():
            squeeze_time = benchmarks['squeeze_detection']['timing']['average']
            
            # Should complete squeeze detection reasonably quickly
            expected_max_time = days * 0.0002  # 0.2ms per data point
            assert squeeze_time < max(0.5, expected_max_time), \
                f"Squeeze detection too slow: {squeeze_time:.4f}s for {days} days"
        
        self.results['squeeze_detector'] = benchmark_results
        
        print("Squeeze Detector Benchmark Summary:")
        for days in data_sizes:
            squeeze_time = benchmark_results[days]['squeeze_detection']['timing']['average']
            print(f"  {days} days: {squeeze_time:.4f}s avg")
    
    @pytest.mark.performance
    def test_greeks_calculator_benchmarks(self, test_data_generator):
        """Benchmark Greeks calculations."""
        print("\n=== Greeks Calculator Benchmarks ===")
        
        calculator = GreeksCalculator()
        
        # Test parameters
        option_params = {
            'spot_price': 100.0,
            'strike_price': 100.0,
            'time_to_expiration': 30/365,
            'risk_free_rate': 0.05,
            'implied_volatility': 0.25,
            'option_type': 'call'
        }
        
        # Benchmark individual Greeks
        delta_benchmark = self.benchmark_runner.benchmark_function(
            "delta_calculation",
            calculator.calculate_delta,
            **option_params,
            iterations=1000
        )
        
        gamma_benchmark = self.benchmark_runner.benchmark_function(
            "gamma_calculation",
            calculator.calculate_gamma,
            **option_params,
            iterations=1000
        )
        
        theta_benchmark = self.benchmark_runner.benchmark_function(
            "theta_calculation",
            calculator.calculate_theta,
            **option_params,
            iterations=1000
        )
        
        vega_benchmark = self.benchmark_runner.benchmark_function(
            "vega_calculation",
            calculator.calculate_vega,
            **option_params,
            iterations=1000
        )
        
        # Benchmark all Greeks together
        all_greeks_benchmark = self.benchmark_runner.benchmark_function(
            "all_greeks_calculation",
            calculator.calculate_all_greeks,
            **option_params,
            iterations=500
        )
        
        # Performance assertions
        assert delta_benchmark['timing']['average'] < 0.001, "Delta calculation too slow"
        assert gamma_benchmark['timing']['average'] < 0.001, "Gamma calculation too slow"
        assert theta_benchmark['timing']['average'] < 0.001, "Theta calculation too slow"
        assert vega_benchmark['timing']['average'] < 0.001, "Vega calculation too slow"
        assert all_greeks_benchmark['timing']['average'] < 0.005, "All Greeks calculation too slow"
        
        self.results['greeks_calculator'] = {
            'delta': delta_benchmark,
            'gamma': gamma_benchmark,
            'theta': theta_benchmark,
            'vega': vega_benchmark,
            'all_greeks': all_greeks_benchmark
        }
        
        print("Greeks Calculator Benchmark Summary:")
        print(f"  Delta: {delta_benchmark['timing']['average']:.6f}s avg")
        print(f"  Gamma: {gamma_benchmark['timing']['average']:.6f}s avg")
        print(f"  Theta: {theta_benchmark['timing']['average']:.6f}s avg")
        print(f"  Vega: {vega_benchmark['timing']['average']:.6f}s avg")
        print(f"  All Greeks: {all_greeks_benchmark['timing']['average']:.6f}s avg")
    
    @pytest.mark.performance
    def test_memory_usage_analytics(self, test_data_generator, memory_monitor):
        """Test memory usage of analytics components."""
        print("\n=== Analytics Memory Usage Test ===")
        
        initial_memory = memory_monitor()
        memory_results = {}
        
        # Test technical indicators memory usage
        large_data = test_data_generator.generate_historical_data("MEMORY", days=1000)
        
        pre_indicators_memory = memory_monitor()
        indicators = TechnicalIndicators.calculate_all_indicators(large_data)
        post_indicators_memory = memory_monitor()
        
        indicators_memory = post_indicators_memory - pre_indicators_memory
        memory_results['technical_indicators'] = indicators_memory
        
        # Test momentum analysis memory usage
        pre_momentum_memory = memory_monitor()
        analyzer = MomentumAnalyzer()
        momentum_result = analyzer.calculate_momentum_score(large_data)
        post_momentum_memory = memory_monitor()
        
        momentum_memory = post_momentum_memory - pre_momentum_memory
        memory_results['momentum_analysis'] = momentum_memory
        
        # Test options selector memory usage
        large_options_chain = []
        for i in range(500):  # Large options chain
            option = test_data_generator.generate_options_chain("MEM_TEST", 100 + i, 30)[0]
            large_options_chain.append(option)
        
        pre_options_memory = memory_monitor()
        criteria = CallOptionCriteria()
        selector = OptionsSelector(criteria)
        best_calls = selector.select_best_calls(large_options_chain, top_n=10)
        post_options_memory = memory_monitor()
        
        options_memory = post_options_memory - pre_options_memory
        memory_results['options_selector'] = options_memory
        
        # Clean up and measure final memory
        del large_data, indicators, momentum_result, large_options_chain, best_calls
        gc.collect()
        
        final_memory = memory_monitor()
        
        # Memory usage assertions
        for component, memory_usage in memory_results.items():
            assert memory_usage < 200, f"{component} uses too much memory: {memory_usage:.1f}MB"
        
        print("Analytics Memory Usage Summary:")
        for component, memory_usage in memory_results.items():
            print(f"  {component}: {memory_usage:.1f}MB")
        print(f"Total memory increase: {final_memory - initial_memory:.1f}MB")
    
    def _analyze_scaling_performance(self, benchmark_results: Dict, component_name: str):
        """Analyze scaling characteristics of benchmark results."""
        print(f"\n{component_name} Scaling Analysis:")
        
        data_sizes = sorted(benchmark_results.keys())
        
        # Analyze scaling for 'all' or main benchmark
        main_key = 'all_indicators' if 'all_indicators' in next(iter(benchmark_results.values())) else list(next(iter(benchmark_results.values())).keys())[0]
        
        scaling_data = []
        for size in data_sizes:
            timing = benchmark_results[size][main_key]['timing']['average']
            scaling_data.append((size, timing))
        
        # Calculate scaling factor
        if len(scaling_data) >= 2:
            small_size, small_time = scaling_data[0]
            large_size, large_time = scaling_data[-1]
            
            size_ratio = large_size / small_size
            time_ratio = large_time / small_time
            
            scaling_factor = time_ratio / size_ratio
            
            print(f"  Size ratio: {size_ratio:.1f}x")
            print(f"  Time ratio: {time_ratio:.1f}x") 
            print(f"  Scaling factor: {scaling_factor:.2f} (1.0 = linear)")
            
            if scaling_factor > 2.0:
                print(f"  WARNING: Poor scaling detected!")
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report_lines = [
            "Analytics Performance Benchmark Report",
            "=" * 50,
            ""
        ]
        
        # Summary of all components
        for component_name, component_results in self.results.items():
            report_lines.extend([
                f"{component_name.replace('_', ' ').title()}:",
                "-" * 30
            ])
            
            if isinstance(component_results, dict) and 'timing' in next(iter(component_results.values()), {}):
                # Simple benchmark results
                for test_name, result in component_results.items():
                    if isinstance(result, dict) and 'timing' in result:
                        avg_time = result['timing']['average']
                        throughput = result['throughput']
                        report_lines.append(f"  {test_name}: {avg_time:.6f}s avg, {throughput:.1f} ops/sec")
            else:
                # Nested results (by data size)
                for size, benchmarks in component_results.items():
                    if isinstance(benchmarks, dict):
                        report_lines.append(f"  Data size {size}:")
                        for test_name, result in benchmarks.items():
                            if isinstance(result, dict) and 'timing' in result:
                                avg_time = result['timing']['average']
                                throughput = result['throughput']
                                report_lines.append(f"    {test_name}: {avg_time:.6f}s, {throughput:.1f} ops/sec")
            
            report_lines.append("")
        
        return "\n".join(report_lines)