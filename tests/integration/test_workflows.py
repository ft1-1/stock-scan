"""Integration tests for full workflow scenarios."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

# Import all modules for integration testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from screener.workflow_engine import WorkflowEngine, StepResult, WorkflowState
from models.workflow_models import WorkflowConfig, WorkflowExecutionContext
from models.base_models import ScreeningCriteria, AnalysisResult
from providers.provider_manager import ProviderManager
from analytics.technical_indicators import TechnicalIndicators
from analytics.momentum_analysis import MomentumAnalyzer
from analytics.options_selector import OptionsSelector, CallOptionCriteria
from ai_analysis.rating_engine import RatingEngine
from config.settings import Settings


class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""
    
    @pytest.fixture
    def test_settings(self):
        """Test settings configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return Settings(
                environment="testing",
                debug=True,
                eodhd_api_key="test_eodhd_key",
                marketdata_api_key="test_marketdata_key",
                claude_api_key="test_claude_key",
                output_directory=f"{temp_dir}/output",
                log_directory=f"{temp_dir}/logs",
                cache_directory=f"{temp_dir}/cache",
                max_stocks_to_screen=5,
                enable_caching=False,
                enable_ai_analysis=False,  # Disable for faster tests
                max_concurrent_stocks=3
            )
    
    @pytest.fixture
    def workflow_config(self):
        """Workflow configuration for testing."""
        return WorkflowConfig(
            max_concurrent_stocks=3,
            max_retry_attempts=2,
            step_timeout_seconds=30,
            continue_on_errors=True,
            error_threshold_percent=50.0,
            enable_ai_analysis=False,
            save_intermediate_results=True,
            min_data_quality_score=60.0
        )
    
    @pytest.fixture
    def screening_criteria(self):
        """Test screening criteria."""
        return ScreeningCriteria(
            min_market_cap=1_000_000_000,
            max_market_cap=50_000_000_000,
            min_price=20.0,
            max_price=300.0,
            min_volume=1_000_000,
            min_rsi=35.0,
            max_rsi=65.0,
            min_option_volume=50,
            min_open_interest=100,
            max_days_to_expiration=45,
            exclude_sectors=[],
            exclude_symbols=[]
        )
    
    @pytest.fixture
    def integrated_workflow(self, workflow_config, test_data_generator):
        """Create integrated workflow with all components."""
        engine = WorkflowEngine(workflow_config)
        
        # Register all workflow steps
        
        # Step 1: Screen stocks
        async def screen_stocks_step(context, symbols):
            """Mock stock screening step."""
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Filter symbols based on criteria (mock implementation)
            filtered_symbols = symbols[:3]  # Limit for testing
            
            return StepResult(
                step_id="screen_stocks",
                success=True,
                results={
                    "filtered_symbols": filtered_symbols,
                    "total_screened": len(symbols),
                    "passed_screening": len(filtered_symbols)
                },
                execution_time_seconds=0.1
            )
        
        # Step 2: Fetch quotes
        async def fetch_quotes_step(context, symbols):
            """Mock quote fetching step."""
            await asyncio.sleep(0.1)
            
            quotes = []
            for symbol in symbols:
                quote_data = test_data_generator.generate_stock_quote(symbol)
                quotes.append(quote_data)
            
            return StepResult(
                step_id="fetch_quotes",
                success=True,
                results={"quotes": quotes},
                execution_time_seconds=0.1,
                api_calls_made=len(symbols),
                api_cost=len(symbols) * 0.01
            )
        
        # Step 3: Calculate technical indicators
        async def technical_analysis_step(context, symbols):
            """Mock technical analysis step."""
            await asyncio.sleep(0.15)
            
            technical_results = []
            for symbol in symbols:
                # Generate realistic historical data
                hist_data = test_data_generator.generate_historical_data(symbol, days=50)
                
                # Calculate indicators
                indicators = TechnicalIndicators.calculate_all_indicators(hist_data)
                indicators['symbol'] = symbol
                
                technical_results.append(indicators)
            
            return StepResult(
                step_id="technical_analysis",
                success=True,
                results={"technical_indicators": technical_results},
                execution_time_seconds=0.15
            )
        
        # Step 4: Momentum analysis
        async def momentum_analysis_step(context, symbols):
            """Mock momentum analysis step."""
            await asyncio.sleep(0.1)
            
            momentum_analyzer = MomentumAnalyzer()
            momentum_results = []
            
            for symbol in symbols:
                # Generate price data for momentum analysis
                price_data = test_data_generator.generate_historical_data(symbol, days=30)
                momentum_score = momentum_analyzer.calculate_momentum_score(price_data)
                momentum_score['symbol'] = symbol
                
                momentum_results.append(momentum_score)
            
            return StepResult(
                step_id="momentum_analysis",
                success=True,
                results={"momentum_analysis": momentum_results},
                execution_time_seconds=0.1
            )
        
        # Step 5: Options analysis
        async def options_analysis_step(context, symbols):
            """Mock options analysis step."""
            await asyncio.sleep(0.2)
            
            options_results = []
            criteria = CallOptionCriteria(
                min_volume=50,
                min_open_interest=100,
                max_days_to_expiration=45,
                min_delta=0.2,
                max_delta=0.8
            )
            selector = OptionsSelector(criteria)
            
            for symbol in symbols:
                # Generate options chain
                options_chain = test_data_generator.generate_options_chain(
                    symbol, 150.0, days_to_exp=30
                )
                
                # Select best calls
                best_calls = selector.select_best_calls(options_chain, top_n=3)
                
                options_results.append({
                    'symbol': symbol,
                    'best_calls': best_calls,
                    'total_options': len(options_chain)
                })
            
            return StepResult(
                step_id="options_analysis",
                success=True,
                results={"options_analysis": options_results},
                execution_time_seconds=0.2,
                api_calls_made=len(symbols),
                api_cost=len(symbols) * 0.02
            )
        
        # Step 6: Generate ratings
        async def generate_ratings_step(context, symbols):
            """Mock rating generation step."""
            await asyncio.sleep(0.1)
            
            ratings = []
            for symbol in symbols:
                # Combine all analysis results
                technical_data = next(
                    (t for t in context.step_results["technical_analysis"].results["technical_indicators"] 
                     if t["symbol"] == symbol), 
                    {}
                )
                momentum_data = next(
                    (m for m in context.step_results["momentum_analysis"].results["momentum_analysis"]
                     if m["symbol"] == symbol),
                    {}
                )
                options_data = next(
                    (o for o in context.step_results["options_analysis"].results["options_analysis"]
                     if o["symbol"] == symbol),
                    {}
                )
                
                # Calculate combined rating
                technical_score = technical_data.get("rsi_14", 50) + 30  # Mock scoring
                momentum_score = momentum_data.get("momentum_score", 50)
                options_score = len(options_data.get("best_calls", [])) * 20 + 20  # Mock scoring
                
                overall_rating = (technical_score + momentum_score + options_score) / 3
                overall_rating = max(0, min(100, overall_rating))  # Clamp to 0-100
                
                rating = AnalysisResult(
                    symbol=symbol,
                    analysis_id=f"test_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    technical_score=technical_score,
                    momentum_score=momentum_score,
                    options_score=options_score,
                    overall_rating=overall_rating,
                    confidence=85.0,
                    recommendation="HOLD",  # Mock recommendation
                    reasoning=f"Mock analysis for {symbol}",
                    key_factors=["Technical strength", "Momentum present"],
                    risks=["Market volatility"],
                    data_quality_score=88.0
                )
                
                ratings.append(rating)
            
            return StepResult(
                step_id="generate_ratings",
                success=True,
                results={"ratings": ratings},
                execution_time_seconds=0.1
            )
        
        # Register all steps
        engine.register_step("screen_stocks", screen_stocks_step, order=1)
        engine.register_step("fetch_quotes", fetch_quotes_step, order=2)
        engine.register_step("technical_analysis", technical_analysis_step, order=3)
        engine.register_step("momentum_analysis", momentum_analysis_step, order=4)
        engine.register_step("options_analysis", options_analysis_step, order=5)
        engine.register_step("generate_ratings", generate_ratings_step, order=6)
        
        return engine
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow_execution(self, integrated_workflow, test_data_generator):
        """Test complete workflow from start to finish."""
        # Input symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        # Execute workflow
        start_time = datetime.now()
        context = await integrated_workflow.execute_workflow(symbols)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Verify workflow completion
        assert context.state == WorkflowState.COMPLETED
        assert len(context.step_results) == 6  # All steps completed
        
        # Verify each step results
        assert "screen_stocks" in context.step_results
        assert "fetch_quotes" in context.step_results
        assert "technical_analysis" in context.step_results
        assert "momentum_analysis" in context.step_results
        assert "options_analysis" in context.step_results
        assert "generate_ratings" in context.step_results
        
        # All steps should have succeeded
        for step_id, result in context.step_results.items():
            assert result.success, f"Step {step_id} failed"
        
        # Verify final results
        final_ratings = context.step_results["generate_ratings"].results["ratings"]
        assert len(final_ratings) > 0
        
        for rating in final_ratings:
            assert isinstance(rating, AnalysisResult)
            assert 0 <= rating.overall_rating <= 100
            assert 0 <= rating.confidence <= 100
            assert rating.symbol in symbols
        
        # Verify execution time is reasonable
        assert execution_time < 30  # Should complete within 30 seconds
        
        print(f"Workflow completed in {execution_time:.2f} seconds")
        print(f"Processed {len(final_ratings)} symbols")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_with_failures(self, integrated_workflow):
        """Test workflow behavior with step failures."""
        # Add a failing step
        async def failing_step(context, symbols):
            raise Exception("Simulated failure in integration test")
        
        integrated_workflow.register_step("failing_step", failing_step, order=7)
        
        symbols = ["AAPL", "MSFT"]
        context = await integrated_workflow.execute_workflow(symbols)
        
        # Workflow should complete with errors (continue_on_errors=True)
        assert context.state == WorkflowState.COMPLETED_WITH_ERRORS
        assert len(context.errors) > 0
        
        # Previous steps should still have succeeded
        assert context.step_results["generate_ratings"].success
        
        # Failing step should be recorded
        assert "failing_step" in context.step_results
        assert not context.step_results["failing_step"].success
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_checkpointing(self, integrated_workflow):
        """Test workflow checkpointing and recovery."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Execute workflow with checkpointing enabled
        context = await integrated_workflow.execute_workflow(symbols)
        
        # Verify checkpoints were created
        assert len(context.checkpoints) > 0
        
        # Test checkpoint recovery
        latest_checkpoint = context.checkpoints[-1]
        recovered_context = await integrated_workflow.recover_from_checkpoint(
            latest_checkpoint.checkpoint_id
        )
        
        assert recovered_context is not None
        assert recovered_context.workflow_id == context.workflow_id
        assert recovered_context.symbols == symbols
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_symbol_processing(self, integrated_workflow):
        """Test concurrent processing of multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]
        
        start_time = datetime.now()
        context = await integrated_workflow.execute_workflow(symbols)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        assert context.state == WorkflowState.COMPLETED
        
        # With concurrency, execution time should be reasonable even with more symbols
        # Rough estimate: without concurrency, 7 symbols * 6 steps * 0.1s = 4.2s minimum
        # With concurrency (max 3 concurrent), should be faster
        assert execution_time < 10  # Should be well under sequential time
        
        # All symbols should be processed
        final_ratings = context.step_results["generate_ratings"].results["ratings"]
        processed_symbols = {r.symbol for r in final_ratings}
        
        # At least some symbols should be processed (depending on screening)
        assert len(processed_symbols) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_metrics_collection(self, integrated_workflow):
        """Test comprehensive metrics collection during workflow."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        context = await integrated_workflow.execute_workflow(symbols)
        metrics = await integrated_workflow.get_workflow_metrics(context.workflow_id)
        
        assert metrics is not None
        assert metrics.workflow_id == context.workflow_id
        assert metrics.symbols_processed > 0
        assert metrics.steps_completed == 6
        assert metrics.steps_failed == 0
        assert metrics.total_execution_time_seconds > 0
        assert metrics.total_api_calls > 0
        assert metrics.total_api_cost > 0
        
        # Print metrics for debugging
        print(f"Workflow Metrics:")
        print(f"  Execution Time: {metrics.total_execution_time_seconds:.2f}s")
        print(f"  Symbols Processed: {metrics.symbols_processed}")
        print(f"  API Calls: {metrics.total_api_calls}")
        print(f"  API Cost: ${metrics.total_api_cost:.4f}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_quality_validation(self, integrated_workflow):
        """Test data quality validation throughout workflow."""
        symbols = ["AAPL", "MSFT"]
        
        context = await integrated_workflow.execute_workflow(symbols)
        
        # Check data quality at each step
        for step_id, result in context.step_results.items():
            assert result.success, f"Step {step_id} failed, affecting data quality"
            
            # Verify result structure
            assert "results" in result.__dict__
            assert isinstance(result.results, dict)
            
            # Check for required data fields based on step
            if step_id == "fetch_quotes":
                assert "quotes" in result.results
                quotes = result.results["quotes"]
                for quote in quotes:
                    assert "symbol" in quote
                    assert "last_price" in quote
                    assert quote["last_price"] > 0
            
            elif step_id == "technical_analysis":
                assert "technical_indicators" in result.results
                indicators = result.results["technical_indicators"]
                for indicator_set in indicators:
                    assert "symbol" in indicator_set
                    assert "rsi_14" in indicator_set
                    assert 0 <= indicator_set["rsi_14"] <= 100
            
            elif step_id == "generate_ratings":
                assert "ratings" in result.results
                ratings = result.results["ratings"]
                for rating in ratings:
                    assert isinstance(rating, AnalysisResult)
                    assert 0 <= rating.overall_rating <= 100
                    assert rating.data_quality_score >= 60  # Min threshold
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_and_resilience(self, integrated_workflow):
        """Test workflow resilience and error recovery."""
        # Create a step that fails intermittently
        failure_count = 0
        
        async def flaky_step(context, symbols):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:  # Fail first 2 attempts
                raise Exception(f"Intermittent failure #{failure_count}")
            
            return StepResult(
                step_id="flaky_step",
                success=True,
                results={"recovery_attempt": failure_count},
                execution_time_seconds=0.1
            )
        
        # Insert flaky step
        integrated_workflow.register_step("flaky_step", flaky_step, order=7)
        
        symbols = ["AAPL"]
        context = await integrated_workflow.execute_workflow(symbols)
        
        # Should recover after retries
        assert context.state == WorkflowState.COMPLETED
        assert "flaky_step" in context.step_results
        assert context.step_results["flaky_step"].success
        assert failure_count == 3  # 1 initial + 2 retries
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_large_batch_processing(self, integrated_workflow, test_data_generator):
        """Test processing of large batch of symbols."""
        # Generate larger symbol list
        symbols = test_data_generator.generate_stock_symbols(20)
        
        start_time = datetime.now()
        context = await integrated_workflow.execute_workflow(symbols)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        assert context.state in [WorkflowState.COMPLETED, WorkflowState.COMPLETED_WITH_ERRORS]
        
        # Should complete in reasonable time even with larger batch
        assert execution_time < 60  # 1 minute max for 20 symbols
        
        # Check that processing scales appropriately
        final_ratings = context.step_results["generate_ratings"].results["ratings"]
        print(f"Processed {len(final_ratings)} out of {len(symbols)} symbols in {execution_time:.2f}s")
        
        # Should process at least some symbols
        assert len(final_ratings) > 0


class TestWorkflowEdgeCases:
    """Test edge cases and boundary conditions in workflows."""
    
    @pytest.fixture
    def minimal_workflow(self):
        """Create minimal workflow for edge case testing."""
        config = WorkflowConfig(
            max_concurrent_stocks=1,
            max_retry_attempts=1,
            step_timeout_seconds=5,
            continue_on_errors=False,
            error_threshold_percent=0.0
        )
        
        engine = WorkflowEngine(config)
        
        async def simple_step(context, symbols):
            return StepResult(
                step_id="simple_step",
                success=True,
                results={"processed": symbols},
                execution_time_seconds=0.1
            )
        
        engine.register_step("simple_step", simple_step, order=1)
        return engine
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_symbol_list(self, minimal_workflow):
        """Test workflow with empty symbol list."""
        context = await minimal_workflow.execute_workflow([])
        
        # Should complete successfully with empty results
        assert context.state == WorkflowState.COMPLETED
        assert context.step_results["simple_step"].success
        assert context.step_results["simple_step"].results["processed"] == []
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_single_symbol_processing(self, minimal_workflow):
        """Test workflow with single symbol."""
        context = await minimal_workflow.execute_workflow(["AAPL"])
        
        assert context.state == WorkflowState.COMPLETED
        assert context.step_results["simple_step"].results["processed"] == ["AAPL"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_cancellation(self, minimal_workflow):
        """Test workflow cancellation."""
        # Create a long-running step
        async def long_step(context, symbols):
            await asyncio.sleep(10)  # Long delay
            return StepResult(
                step_id="long_step",
                success=True,
                results={},
                execution_time_seconds=10.0
            )
        
        minimal_workflow.register_step("long_step", long_step, order=2)
        
        # Start workflow
        task = asyncio.create_task(minimal_workflow.execute_workflow(["AAPL"]))
        
        # Cancel after short delay
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Should handle cancellation gracefully
        with pytest.raises(asyncio.CancelledError):
            await task
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_invalid_symbol_handling(self, minimal_workflow):
        """Test handling of invalid stock symbols."""
        invalid_symbols = ["", "INVALID_SYMBOL_123", "!@#$", None]
        
        # Filter out None values that would cause immediate failure
        test_symbols = [s for s in invalid_symbols if s is not None]
        
        context = await minimal_workflow.execute_workflow(test_symbols)
        
        # Should complete without crashing
        assert context.state in [WorkflowState.COMPLETED, WorkflowState.COMPLETED_WITH_ERRORS]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_usage_monitoring(self, integrated_workflow):
        """Test memory usage during workflow execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        context = await integrated_workflow.execute_workflow(symbols)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_state_consistency(self, integrated_workflow):
        """Test workflow state consistency throughout execution."""
        symbols = ["AAPL", "MSFT"]
        
        # Track state changes
        state_changes = []
        
        # Monkey patch to capture state changes
        original_set_state = integrated_workflow._set_workflow_state
        
        def track_state_change(context, new_state):
            state_changes.append((datetime.now(), new_state))
            return original_set_state(context, new_state)
        
        integrated_workflow._set_workflow_state = track_state_change
        
        context = await integrated_workflow.execute_workflow(symbols)
        
        # Verify state progression
        assert len(state_changes) > 0
        
        # Should start with RUNNING and end with COMPLETED
        assert state_changes[0][1] == WorkflowState.RUNNING
        assert state_changes[-1][1] == WorkflowState.COMPLETED
        
        # States should progress logically
        valid_transitions = {
            WorkflowState.PENDING: [WorkflowState.RUNNING],
            WorkflowState.RUNNING: [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.COMPLETED_WITH_ERRORS],
            WorkflowState.COMPLETED: [],
            WorkflowState.FAILED: [],
            WorkflowState.COMPLETED_WITH_ERRORS: []
        }
        
        for i in range(1, len(state_changes)):
            prev_state = state_changes[i-1][1]
            current_state = state_changes[i][1]
            
            assert current_state in valid_transitions.get(prev_state, [current_state]), \
                f"Invalid state transition: {prev_state} -> {current_state}"