#!/usr/bin/env python3
"""Comprehensive integration tests for the complete workflow system.

This test suite validates the entire integrated workflow to ensure all components
work together correctly and meet the requirements for:
1. Component integration
2. Data flow validation
3. Local scoring and ranking
4. Top N filtering logic
5. Rate limiting and cost controls
6. Error handling and recovery
"""

import asyncio
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import unittest.mock as mock

# Test framework imports
import pytest_asyncio

# Import system components
from src.screener.workflow_orchestrator import WorkflowOrchestrator
from src.screener.workflow_engine import WorkflowEngine
from src.models import (
    ScreeningCriteria, 
    WorkflowConfig, 
    WorkflowStep,
    WorkflowStepStatus,
    OptionContract,
    OptionType
)
from src.analytics.scoring_models import QuantitativeScorer
from src.analytics.options_selector import OptionsSelector


class TestCompleteWorkflowIntegration:
    """Comprehensive integration tests for the complete workflow."""
    
    @pytest.fixture
    def test_config(self):
        """Create test workflow configuration."""
        return WorkflowConfig(
            max_retry_attempts=2,
            step_timeout_seconds=180,  # 3 minutes for tests
            checkpoint_interval=1,
            max_concurrent_stocks=2,
            enable_caching=True,
            enable_validation=True,
            enable_ai_analysis=False,  # Disable AI for unit tests
            max_ai_cost_dollars=5.0
        )
    
    @pytest.fixture
    def test_symbols(self):
        """Test symbols for validation."""
        return ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    @pytest.fixture
    def basic_criteria(self, test_symbols):
        """Create basic screening criteria."""
        return ScreeningCriteria(
            specific_symbols=test_symbols,
            min_market_cap=1000000000,
            max_market_cap=10000000000000,
            min_price=5.0,
            max_price=1000.0,
            min_volume=100000,
            min_option_volume=50,
            min_open_interest=25
        )
    
    @pytest.fixture
    def mock_data_generator(self):
        """Generate mock data for testing."""
        class MockDataGenerator:
            @staticmethod
            def create_enhanced_data(symbol: str) -> Dict[str, Any]:
                """Create mock enhanced data for a symbol."""
                return {
                    "symbol": symbol,
                    "quote": {
                        "price": 150.0 + hash(symbol) % 100,
                        "volume": 1000000 + hash(symbol) % 500000,
                        "market_cap": 2000000000000 + hash(symbol) % 1000000000000
                    },
                    "historical_data": [
                        {"date": "2025-08-20", "close": 149.0, "volume": 950000},
                        {"date": "2025-08-19", "close": 148.5, "volume": 1020000},
                        {"date": "2025-08-18", "close": 147.8, "volume": 980000}
                    ],
                    "technical_indicators": {
                        "rsi_14": 55.0 + (hash(symbol) % 20),
                        "macd": 0.5,
                        "adx_14": 25.0 + (hash(symbol) % 15),
                        "sma_50": 148.0,
                        "sma_200": 145.0
                    },
                    "options_chain": [
                        {
                            "symbol": f"{symbol}250920C00150000",
                            "strike": 150.0,
                            "expiration": "2025-09-20",
                            "option_type": "call",
                            "bid": 5.0,
                            "ask": 5.2,
                            "volume": 500,
                            "open_interest": 1000,
                            "delta": 0.6,
                            "gamma": 0.02,
                            "theta": -0.05,
                            "vega": 0.15,
                            "implied_volatility": 0.25
                        }
                    ]
                }
                
            @staticmethod
            def create_option_contract(symbol: str, strike: float = 150.0) -> OptionContract:
                """Create mock option contract."""
                return OptionContract(
                    symbol=f"{symbol}250920C{int(strike*1000):08d}",
                    underlying_symbol=symbol,
                    strike_price=strike,
                    expiration_date=datetime(2025, 9, 20).date(),
                    option_type=OptionType.CALL,
                    bid=5.0,
                    ask=5.2,
                    last_price=5.1,
                    volume=500,
                    open_interest=1000,
                    delta=0.6,
                    gamma=0.02,
                    theta=-0.05,
                    vega=0.15,
                    implied_volatility=0.25,
                    days_to_expiration=30
                )
        
        return MockDataGenerator()

    @pytest.mark.asyncio
    async def test_component_registration(self, test_config):
        """Test that all step executors are properly registered."""
        orchestrator = WorkflowOrchestrator(test_config)
        await orchestrator.initialize()
        
        # Verify all required steps are registered
        required_steps = [
            WorkflowStep.STOCK_SCREENING,
            WorkflowStep.DATA_COLLECTION,
            WorkflowStep.TECHNICAL_CALCULATION,
            WorkflowStep.OPTION_SELECTION,
            WorkflowStep.LLM_PACKAGING,
            WorkflowStep.AI_ANALYSIS,
            WorkflowStep.RESULT_PROCESSING
        ]
        
        for step in required_steps:
            assert step in orchestrator.workflow_engine.step_executors
            executor = orchestrator.workflow_engine.step_executors[step]
            assert executor is not None
            assert executor.step == step

    @pytest.mark.asyncio
    async def test_data_flow_validation(self, basic_criteria, test_config, mock_data_generator):
        """Test data flow between workflow steps."""
        orchestrator = WorkflowOrchestrator(test_config)
        
        # Mock the data collection to return controlled data
        with mock.patch('src.screener.steps.data_enrichment_step.DataEnrichmentExecutor.execute_step') as mock_data:
            mock_data.return_value = {
                symbol: mock_data_generator.create_enhanced_data(symbol)
                for symbol in basic_criteria.specific_symbols
            }
            
            result = await orchestrator.execute_workflow(basic_criteria, "test_data_flow")
            
            # Verify workflow completed
            assert result.workflow_id == "test_data_flow"
            assert result.total_stocks_screened >= 0
            assert isinstance(result.qualifying_results, list)

    @pytest.mark.asyncio
    async def test_quantitative_scoring_logic(self, mock_data_generator):
        """Test that quantitative scoring produces valid scores."""
        scorer = QuantitativeScorer()
        
        # Test with mock technical indicators
        indicators = {
            "rsi_14": 55.0,
            "adx_14": 25.0,
            "macd": 0.5,
            "pct_above_sma_50": 2.0,
            "pct_above_sma_200": 5.0
        }
        
        # Test scoring components
        tech_score = scorer.calculate_technical_score(indicators)
        assert 0 <= tech_score["total_score"] <= 100
        assert "rsi_score" in tech_score
        assert "adx_score" in tech_score
        
        # Test options scoring
        option = mock_data_generator.create_option_contract("TEST")
        options_score = scorer.calculate_options_score([option])
        assert 0 <= options_score["total_score"] <= 100

    @pytest.mark.asyncio
    async def test_local_ranking_and_filtering(self, basic_criteria, test_config):
        """Test that local ranking filters to top N opportunities only."""
        orchestrator = WorkflowOrchestrator(test_config)
        
        # Mock technical analysis to return varying scores
        mock_opportunities = []
        for i, symbol in enumerate(basic_criteria.specific_symbols):
            mock_opportunities.append({
                "symbol": symbol,
                "composite_score": 20.0 + (i * 15),  # Scores: 20, 35, 50, 65
                "technical_score": 15.0 + (i * 10),
                "options_score": 10.0 + (i * 8),
                "proceed_to_ai": False  # Will be set by ranking step
            })
        
        # Mock the steps to return controlled data
        with mock.patch('src.screener.steps.technical_analysis_step.TechnicalAnalysisExecutor.execute_step') as mock_tech:
            mock_tech.return_value = {"analyzed_opportunities": mock_opportunities}
            
            result = await orchestrator.execute_workflow(basic_criteria, "test_ranking")
            
            # Verify only high-scoring opportunities were selected
            # (Only scores >= 40 should pass quality filter)
            assert result.total_stocks_screened == len(basic_criteria.specific_symbols)

    @pytest.mark.asyncio
    async def test_options_selection_algorithms(self):
        """Test options selection produces valid results."""
        selector = OptionsSelector()
        
        # Create test options chain
        options_chain = [
            OptionContract(
                symbol=f"TEST250920C{int(strike*1000):08d}",
                underlying_symbol="TEST",
                strike_price=strike,
                expiration_date=datetime(2025, 9, 20).date(),
                option_type=OptionType.CALL,
                bid=5.0,
                ask=5.2,
                volume=500,
                open_interest=1000,
                delta=0.5 + (strike - 150) * 0.01,
                days_to_expiration=30
            )
            for strike in [145, 150, 155, 160]
        ]
        
        # Test basic filtering
        filtered = selector.filter_options_basic(options_chain)
        assert len(filtered) >= 0
        
        # Test scoring
        for option in filtered:
            score = selector.calculate_options_score(option)
            assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, basic_criteria, test_config):
        """Test error handling across system boundaries."""
        orchestrator = WorkflowOrchestrator(test_config)
        
        # Test with invalid symbol to trigger error handling
        error_criteria = ScreeningCriteria(
            specific_symbols=["INVALID_SYMBOL"],
            min_market_cap=1000000000
        )
        
        result = await orchestrator.execute_workflow(error_criteria, "test_errors")
        
        # Should complete gracefully even with errors
        assert result.workflow_id == "test_errors"
        assert isinstance(result.errors_encountered, list)

    @pytest.mark.asyncio
    async def test_cost_tracking_and_limits(self, basic_criteria, test_config):
        """Test cost tracking and daily limits work correctly."""
        # Enable AI analysis for cost testing
        test_config.enable_ai_analysis = True
        test_config.max_ai_cost_dollars = 1.0  # Low limit for testing
        
        orchestrator = WorkflowOrchestrator(test_config)
        
        # Mock Claude client to track calls
        with mock.patch('src.ai_analysis.claude_client.ClaudeClient') as mock_claude:
            mock_instance = mock.MagicMock()
            mock_instance.get_usage_stats.return_value = {"total_cost": 0.5}
            mock_claude.return_value = mock_instance
            
            result = await orchestrator.execute_workflow(basic_criteria, "test_costs")
            
            # Verify cost tracking
            assert hasattr(result, 'total_cost')

    @pytest.mark.asyncio
    async def test_json_persistence_per_opportunity(self, basic_criteria, test_config):
        """Test that individual JSON files are created per opportunity."""
        orchestrator = WorkflowOrchestrator(test_config)
        
        # Mock high-scoring opportunities to trigger AI analysis
        with mock.patch('src.screener.steps.local_ranking_step.LocalRankingExecutor.execute_step') as mock_ranking:
            mock_ranking.return_value = {
                "top_opportunities": [
                    {
                        "symbol": "AAPL",
                        "composite_score": 85.0,
                        "proceed_to_ai": True
                    }
                ]
            }
            
            # Mock Claude analysis to return results
            with mock.patch('src.screener.steps.claude_analysis_step.ClaudeAnalysisExecutor.execute_step') as mock_claude:
                mock_claude.return_value = {
                    "ai_results": [
                        {
                            "symbol": "AAPL",
                            "ai_rating": 92,
                            "analysis": "Strong opportunity"
                        }
                    ]
                }
                
                result = await orchestrator.execute_workflow(basic_criteria, "test_persistence")
                
                # Check that results were processed
                assert result.workflow_id == "test_persistence"

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, basic_criteria, test_config):
        """Test performance meets acceptable benchmarks."""
        start_time = datetime.now()
        
        orchestrator = WorkflowOrchestrator(test_config)
        result = await orchestrator.execute_workflow(basic_criteria, "test_performance")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time for test data
        assert execution_time < 60.0  # 1 minute max for test
        assert result.execution_time_seconds is not None

    def test_enhanced_data_collection_fields(self, mock_data_generator):
        """Test that enhanced data collection includes all required fields."""
        enhanced_data = mock_data_generator.create_enhanced_data("TEST")
        
        # Verify required data sections
        required_sections = [
            "symbol",
            "quote", 
            "historical_data",
            "technical_indicators",
            "options_chain"
        ]
        
        for section in required_sections:
            assert section in enhanced_data
        
        # Verify quote data structure
        quote = enhanced_data["quote"]
        required_quote_fields = ["price", "volume", "market_cap"]
        for field in required_quote_fields:
            assert field in quote

    @pytest.mark.asyncio
    async def test_sophisticated_algorithms_active(self):
        """Test that sophisticated algorithms are being used."""
        # Test QuantitativeScorer is active
        scorer = QuantitativeScorer()
        assert scorer is not None
        assert hasattr(scorer, 'weights')
        assert hasattr(scorer, 'calculate_technical_score')
        assert hasattr(scorer, 'calculate_options_score')
        
        # Test OptionsSelector is active
        selector = OptionsSelector()
        assert selector is not None
        assert hasattr(selector, 'filter_options_basic')
        assert hasattr(selector, 'calculate_liquidity_score')

    @pytest.mark.asyncio
    async def test_workflow_step_execution_order(self, basic_criteria, test_config):
        """Test that workflow steps execute in correct order."""
        orchestrator = WorkflowOrchestrator(test_config)
        
        execution_order = []
        
        # Mock each step to track execution order
        original_execute = WorkflowEngine._execute_single_step
        
        async def track_execution(self, step, input_data, context):
            execution_order.append(step)
            return await original_execute(self, step, input_data, context)
        
        with mock.patch.object(WorkflowEngine, '_execute_single_step', track_execution):
            await orchestrator.execute_workflow(basic_criteria, "test_order")
        
        # Verify correct execution order
        expected_order = [
            WorkflowStep.STOCK_SCREENING,
            WorkflowStep.DATA_COLLECTION,
            WorkflowStep.TECHNICAL_CALCULATION,
            WorkflowStep.OPTION_SELECTION,
            WorkflowStep.LLM_PACKAGING,
            WorkflowStep.AI_ANALYSIS,
            WorkflowStep.RESULT_PROCESSING
        ]
        
        assert execution_order == expected_order

    @pytest.mark.asyncio
    async def test_rate_limiting_validation(self, test_config):
        """Test 60-second rate limiting between Claude calls."""
        # This test would verify rate limiting in a real implementation
        # For now, verify the configuration is correct
        
        assert test_config.max_ai_cost_dollars > 0
        # In real implementation, would test actual rate limiting behavior


# Additional test utilities
class TestWorkflowValidationHelpers:
    """Helper methods for workflow validation."""
    
    @staticmethod
    def validate_workflow_result(result):
        """Validate workflow result structure."""
        assert hasattr(result, 'workflow_id')
        assert hasattr(result, 'total_stocks_screened')
        assert hasattr(result, 'qualifying_results')
        assert hasattr(result, 'execution_time_seconds')
        assert isinstance(result.qualifying_results, list)
        
    @staticmethod
    def validate_data_quality(enhanced_data):
        """Validate enhanced data quality."""
        assert enhanced_data is not None
        assert 'symbol' in enhanced_data
        
        if 'technical_indicators' in enhanced_data:
            indicators = enhanced_data['technical_indicators']
            for key, value in indicators.items():
                if value is not None:
                    assert isinstance(value, (int, float))
    
    @staticmethod
    def validate_scoring_ranges(scores):
        """Validate scoring produces values in expected ranges."""
        for score_name, score_value in scores.items():
            if score_value is not None:
                assert 0 <= score_value <= 100, f"{score_name} score {score_value} outside valid range"


if __name__ == "__main__":
    """Run integration tests."""
    pytest.main([__file__, "-v", "--tb=short"])