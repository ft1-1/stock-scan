"""Comprehensive unit tests for core modules."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import json

# Import core modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.base_models import (
    StockQuote, OptionContract, TechnicalIndicators, 
    ScreeningCriteria, WorkflowConfig, AnalysisResult
)
from models.provider_models import (
    ProviderConfig, CacheConfig, ProviderHealthStatus, ProviderUsageStats
)
from models.workflow_models import (
    WorkflowStep, WorkflowExecutionContext, WorkflowCheckpoint, WorkflowMetrics
)
from screener.workflow_engine import WorkflowEngine, StepResult, WorkflowState
from config.settings import Settings


class TestBaseModels:
    """Tests for core data models."""
    
    def test_stock_quote_model(self):
        """Test StockQuote model validation and functionality."""
        quote = StockQuote(
            symbol="AAPL",
            last_price=150.50,
            change=2.25,
            change_percent=1.52,
            volume=50_000_000,
            bid=150.45,
            ask=150.55,
            high=152.00,
            low=149.00,
            open=150.00
        )
        
        assert quote.symbol == "AAPL"
        assert quote.last_price == 150.50
        assert quote.volume == 50_000_000
        
        # Test price validation
        with pytest.raises(ValueError):
            StockQuote(
                symbol="INVALID",
                last_price=-150.0,  # Negative price should fail
                volume=1000000
            )
    
    def test_option_contract_model(self):
        """Test OptionContract model validation."""
        option = OptionContract(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            strike=Decimal("150.00"),
            expiration=date.today() + timedelta(days=30),
            option_type="call",
            bid=Decimal("5.20"),
            ask=Decimal("5.30"),
            last=Decimal("5.25"),
            volume=1500,
            open_interest=8500,
            delta=0.55,
            gamma=0.025,
            theta=-0.08,
            vega=0.35,
            implied_volatility=0.28,
            days_to_expiration=30
        )
        
        assert option.underlying_symbol == "AAPL"
        assert option.option_type in ["call", "put"]
        assert option.ask >= option.bid
        assert 0 <= option.delta <= 1 if option.option_type == "call" else -1 <= option.delta <= 0
        
        # Test Greeks validation
        with pytest.raises(ValueError):
            OptionContract(
                option_symbol="INVALID",
                underlying_symbol="TEST",
                strike=Decimal("100.00"),
                expiration=date.today(),
                option_type="call",
                delta=1.5,  # Invalid delta > 1
                gamma=0.1,
                theta=-0.1,
                vega=0.1
            )
    
    def test_technical_indicators_model(self):
        """Test TechnicalIndicators model."""
        indicators = TechnicalIndicators(
            symbol="AAPL",
            sma_20=148.50,
            sma_50=145.20,
            sma_200=140.80,
            ema_12=149.75,
            ema_26=147.30,
            rsi_14=58.5,
            macd=2.45,
            macd_signal=1.80,
            macd_histogram=0.65,
            bollinger_upper=155.20,
            bollinger_middle=148.50,
            bollinger_lower=141.80,
            atr_14=3.25,
            volume_sma_20=45_000_000,
            volume_ratio=1.12
        )
        
        assert indicators.symbol == "AAPL"
        assert 0 <= indicators.rsi_14 <= 100
        assert indicators.bollinger_upper > indicators.bollinger_middle > indicators.bollinger_lower
        assert indicators.atr_14 > 0
        assert indicators.volume_ratio > 0
    
    def test_screening_criteria_model(self):
        """Test ScreeningCriteria validation."""
        criteria = ScreeningCriteria(
            min_market_cap=1_000_000_000,
            max_market_cap=100_000_000_000,
            min_price=10.0,
            max_price=500.0,
            min_volume=500_000,
            min_rsi=30.0,
            max_rsi=70.0,
            min_option_volume=100,
            min_open_interest=500,
            max_days_to_expiration=45,
            exclude_sectors=["Utilities"],
            exclude_symbols=["TSLA"]
        )
        
        assert criteria.min_market_cap < criteria.max_market_cap
        assert criteria.min_price < criteria.max_price
        assert criteria.min_rsi < criteria.max_rsi
        assert criteria.max_days_to_expiration > 0
        
        # Test validation
        with pytest.raises(ValueError):
            ScreeningCriteria(
                min_price=100.0,
                max_price=50.0  # Max less than min should fail
            )
    
    def test_workflow_config_model(self):
        """Test WorkflowConfig model."""
        config = WorkflowConfig(
            max_concurrent_stocks=10,
            max_retry_attempts=3,
            step_timeout_seconds=60,
            continue_on_errors=True,
            error_threshold_percent=20.0,
            enable_ai_analysis=True,
            ai_batch_size=5,
            max_ai_cost_dollars=50.0,
            save_intermediate_results=True,
            min_data_quality_score=70.0
        )
        
        assert config.max_concurrent_stocks > 0
        assert config.max_retry_attempts >= 0
        assert 0 <= config.error_threshold_percent <= 100
        assert config.max_ai_cost_dollars > 0
        assert 0 <= config.min_data_quality_score <= 100
    
    def test_analysis_result_model(self):
        """Test AnalysisResult model."""
        result = AnalysisResult(
            symbol="AAPL",
            analysis_id="test-123",
            timestamp=datetime.now(),
            technical_score=75.0,
            momentum_score=82.0,
            options_score=68.0,
            overall_rating=74.0,
            confidence=85.0,
            recommendation="BUY",
            reasoning="Strong technical setup with good momentum",
            key_factors=["Rising RSI", "Bullish MACD crossover"],
            risks=["Market volatility", "Earnings uncertainty"],
            data_quality_score=88.0
        )
        
        assert result.symbol == "AAPL"
        assert 0 <= result.overall_rating <= 100
        assert 0 <= result.confidence <= 100
        assert result.recommendation in ["BUY", "SELL", "HOLD", "WATCHLIST"]
        assert len(result.key_factors) > 0
        assert len(result.risks) > 0


class TestProviderModels:
    """Tests for provider-related models."""
    
    def test_provider_config_model(self):
        """Test ProviderConfig model."""
        config = ProviderConfig(
            provider_type="eodhd",
            api_key="test_key",
            base_url="https://api.example.com",
            rate_limit_per_minute=60,
            timeout_seconds=30,
            max_retries=3,
            backoff_factor=1.5,
            enable_caching=True,
            cache_ttl_seconds=300
        )
        
        assert config.provider_type in ["eodhd", "marketdata", "mock"]
        assert config.rate_limit_per_minute > 0
        assert config.timeout_seconds > 0
        assert config.max_retries >= 0
        assert config.backoff_factor > 0
        assert config.cache_ttl_seconds > 0
    
    def test_cache_config_model(self):
        """Test CacheConfig model."""
        config = CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl_seconds=300,
            max_size_mb=100,
            compression_enabled=True,
            redis_url="redis://localhost:6379",
            cleanup_interval_minutes=60
        )
        
        assert config.backend in ["memory", "redis", "disk"]
        assert config.default_ttl_seconds > 0
        assert config.max_size_mb > 0
        assert config.cleanup_interval_minutes > 0
    
    def test_provider_health_status_model(self):
        """Test ProviderHealthStatus model."""
        status = ProviderHealthStatus(
            provider="eodhd",
            status="healthy",
            response_time_ms=150.0,
            last_check=datetime.now(),
            error_rate_percent=2.5,
            requests_per_minute=45,
            details={"api_version": "1.0", "region": "us-east-1"}
        )
        
        assert status.provider in ["eodhd", "marketdata", "mock"]
        assert status.status in ["healthy", "degraded", "unhealthy"]
        assert status.response_time_ms >= 0
        assert 0 <= status.error_rate_percent <= 100
        assert status.requests_per_minute >= 0
    
    def test_provider_usage_stats_model(self):
        """Test ProviderUsageStats model."""
        stats = ProviderUsageStats(
            provider="marketdata",
            total_requests=1250,
            successful_requests=1200,
            failed_requests=50,
            total_cost=15.75,
            requests_today=150,
            cost_today=2.25,
            average_response_time_ms=200.0,
            rate_limit_hits=3
        )
        
        assert stats.total_requests >= 0
        assert stats.successful_requests + stats.failed_requests <= stats.total_requests
        assert stats.total_cost >= 0
        assert stats.average_response_time_ms >= 0
        assert stats.rate_limit_hits >= 0


class TestWorkflowModels:
    """Tests for workflow-related models."""
    
    def test_workflow_step_model(self):
        """Test WorkflowStep model."""
        step = WorkflowStep(
            step_id="fetch_quotes",
            name="Fetch Stock Quotes",
            description="Retrieve current stock quotes from providers",
            order=1,
            required=True,
            timeout_seconds=30,
            retry_attempts=3,
            dependencies=[]
        )
        
        assert step.step_id
        assert step.order > 0
        assert step.timeout_seconds > 0
        assert step.retry_attempts >= 0
        assert isinstance(step.dependencies, list)
    
    def test_workflow_execution_context_model(self):
        """Test WorkflowExecutionContext model."""
        config = WorkflowConfig(
            max_concurrent_stocks=5,
            max_retry_attempts=2,
            step_timeout_seconds=60
        )
        
        context = WorkflowExecutionContext(
            workflow_id="wf-123",
            config=config,
            start_time=datetime.now(),
            symbols=["AAPL", "MSFT"],
            step_results={},
            errors=[],
            checkpoints=[]
        )
        
        assert context.workflow_id
        assert isinstance(context.symbols, list)
        assert isinstance(context.step_results, dict)
        assert isinstance(context.errors, list)
        assert isinstance(context.checkpoints, list)
    
    def test_workflow_checkpoint_model(self):
        """Test WorkflowCheckpoint model."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp-001",
            workflow_id="wf-123",
            step_id="technical_analysis",
            timestamp=datetime.now(),
            data_snapshot={"symbols_processed": 5, "total_symbols": 10},
            can_resume=True
        )
        
        assert checkpoint.checkpoint_id
        assert checkpoint.workflow_id
        assert checkpoint.step_id
        assert isinstance(checkpoint.data_snapshot, dict)
        assert isinstance(checkpoint.can_resume, bool)
    
    def test_workflow_metrics_model(self):
        """Test WorkflowMetrics model."""
        metrics = WorkflowMetrics(
            workflow_id="wf-123",
            total_execution_time_seconds=125.5,
            symbols_processed=15,
            symbols_failed=2,
            steps_completed=6,
            steps_failed=1,
            total_api_calls=45,
            total_api_cost=12.75,
            memory_usage_mb=150.0,
            cache_hit_rate=0.85
        )
        
        assert metrics.total_execution_time_seconds > 0
        assert metrics.symbols_processed >= 0
        assert metrics.symbols_failed >= 0
        assert metrics.steps_completed >= 0
        assert metrics.steps_failed >= 0
        assert metrics.total_api_calls >= 0
        assert metrics.total_api_cost >= 0
        assert 0 <= metrics.cache_hit_rate <= 1


class TestWorkflowEngine:
    """Tests for workflow engine functionality."""
    
    @pytest.fixture
    def workflow_config(self):
        """Test workflow configuration."""
        return WorkflowConfig(
            max_concurrent_stocks=3,
            max_retry_attempts=2,
            step_timeout_seconds=30,
            continue_on_errors=True,
            error_threshold_percent=25.0,
            enable_ai_analysis=False,  # Disable for unit tests
            save_intermediate_results=True
        )
    
    @pytest.fixture
    def workflow_engine(self, workflow_config):
        """Create workflow engine instance."""
        return WorkflowEngine(workflow_config)
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow_engine):
        """Test workflow engine initialization."""
        assert workflow_engine.config is not None
        assert workflow_engine._current_context is None
        assert len(workflow_engine._step_registry) > 0
    
    @pytest.mark.asyncio
    async def test_step_registration(self, workflow_engine):
        """Test step registration and retrieval."""
        # Define a test step
        async def test_step(context, symbols):
            return StepResult(
                step_id="test_step",
                success=True,
                results={"processed": len(symbols)},
                execution_time_seconds=1.0
            )
        
        workflow_engine.register_step("test_step", test_step, order=1)
        
        assert "test_step" in workflow_engine._step_registry
        registered_step = workflow_engine._step_registry["test_step"]
        assert registered_step["function"] == test_step
        assert registered_step["order"] == 1
    
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self, workflow_engine):
        """Test successful workflow execution."""
        # Register mock steps
        async def mock_step_1(context, symbols):
            await asyncio.sleep(0.1)  # Simulate work
            return StepResult(
                step_id="step_1",
                success=True,
                results={"quotes": [{"symbol": s, "price": 100.0} for s in symbols]},
                execution_time_seconds=0.1
            )
        
        async def mock_step_2(context, symbols):
            await asyncio.sleep(0.1)
            return StepResult(
                step_id="step_2", 
                success=True,
                results={"indicators": [{"symbol": s, "rsi": 65.0} for s in symbols]},
                execution_time_seconds=0.1
            )
        
        workflow_engine.register_step("fetch_quotes", mock_step_1, order=1)
        workflow_engine.register_step("calculate_indicators", mock_step_2, order=2)
        
        symbols = ["AAPL", "MSFT"]
        context = await workflow_engine.execute_workflow(symbols)
        
        assert context.state == WorkflowState.COMPLETED
        assert len(context.step_results) == 2
        assert "fetch_quotes" in context.step_results
        assert "calculate_indicators" in context.step_results
        assert context.step_results["fetch_quotes"].success
        assert context.step_results["calculate_indicators"].success
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_engine):
        """Test workflow error handling and recovery."""
        # Register steps with one that fails
        async def success_step(context, symbols):
            return StepResult(
                step_id="success_step",
                success=True,
                results={"data": "success"},
                execution_time_seconds=0.1
            )
        
        async def failing_step(context, symbols):
            raise Exception("Simulated step failure")
        
        workflow_engine.register_step("success_step", success_step, order=1)
        workflow_engine.register_step("failing_step", failing_step, order=2)
        
        symbols = ["AAPL"]
        context = await workflow_engine.execute_workflow(symbols)
        
        # With continue_on_errors=True, workflow should complete despite failure
        assert context.state in [WorkflowState.COMPLETED_WITH_ERRORS, WorkflowState.FAILED]
        assert len(context.errors) > 0
        assert "success_step" in context.step_results
        assert context.step_results["success_step"].success
    
    @pytest.mark.asyncio
    async def test_step_retry_mechanism(self, workflow_engine):
        """Test step retry functionality."""
        attempt_count = 0
        
        async def flaky_step(context, symbols):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:  # Fail first 2 attempts
                raise Exception(f"Attempt {attempt_count} failed")
            
            return StepResult(
                step_id="flaky_step",
                success=True,
                results={"attempts": attempt_count},
                execution_time_seconds=0.1
            )
        
        workflow_engine.register_step("flaky_step", flaky_step, order=1)
        
        symbols = ["AAPL"]
        context = await workflow_engine.execute_workflow(symbols)
        
        # Should succeed after retries
        assert context.step_results["flaky_step"].success
        assert attempt_count == 3  # 1 initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, workflow_engine):
        """Test step timeout handling."""
        async def slow_step(context, symbols):
            await asyncio.sleep(2.0)  # Longer than 30s timeout in real scenario
            return StepResult(
                step_id="slow_step",
                success=True,
                results={"data": "slow"},
                execution_time_seconds=2.0
            )
        
        # Register with short timeout for testing
        workflow_engine.register_step("slow_step", slow_step, order=1, timeout_seconds=0.5)
        
        symbols = ["AAPL"]
        context = await workflow_engine.execute_workflow(symbols)
        
        # Step should fail due to timeout
        assert "slow_step" in context.step_results
        assert not context.step_results["slow_step"].success
        assert len(context.errors) > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation_and_recovery(self, workflow_engine):
        """Test workflow checkpointing functionality."""
        # Enable checkpointing
        workflow_engine.config.save_intermediate_results = True
        
        async def checkpointed_step(context, symbols):
            return StepResult(
                step_id="checkpointed_step",
                success=True,
                results={"checkpoint_data": "important"},
                execution_time_seconds=0.1
            )
        
        workflow_engine.register_step("checkpointed_step", checkpointed_step, order=1)
        
        symbols = ["AAPL", "MSFT"]
        context = await workflow_engine.execute_workflow(symbols)
        
        # Should have created checkpoints
        assert len(context.checkpoints) > 0
        
        # Test checkpoint recovery
        latest_checkpoint = context.checkpoints[-1]
        recovered_context = await workflow_engine.recover_from_checkpoint(latest_checkpoint.checkpoint_id)
        
        assert recovered_context is not None
        assert recovered_context.workflow_id == context.workflow_id
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, workflow_engine):
        """Test concurrent processing of multiple symbols."""
        execution_times = []
        
        async def timed_step(context, symbols):
            start_time = asyncio.get_event_loop().time()
            await asyncio.sleep(0.2)  # Simulate work
            end_time = asyncio.get_event_loop().time()
            
            execution_times.append(end_time - start_time)
            
            return StepResult(
                step_id="timed_step",
                success=True,
                results={"symbols": symbols},
                execution_time_seconds=end_time - start_time
            )
        
        workflow_engine.register_step("timed_step", timed_step, order=1)
        
        # Process multiple symbols concurrently
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        start_time = asyncio.get_event_loop().time()
        
        context = await workflow_engine.execute_workflow(symbols)
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # With concurrency, total time should be less than sum of individual times
        expected_sequential_time = len(symbols) * 0.2
        assert total_time < expected_sequential_time
        assert context.state == WorkflowState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_collection(self, workflow_engine):
        """Test workflow metrics collection."""
        async def metrics_step(context, symbols):
            return StepResult(
                step_id="metrics_step",
                success=True,
                results={"processed": len(symbols)},
                execution_time_seconds=0.1,
                api_calls_made=len(symbols) * 2,
                api_cost=len(symbols) * 0.01
            )
        
        workflow_engine.register_step("metrics_step", metrics_step, order=1)
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        context = await workflow_engine.execute_workflow(symbols)
        
        metrics = await workflow_engine.get_workflow_metrics(context.workflow_id)
        
        assert metrics is not None
        assert metrics.symbols_processed == 3
        assert metrics.steps_completed >= 1
        assert metrics.total_api_calls > 0
        assert metrics.total_api_cost > 0
    
    def test_step_dependency_resolution(self, workflow_engine):
        """Test step dependency resolution and ordering."""
        # Register steps with dependencies
        workflow_engine.register_step("step_a", lambda: None, order=1)
        workflow_engine.register_step("step_b", lambda: None, order=2, dependencies=["step_a"])
        workflow_engine.register_step("step_c", lambda: None, order=3, dependencies=["step_a", "step_b"])
        
        execution_order = workflow_engine._resolve_execution_order()
        
        # Verify correct ordering based on dependencies
        step_a_index = execution_order.index("step_a")
        step_b_index = execution_order.index("step_b")
        step_c_index = execution_order.index("step_c")
        
        assert step_a_index < step_b_index
        assert step_b_index < step_c_index
        assert step_a_index < step_c_index


class TestSettings:
    """Tests for application settings and configuration."""
    
    def test_settings_initialization(self):
        """Test settings initialization with defaults."""
        settings = Settings()
        
        assert settings.environment in ["development", "testing", "production"]
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert settings.max_stocks_to_screen > 0
        assert isinstance(settings.enable_caching, bool)
        assert isinstance(settings.enable_ai_analysis, bool)
    
    def test_settings_validation(self):
        """Test settings validation.""" 
        # Test invalid environment
        with pytest.raises(ValueError):
            Settings(environment="invalid_env")
        
        # Test invalid log level
        with pytest.raises(ValueError):
            Settings(log_level="INVALID_LEVEL")
        
        # Test invalid numeric values
        with pytest.raises(ValueError):
            Settings(max_stocks_to_screen=-1)
    
    def test_settings_from_environment(self):
        """Test settings loading from environment variables."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'testing',
            'LOG_LEVEL': 'DEBUG',
            'MAX_STOCKS_TO_SCREEN': '50',
            'ENABLE_CACHING': 'false'
        }):
            settings = Settings()
            
            assert settings.environment == "testing"
            assert settings.log_level == "DEBUG"
            assert settings.max_stocks_to_screen == 50
            assert settings.enable_caching is False
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Test with missing API keys
        settings = Settings(
            eodhd_api_key="",
            marketdata_api_key="",
            claude_api_key=""
        )
        
        validation_errors = settings.validate_api_keys()
        assert len(validation_errors) > 0
        
        # Test with valid API keys
        settings = Settings(
            eodhd_api_key="valid_eodhd_key",
            marketdata_api_key="valid_marketdata_key",
            claude_api_key="valid_claude_key"
        )
        
        validation_errors = settings.validate_api_keys()
        assert len(validation_errors) == 0
    
    def test_directory_configuration(self):
        """Test directory configuration and creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            settings = Settings(
                output_directory=str(temp_path / "output"),
                log_directory=str(temp_path / "logs"),
                cache_directory=str(temp_path / "cache")
            )
            
            # Directories should be created if they don't exist
            settings.ensure_directories_exist()
            
            assert Path(settings.output_directory).exists()
            assert Path(settings.log_directory).exists()
            assert Path(settings.cache_directory).exists()
    
    def test_settings_serialization(self):
        """Test settings serialization and deserialization."""
        original_settings = Settings(
            environment="testing",
            debug=True,
            max_stocks_to_screen=100,
            enable_ai_analysis=False
        )
        
        # Serialize to dict
        settings_dict = original_settings.model_dump()
        
        # Deserialize from dict
        restored_settings = Settings(**settings_dict)
        
        assert restored_settings.environment == original_settings.environment
        assert restored_settings.debug == original_settings.debug
        assert restored_settings.max_stocks_to_screen == original_settings.max_stocks_to_screen
        assert restored_settings.enable_ai_analysis == original_settings.enable_ai_analysis
    
    def test_cost_limits_configuration(self):
        """Test AI cost limits configuration."""
        settings = Settings(
            max_ai_cost_per_day=100.0,
            max_ai_cost_per_month=1000.0,
            ai_cost_alert_threshold=80.0
        )
        
        assert settings.max_ai_cost_per_day > 0
        assert settings.max_ai_cost_per_month > settings.max_ai_cost_per_day
        assert 0 < settings.ai_cost_alert_threshold < 100
    
    def test_provider_configuration(self):
        """Test provider-specific configuration."""
        settings = Settings(
            preferred_provider="eodhd",
            provider_timeout_seconds=30,
            provider_max_retries=3,
            enable_provider_failover=True
        )
        
        assert settings.preferred_provider in ["eodhd", "marketdata"]
        assert settings.provider_timeout_seconds > 0
        assert settings.provider_max_retries >= 0
        assert isinstance(settings.enable_provider_failover, bool)
    
    def test_workflow_configuration(self):
        """Test workflow-specific configuration."""
        settings = Settings(
            max_concurrent_stocks=10,
            workflow_step_timeout=60,
            enable_checkpointing=True,
            checkpoint_interval_steps=3
        )
        
        assert settings.max_concurrent_stocks > 0
        assert settings.workflow_step_timeout > 0
        assert isinstance(settings.enable_checkpointing, bool)
        assert settings.checkpoint_interval_steps > 0