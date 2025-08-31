"""
Comprehensive Unit Tests for AI Analysis Components

Tests all components of the Claude AI integration:
- DataPackager: Data combination and formatting
- ClaudeClient: API integration and rate limiting
- PromptTemplates: Prompt generation and formatting
- ResponseParser: JSON parsing and validation
- CostManager: Token counting and cost controls
- RatingEngine: End-to-end workflow orchestration
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

# Import all AI analysis components
import sys
sys.path.append('/home/deployuser/stock-scan/stock-scanner')

from src.ai_analysis.data_packager import DataPackager, DataCompletenessMetrics, serialize_for_json, estimate_token_count
from src.ai_analysis.claude_client import ClaudeClient, ClaudeConfig, ClaudeResponse, MockClaudeClient, create_claude_client
from src.ai_analysis.prompt_templates import PromptTemplates, RatingCriteria, create_analysis_prompt
from src.ai_analysis.response_parser import (
    ResponseParser, ClaudeAnalysis, ComponentScores, OptionRecommendation, 
    FallbackAnalysis, ConfidenceLevel, parse_claude_response
)
from src.ai_analysis.cost_manager import CostManager, CostConfig, TokenCost, DailyUsage, create_cost_manager
from src.ai_analysis.persistence import AnalysisPersistence, AnalysisRecord, BatchAnalysisSession, create_persistence_manager
from src.ai_analysis.rating_engine import RatingEngine, RatingConfig, AnalysisResult, create_rating_engine


class TestDataPackager:
    """Test data packaging and formatting functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.packager = DataPackager(min_completeness_threshold=60.0)
        
        # Sample test data
        self.sample_opportunity = {
            "symbol": "AAPL",
            "current_price": 175.50,
            "overall_score": 82.5,
            "technical_indicators": {
                "rsi": 58.2,
                "macd": 1.25,
                "sma_20": 172.30,
                "sma_50": 168.50,
                "bollinger_bands": {"upper": 178.50, "lower": 166.10}
            },
            "momentum_analysis": {
                "21_day_return": 0.045,
                "momentum_score": 75.0
            },
            "squeeze_analysis": {
                "squeeze_intensity": 0.8,
                "duration_days": 15
            },
            "score_breakdown": {
                "technical_score": 85.0,
                "momentum_score": 80.0
            }
        }
        
        self.sample_option = {
            "option_symbol": "AAPL240315C00170000",
            "underlying_symbol": "AAPL",
            "strike": 170.0,
            "expiration": "2024-03-15",
            "dte": 45,
            "bid": 8.50,
            "ask": 8.75,
            "last": 8.60,
            "delta": 0.65,
            "gamma": 0.025,
            "theta": -0.08,
            "vega": 0.22,
            "implied_volatility": 0.28,
            "volume": 1000,
            "open_interest": 5000,
            "liquidity_score": 85.0
        }
        
        self.sample_enhanced = {
            "fundamentals": {
                "company_info": {
                    "name": "Apple Inc.",
                    "sector": "Technology",
                    "market_cap_mln": 2800000
                },
                "financial_health": {
                    "eps_ttm": 5.95,
                    "profit_margin": 0.253,
                    "roe": 0.147
                },
                "valuation_metrics": {
                    "pe_ratio": 29.5,
                    "price_to_book": 8.2
                }
            },
            "news": [
                {"title": "Apple announces new product", "date": "2024-01-15", "sentiment": "positive"},
                {"title": "Tech sector outlook", "date": "2024-01-14", "sentiment": "neutral"}
            ],
            "earnings": [{"date": "2024-02-01", "estimate": 2.10}],
            "sentiment": {"overall_sentiment": "positive", "sentiment_score": 0.6}
        }
    
    def test_create_analysis_package_success(self):
        """Test successful package creation with complete data"""
        package = self.packager.create_analysis_package(
            self.sample_opportunity, self.sample_option, self.sample_enhanced
        )
        
        # Check package structure
        assert "metadata" in package
        assert "opportunity_analysis" in package
        assert "selected_option_contract" in package
        assert "enhanced_stock_data" in package
        assert "data_quality" in package
        
        # Check metadata
        metadata = package["metadata"]
        assert metadata["symbol"] == "AAPL"
        assert metadata["package_version"] == "1.0"
        assert "package_timestamp" in metadata
        
        # Check data quality metrics
        quality = package["data_quality"]
        assert quality["overall_score"] > 80.0  # Should be high with complete data
        assert quality["fundamental_score"] > 0
        assert quality["technical_score"] > 0
        assert quality["options_score"] > 0
    
    def test_data_completeness_assessment(self):
        """Test data completeness scoring"""
        # Test with complete data
        metrics = self.packager._assess_data_completeness(
            self.sample_opportunity, self.sample_option, self.sample_enhanced
        )
        
        assert metrics.overall_score > 80.0
        assert metrics.fundamental_score > 0
        assert metrics.technical_score > 0
        assert metrics.options_score > 0
        
        # Test with missing data
        incomplete_enhanced = {}
        incomplete_metrics = self.packager._assess_data_completeness(
            self.sample_opportunity, self.sample_option, incomplete_enhanced
        )
        
        assert incomplete_metrics.overall_score < metrics.overall_score
        assert "fundamentals" in incomplete_metrics.missing_fields
    
    def test_data_cleaning(self):
        """Test data cleaning and None value removal"""
        dirty_data = {
            "valid_field": 10.5,
            "none_field": None,
            "empty_string": "",
            "nested": {
                "good_value": 20,
                "bad_value": None
            },
            "list_with_nones": [1, None, 3, None]
        }
        
        cleaned = self.packager._clean_dict(dirty_data)
        
        assert "valid_field" in cleaned
        assert "none_field" not in cleaned
        assert "empty_string" in cleaned  # Empty strings are preserved
        assert cleaned["nested"]["good_value"] == 20
        assert "bad_value" not in cleaned["nested"]
        assert cleaned["list_with_nones"] == [1, 3]
    
    def test_utility_calculations(self):
        """Test utility calculation methods"""
        # Test mid price calculation
        mid_price = self.packager._calculate_mid_price(8.50, 8.75)
        assert mid_price == 8.625
        
        # Test spread percentage
        spread_pct = self.packager._calculate_spread_percent(8.50, 8.75, 8.60)
        expected_spread = ((8.75 - 8.50) / 8.75) * 100
        assert abs(spread_pct - expected_spread) < 0.01
        
        # Test IV/HV ratio
        iv_hv_ratio = self.packager._calculate_iv_hv_ratio(0.28, 0.25)
        assert iv_hv_ratio == 0.28 / 0.25
    
    def test_json_serialization(self):
        """Test JSON serialization functionality"""
        package = self.packager.create_analysis_package(
            self.sample_opportunity, self.sample_option, self.sample_enhanced
        )
        
        # Test serialization
        json_str = serialize_for_json(package)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized["metadata"]["symbol"] == "AAPL"
    
    def test_token_estimation(self):
        """Test token count estimation"""
        package = self.packager.create_analysis_package(
            self.sample_opportunity, self.sample_option, self.sample_enhanced
        )
        
        token_count = estimate_token_count(package)
        assert isinstance(token_count, int)
        assert token_count > 100  # Should be substantial for complete package


class TestClaudeClient:
    """Test Claude API client functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = ClaudeConfig(
            api_key="test-key",
            daily_cost_limit=10.0,
            min_request_interval=1.0  # Shorter for testing
        )
        self.mock_client = MockClaudeClient(self.config)
    
    @pytest.mark.asyncio
    async def test_mock_client_analysis(self):
        """Test mock client functionality"""
        response = await self.mock_client.analyze_opportunity(
            "Test prompt for AAPL analysis",
            "AAPL"
        )
        
        assert response.success is True
        assert response.parsed_json is not None
        assert response.parsed_json["symbol"] == "AAPL"
        assert "rating" in response.parsed_json
        assert response.tokens_used > 0
        assert response.cost_estimate > 0
    
    def test_usage_statistics(self):
        """Test usage statistics tracking"""
        initial_stats = self.mock_client.get_usage_stats()
        assert initial_stats["requests_today"] == 0
        assert initial_stats["cost_today"] == 0.0
        
        # Simulate usage update
        self.mock_client._update_usage_stats(500, 0.005)
        
        updated_stats = self.mock_client.get_usage_stats()
        assert updated_stats["requests_today"] == 1
        assert updated_stats["cost_today"] == 0.005
    
    def test_can_make_request(self):
        """Test request capability checking"""
        can_request, reason = self.mock_client.can_make_request()
        assert can_request is True
        assert reason == "OK"
        
        # Simulate exhausted budget
        self.mock_client.usage_stats.cost_today = self.config.daily_cost_limit
        can_request, reason = self.mock_client.can_make_request()
        assert can_request is False
        assert "cost limit" in reason.lower()
    
    def test_client_creation(self):
        """Test client factory function"""
        # Test mock client creation
        client = create_claude_client("test-key", use_mock=True)
        assert isinstance(client, MockClaudeClient)
        
        # Test without API key
        client_no_key = create_claude_client("", use_mock=True)
        assert isinstance(client_no_key, MockClaudeClient)


class TestPromptTemplates:
    """Test prompt generation and formatting"""
    
    def setup_method(self):
        """Setup test environment"""
        self.templates = PromptTemplates()
        
        self.sample_package = {
            "metadata": {"symbol": "AAPL"},
            "opportunity_analysis": {
                "basic_info": {
                    "symbol": "AAPL",
                    "current_price": 175.50,
                    "overall_score": 82.5
                },
                "technical_analysis": {"rsi": 58.2, "macd": 1.25}
            },
            "selected_option_contract": {
                "contract_details": {
                    "symbol": "AAPL240315C00170000",
                    "strike_price": 170.0,
                    "days_to_expiration": 45
                },
                "pricing_data": {"bid": 8.50, "ask": 8.75}
            },
            "enhanced_stock_data": {
                "fundamental_analysis": {
                    "company_overview": {"name": "Apple Inc.", "sector": "Technology"}
                }
            }
        }
    
    def test_rating_criteria(self):
        """Test rating criteria configuration"""
        criteria = RatingCriteria()
        assert criteria.get_total() == 100
        assert criteria.trend_momentum == 35
        assert criteria.options_quality == 20
    
    def test_prompt_creation(self):
        """Test comprehensive prompt creation"""
        prompt = self.templates.create_analysis_prompt(self.sample_package)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 1000  # Should be substantial
        assert "AAPL" in prompt
        assert "SCORING FRAMEWORK" in prompt
        assert "JSON" in prompt
        assert "35 points" in prompt  # Trend & momentum points
    
    def test_system_prompt(self):
        """Test system prompt generation"""
        system_prompt = self.templates._get_system_prompt()
        assert "expert" in system_prompt.lower()
        assert "options" in system_prompt.lower()
        assert "systematic" in system_prompt.lower()
    
    def test_scoring_framework(self):
        """Test scoring framework formatting"""
        framework = self.templates._get_scoring_framework()
        assert "35 points" in framework  # Trend & momentum
        assert "20 points" in framework  # Options quality
        assert "15 points" in framework  # IV value
        assert "100 POINTS" in framework
    
    def test_response_format(self):
        """Test response format specification"""
        format_spec = self.templates._get_response_format()
        assert "JSON" in format_spec
        assert "component_scores" in format_spec
        assert "opportunities" in format_spec
        assert "risks" in format_spec
    
    def test_convenience_function(self):
        """Test convenience function"""
        prompt = create_analysis_prompt(self.sample_package)
        assert isinstance(prompt, str)
        assert "AAPL" in prompt


class TestResponseParser:
    """Test response parsing and validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.parser = ResponseParser()
        
        self.valid_response = '''
        {
            "symbol": "AAPL",
            "rating": 85,
            "component_scores": {
                "trend_momentum": 28,
                "options_quality": 18,
                "iv_value": 12,
                "squeeze_volatility": 8,
                "fundamentals": 9,
                "event_news": 10
            },
            "confidence": "high",
            "thesis": "Strong technical momentum with favorable IV conditions",
            "opportunities": [
                "Strong uptrend with momentum acceleration",
                "IV in attractive range for premium collection"
            ],
            "risks": [
                "General market volatility risk",
                "Earnings announcement in 3 weeks"
            ],
            "option_contract": {
                "recommendation": "Consider AAPL 180C expiring in 30-45 days",
                "entry_timing": "Current levels offer good entry point",
                "risk_management": "Position size appropriately"
            },
            "red_flags": [],
            "notes": "Solid opportunity with good risk/reward profile"
        }
        '''
        
        self.invalid_response = '''
        {
            "symbol": "AAPL",
            "rating": "not a number",
            "component_scores": {
                "trend_momentum": 28
            }
        }
        '''
    
    def test_valid_response_parsing(self):
        """Test parsing of valid response"""
        analysis = self.parser.parse_response(self.valid_response, "AAPL")
        
        assert analysis.is_valid is True
        assert analysis.symbol == "AAPL"
        assert analysis.rating == 85
        assert analysis.component_scores.get_total() == 85
        assert analysis.confidence == ConfidenceLevel.HIGH
        assert len(analysis.opportunities) >= 2
        assert len(analysis.risks) >= 2
        assert len(analysis.validation_errors) == 0
    
    def test_invalid_response_parsing(self):
        """Test parsing of invalid response"""
        analysis = self.parser.parse_response(self.invalid_response, "AAPL", original_rating=75.0)
        
        assert analysis.is_valid is False
        assert analysis.symbol == "AAPL"
        assert analysis.rating == 75  # Should use fallback
        assert len(analysis.validation_errors) > 0
    
    def test_malformed_json_parsing(self):
        """Test parsing of malformed JSON"""
        malformed = '{"symbol": "AAPL", "rating":'  # Incomplete JSON
        analysis = self.parser.parse_response(malformed, "AAPL", original_rating=80.0)
        
        assert analysis.is_valid is False
        assert analysis.rating == 80  # Should use fallback
        assert "Fallback analysis" in analysis.thesis
    
    def test_json_extraction_strategies(self):
        """Test different JSON extraction strategies"""
        # Test with extra text around JSON
        response_with_text = '''
        Here is my analysis:
        ''' + self.valid_response + '''
        
        This concludes the analysis.
        '''
        
        data = self.parser._parse_json(response_with_text)
        assert data is not None
        assert data["symbol"] == "AAPL"
        
        # Test with code block
        code_block_response = f'''
        ```json
        {self.valid_response}
        ```
        '''
        
        data = self.parser._parse_json(code_block_response)
        assert data is not None
        assert data["symbol"] == "AAPL"
    
    def test_component_score_validation(self):
        """Test component score validation and correction"""
        # Create analysis with mismatched scores
        component_scores = ComponentScores(
            trend_momentum=30,
            options_quality=20,
            iv_value=15,
            squeeze_volatility=10,
            fundamentals=10,
            event_news=10
        )
        
        analysis = ClaudeAnalysis(
            symbol="AAPL",
            rating=85,  # Doesn't match component total of 95
            component_scores=component_scores,
            confidence=ConfidenceLevel.HIGH,
            thesis="Test analysis",
            opportunities=["Test opportunity"],
            risks=["Test risk"],
            option_contract=OptionRecommendation(
                recommendation="Test",
                entry_timing="Test",
                risk_management="Test"
            )
        )
        
        corrected = self.parser.validate_and_correct_scores(analysis)
        assert corrected.component_scores.get_total() == corrected.rating
    
    def test_fallback_analysis(self):
        """Test fallback analysis creation"""
        fallback = FallbackAnalysis(
            symbol="AAPL",
            rating=60,
            error_reason="Test error"
        )
        
        analysis = fallback.to_claude_analysis()
        assert analysis.symbol == "AAPL"
        assert analysis.rating == 60
        assert analysis.is_valid is False
        assert "Test error" in analysis.notes
    
    def test_convenience_function(self):
        """Test convenience parsing function"""
        analysis = parse_claude_response(self.valid_response, "AAPL")
        assert analysis.symbol == "AAPL"
        assert analysis.rating == 85


class TestCostManager:
    """Test cost management and token tracking"""
    
    def setup_method(self):
        """Setup test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_path = Path(temp_dir)
            config = CostConfig(daily_limit=10.0)
            self.manager = CostManager(config, self.temp_path)
    
    def test_cost_estimation(self):
        """Test cost estimation functionality"""
        prompt = "Analyze this options opportunity: " * 100
        estimate = self.manager.estimate_request_cost(prompt)
        
        assert isinstance(estimate, TokenCost)
        assert estimate.input_tokens > 0
        assert estimate.output_tokens > 0
        assert estimate.total_cost > 0
        assert estimate.input_cost < estimate.output_cost  # Output costs more
    
    def test_usage_recording(self):
        """Test actual usage recording"""
        prompt = "Test prompt"
        response = "Test response with analysis results"
        
        initial_usage = self.manager.get_daily_usage()
        assert initial_usage.requests_count == 0
        
        cost = self.manager.record_actual_usage(prompt, response, "AAPL")
        
        updated_usage = self.manager.get_daily_usage()
        assert updated_usage.requests_count == 1
        assert updated_usage.total_cost == cost.total_cost
        assert updated_usage.total_tokens == cost.total_tokens
    
    def test_affordability_checking(self):
        """Test request affordability checking"""
        # Initially should be able to afford requests
        can_afford, reason, remaining = self.manager.can_afford_request(1.0)
        assert can_afford is True
        assert reason == "OK"
        assert remaining > 0
        
        # Simulate usage near limit
        large_usage = DailyUsage(date=date.today().isoformat())
        large_usage.total_cost = 9.5  # Close to $10 limit
        self.manager._daily_usage_cache[date.today().isoformat()] = large_usage
        
        can_afford, reason, remaining = self.manager.can_afford_request(1.0)
        assert can_afford is False
        assert "exceed" in reason.lower()
    
    def test_batch_cost_estimation(self):
        """Test batch cost estimation"""
        prompts = ["Test prompt " + str(i) for i in range(5)]
        estimate = self.manager.estimate_batch_cost(prompts)
        
        assert estimate["batch_size"] == 5
        assert estimate["total_estimated_cost"] > 0
        assert estimate["can_afford_all"] is True
        assert len(estimate["recommendations"]) > 0
        assert estimate["estimated_processing_time_minutes"] > 0
    
    def test_usage_summary(self):
        """Test usage summary generation"""
        # Record some usage
        self.manager.record_actual_usage("prompt1", "response1", "AAPL")
        self.manager.record_actual_usage("prompt2", "response2", "MSFT")
        
        summary = self.manager.get_usage_summary(7)
        
        assert summary["totals"]["total_requests"] == 2
        assert summary["totals"]["total_cost"] > 0
        assert summary["today"]["requests"] == 2
        assert summary["today"]["percentage_used"] > 0
        assert len(summary["daily_breakdown"]) == 7
    
    def test_usage_warnings(self):
        """Test usage warning system"""
        with patch('logging.Logger.warning') as mock_warning:
            # Simulate high usage
            high_usage = DailyUsage(date=date.today().isoformat())
            high_usage.total_cost = 8.5  # 85% of $10 limit
            self.manager._daily_usage_cache[date.today().isoformat()] = high_usage
            
            self.manager._check_usage_warnings(date.today().isoformat())
            mock_warning.assert_called_once()
    
    def test_factory_function(self):
        """Test cost manager factory function"""
        manager = create_cost_manager(daily_limit=20.0)
        assert manager.config.daily_limit == 20.0
        assert isinstance(manager, CostManager)


class TestPersistence:
    """Test data persistence functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.persistence = AnalysisPersistence(self.temp_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.temp_dir.cleanup()
    
    def test_directory_creation(self):
        """Test directory structure creation"""
        expected_dirs = ["daily_records", "data_packages", "sessions", "errors", "exports"]
        for directory in expected_dirs:
            assert (self.temp_path / directory).exists()
    
    def test_analysis_record_saving(self):
        """Test saving analysis records"""
        sample_package = {"symbol": "AAPL", "data": "test"}
        
        record_id = self.persistence.save_analysis_record(
            symbol="AAPL",
            data_package=sample_package,
            prompt="Test prompt",
            raw_response='{"rating": 85}',
            performance_metadata={"response_time": 2.5}
        )
        
        assert isinstance(record_id, str)
        assert len(record_id) > 0
        
        # Check that record file was created
        today_str = date.today().isoformat()
        daily_file = self.temp_path / "daily_records" / f"analyses_{today_str}.jsonl"
        assert daily_file.exists()
    
    def test_batch_session_tracking(self):
        """Test batch session management"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        session_id = self.persistence.start_batch_session(symbols)
        
        assert isinstance(session_id, str)
        assert "batch_" in session_id
        assert self.persistence.current_session is not None
        assert self.persistence.current_session.total_opportunities == 3
        
        # Finish session
        summary = self.persistence.finish_batch_session()
        assert summary is not None
        assert summary["total_opportunities"] == 3
        assert self.persistence.current_session is None
    
    def test_daily_summary(self):
        """Test daily summary generation"""
        # Save some test records
        for i, symbol in enumerate(["AAPL", "MSFT"]):
            self.persistence.save_analysis_record(
                symbol=symbol,
                data_package={"symbol": symbol},
                prompt=f"Test prompt {i}",
                raw_response=f'{{"rating": {80 + i}}}',
                performance_metadata={"cost": 0.01}
            )
        
        summary = self.persistence.get_daily_summary()
        
        assert summary["total_analyses"] == 2
        assert summary["unique_symbols"] == 2
        assert summary["date"] == date.today().isoformat()
    
    def test_record_retrieval(self):
        """Test record retrieval functionality"""
        # Save a record
        sample_package = {"symbol": "AAPL"}
        record_id = self.persistence.save_analysis_record(
            symbol="AAPL",
            data_package=sample_package,
            prompt="Test prompt"
        )
        
        # Retrieve by ID
        retrieved = self.persistence.get_analysis_by_id(record_id)
        assert retrieved is not None
        assert retrieved["symbol"] == "AAPL"
        assert retrieved["record_id"] == record_id
        
        # Retrieve by symbol
        symbol_records = self.persistence.get_analyses_for_symbol("AAPL")
        assert len(symbol_records) == 1
        assert symbol_records[0]["symbol"] == "AAPL"
    
    def test_factory_function(self):
        """Test persistence factory function"""
        persistence = create_persistence_manager(self.temp_path)
        assert isinstance(persistence, AnalysisPersistence)
        assert persistence.storage_path == self.temp_path


class TestRatingEngine:
    """Test end-to-end rating engine functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.config = RatingConfig(
            claude_api_key="test-key",
            claude_daily_limit=10.0,
            storage_path=self.temp_path,
            use_mock_client=True,
            min_request_interval=0.1  # Faster for testing
        )
        
        self.engine = RatingEngine(self.config)
        
        # Sample data
        self.sample_opportunity = {
            "symbol": "AAPL",
            "current_price": 175.50,
            "overall_score": 82.5,
            "technical_indicators": {"rsi": 58.2}
        }
        
        self.sample_option = {
            "option_symbol": "AAPL240315C00170000",
            "strike": 170.0,
            "bid": 8.50,
            "ask": 8.75,
            "delta": 0.65,
            "implied_volatility": 0.28
        }
        
        self.sample_enhanced = {
            "fundamentals": {
                "company_info": {"name": "Apple Inc.", "sector": "Technology"},
                "financial_health": {"eps_ttm": 5.95}
            }
        }
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.temp_dir.cleanup()
    
    @pytest.mark.asyncio
    async def test_single_opportunity_analysis(self):
        """Test single opportunity analysis"""
        result = await self.engine.analyze_single_opportunity(
            self.sample_opportunity, self.sample_option, self.sample_enhanced
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.symbol == "AAPL"
        assert result.success is True
        assert result.ai_analysis is not None
        assert result.combined_score is not None
        assert result.cost > 0
        assert result.processing_time > 0
        assert result.record_id is not None
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        """Test batch analysis functionality"""
        opportunities = [
            (self.sample_opportunity, self.sample_option, self.sample_enhanced),
            ({**self.sample_opportunity, "symbol": "MSFT"}, self.sample_option, self.sample_enhanced)
        ]
        
        results = await self.engine.analyze_opportunity_batch(opportunities)
        
        assert len(results) == 2
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert results[0].symbol == "AAPL"
        assert results[1].symbol == "MSFT"
    
    def test_combined_score_calculation(self):
        """Test combined score calculation"""
        # Test with both scores
        combined = self.engine._calculate_combined_score(80.0, 85)
        expected = (80.0 * 0.6) + (85 * 0.4)
        assert abs(combined - expected) < 0.01
        
        # Test with only AI score
        ai_only = self.engine._calculate_combined_score(None, 75)
        assert ai_only == 75.0
        
        # Test score bounds
        bounded = self.engine._calculate_combined_score(120.0, -10)
        assert 0 <= bounded <= 100
    
    def test_usage_statistics(self):
        """Test usage statistics collection"""
        stats = self.engine.get_usage_statistics()
        
        assert "cost_management" in stats
        assert "claude_api" in stats
        assert "daily_analysis" in stats
        assert "configuration" in stats
        
        config = stats["configuration"]
        assert config["daily_limit"] == self.config.claude_daily_limit
        assert config["mock_mode"] is True
    
    def test_analysis_capability_check(self):
        """Test analysis capability checking"""
        can_analyze, reason = self.engine.can_analyze_now()
        assert can_analyze is True
        assert reason == "Ready for analysis"
    
    def test_factory_function(self):
        """Test rating engine factory function"""
        engine = create_rating_engine(
            claude_api_key="test-key",
            daily_limit=20.0,
            use_mock=True
        )
        
        assert isinstance(engine, RatingEngine)
        assert engine.config.claude_daily_limit == 20.0
        assert engine.config.use_mock_client is True


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        self.temp_dir.cleanup()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create engine
        engine = create_rating_engine(
            claude_api_key="test-key",
            daily_limit=5.0,
            storage_path=self.temp_path,
            use_mock=True
        )
        
        # Prepare test data
        opportunity = {
            "symbol": "AAPL",
            "current_price": 175.50,
            "overall_score": 85.0,
            "technical_indicators": {
                "rsi": 65.0,
                "macd": 1.5,
                "sma_20": 172.0,
                "sma_50": 168.0,
                "bollinger_bands": {"upper": 180.0, "lower": 165.0}
            },
            "momentum_analysis": {
                "21_day_return": 0.08,
                "63_day_return": 0.15,
                "momentum_score": 88.0
            },
            "squeeze_analysis": {
                "squeeze_intensity": 0.75,
                "duration_days": 12,
                "breakout_probability": 0.82
            },
            "score_breakdown": {
                "technical_score": 87.0,
                "momentum_score": 85.0,
                "liquidity_score": 92.0
            }
        }
        
        option_data = {
            "option_symbol": "AAPL240315C00170000",
            "underlying_symbol": "AAPL",
            "strike": 170.0,
            "expiration": "2024-03-15",
            "dte": 45,
            "option_type": "call",
            "bid": 8.50,
            "ask": 8.75,
            "last": 8.60,
            "delta": 0.68,
            "gamma": 0.025,
            "theta": -0.08,
            "vega": 0.22,
            "rho": 0.05,
            "implied_volatility": 0.28,
            "iv_percentile": 65.0,
            "historical_volatility": 0.25,
            "volume": 2500,
            "open_interest": 8000,
            "avg_volume": 1800,
            "liquidity_score": 88.0,
            "option_score": 86.0,
            "selection_reason": "High delta with good liquidity"
        }
        
        enhanced_data = {
            "economic_events": [
                {
                    "date": "2024-01-15",
                    "country": "US",
                    "event": "Consumer Price Index",
                    "impact": "medium",
                    "forecast": "3.2%",
                    "previous": "3.1%"
                }
            ],
            "news": [
                {
                    "date": "2024-01-15",
                    "title": "Apple reports strong iPhone sales in Q4",
                    "content": "Apple Inc. announced better-than-expected iPhone sales...",
                    "sentiment": "positive",
                    "url": "https://example.com/news1"
                },
                {
                    "date": "2024-01-14",
                    "title": "Tech sector shows resilience amid market volatility",
                    "content": "Technology stocks continue to outperform...",
                    "sentiment": "positive",
                    "url": "https://example.com/news2"
                }
            ],
            "fundamentals": {
                "company_info": {
                    "name": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "market_cap_mln": 2850000,
                    "employees": 164000,
                    "description": "Apple Inc. designs, manufactures, and markets consumer electronics..."
                },
                "financial_health": {
                    "eps_ttm": 6.05,
                    "profit_margin": 0.26,
                    "operating_margin": 0.31,
                    "roe": 0.173,
                    "roa": 0.089,
                    "revenue_growth_yoy": 0.08,
                    "earnings_growth_yoy": 0.12,
                    "dividend_yield": 0.0045,
                    "revenue_ttm": 394000000000,
                    "revenue_per_share": 25.12
                },
                "valuation_metrics": {
                    "pe_ratio": 28.9,
                    "forward_pe": 26.5,
                    "price_to_sales": 7.2,
                    "price_to_book": 8.8,
                    "enterprise_value": 2900000000000,
                    "ev_to_revenue": 7.4,
                    "ev_to_ebitda": 22.1
                },
                "analyst_sentiment": {
                    "avg_rating": 4.3,
                    "target_price": 195.0,
                    "strong_buy": 18,
                    "buy": 12,
                    "hold": 6,
                    "sell": 1,
                    "strong_sell": 0
                }
            },
            "live_price": {
                "price": 175.50,
                "change": 2.35,
                "change_p": 1.36,
                "volume": 52000000,
                "timestamp": "2024-01-15T21:00:00Z"
            },
            "earnings": [
                {
                    "date": "2024-02-01",
                    "estimate": 2.15,
                    "symbol": "AAPL"
                }
            ],
            "sentiment": {
                "sentiment": "Positive",
                "sentiment_score": 0.68,
                "buzz": 85,
                "sentiment_change": 0.05
            }
        }
        
        # Run analysis
        result = await engine.analyze_single_opportunity(
            opportunity, option_data, enhanced_data
        )
        
        # Verify results
        assert result.success is True
        assert result.symbol == "AAPL"
        assert result.ai_analysis is not None
        assert result.ai_analysis.rating > 0
        assert result.combined_score is not None
        assert result.cost > 0
        assert result.processing_time > 0
        assert result.record_id is not None
        
        # Verify AI analysis structure
        ai_analysis = result.ai_analysis
        assert ai_analysis.symbol == "AAPL"
        assert 0 <= ai_analysis.rating <= 100
        assert ai_analysis.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
        assert len(ai_analysis.thesis) > 10
        assert len(ai_analysis.opportunities) > 0
        assert len(ai_analysis.risks) > 0
        assert ai_analysis.option_contract.recommendation
        
        # Verify component scores
        component_total = ai_analysis.component_scores.get_total()
        assert abs(component_total - ai_analysis.rating) <= 2  # Allow small rounding differences
        
        # Verify persistence
        retrieved = engine.persistence.get_analysis_by_id(result.record_id)
        assert retrieved is not None
        assert retrieved["symbol"] == "AAPL"
        
        # Verify cost tracking
        usage_stats = engine.get_usage_statistics()
        assert usage_stats["cost_management"]["today"]["requests"] > 0
        assert usage_stats["cost_management"]["today"]["cost"] > 0
        
        print(f"✓ End-to-end test completed successfully:")
        print(f"  Symbol: {result.symbol}")
        print(f"  AI Rating: {ai_analysis.rating}")
        print(f"  Combined Score: {result.combined_score}")
        print(f"  Cost: ${result.cost:.4f}")
        print(f"  Processing Time: {result.processing_time:.1f}s")
        print(f"  Record ID: {result.record_id}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])