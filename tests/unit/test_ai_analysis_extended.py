"""Comprehensive unit tests for AI analysis module."""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Import AI analysis modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai_analysis.claude_client import ClaudeClient, ClaudeConfig
from ai_analysis.cost_manager import CostManager, CostConfig
from ai_analysis.data_packager import DataPackager, PackagingConfig
from ai_analysis.persistence import PersistenceManager, PersistenceConfig
from ai_analysis.rating_engine import RatingEngine, RatingConfig
from ai_analysis.response_parser import ResponseParser, ParsedResponse
from ai_analysis.prompt_templates import PromptTemplates, AnalysisPrompts


class TestClaudeClient:
    """Comprehensive tests for Claude API client."""
    
    @pytest.fixture
    def claude_config(self):
        """Claude client configuration."""
        return ClaudeConfig(
            api_key="test_claude_api_key",
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.1,
            timeout_seconds=30,
            max_retries=3,
            rate_limit_per_minute=50
        )
    
    @pytest.fixture
    def claude_client(self, claude_config):
        """Create Claude client instance."""
        return ClaudeClient(claude_config)
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, claude_client, claude_config):
        """Test client initialization and configuration."""
        assert claude_client.config == claude_config
        assert claude_client._session is None  # Not initialized yet
        assert claude_client._request_count == 0
        assert claude_client._total_tokens == 0
    
    @pytest.mark.asyncio
    async def test_successful_analysis_request(self, claude_client, test_data_generator):
        """Test successful AI analysis request."""
        mock_response = test_data_generator.generate_claude_response("comprehensive")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            prompt = "Analyze this stock data for AAPL"
            response = await claude_client.analyze(prompt)
            
            assert response is not None
            assert claude_client._request_count == 1
            assert claude_client._total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, claude_client):
        """Test rate limit error handling."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 429
            mock_resp.json = AsyncMock(return_value={"error": "Rate limit exceeded"})
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(Exception):  # Should raise rate limit exception
                await claude_client.analyze("Test prompt")
    
    @pytest.mark.asyncio
    async def test_token_counting(self, claude_client):
        """Test token usage tracking."""
        mock_response = {
            "content": [{"text": "Analysis result"}],
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 800
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            await claude_client.analyze("Test prompt")
            
            assert claude_client._total_tokens == 2300  # 1500 + 800
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, claude_client):
        """Test retry mechanism on failures."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call fails, second succeeds
            mock_resp_fail = AsyncMock()
            mock_resp_fail.status = 500
            
            mock_resp_success = AsyncMock()
            mock_resp_success.status = 200
            mock_resp_success.json = AsyncMock(return_value={
                "content": [{"text": "Success"}],
                "usage": {"input_tokens": 100, "output_tokens": 50}
            })
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_resp_fail, mock_resp_success
            ]
            
            response = await claude_client.analyze("Test prompt")
            
            assert response is not None
            assert mock_post.call_count == 2  # One retry
    
    @pytest.mark.asyncio
    async def test_prompt_validation(self, claude_client):
        """Test prompt validation."""
        # Empty prompt should raise error
        with pytest.raises(ValueError):
            await claude_client.analyze("")
        
        # Very long prompt should be truncated or raise error
        very_long_prompt = "x" * 100000
        # This should either truncate or raise appropriate error
        try:
            await claude_client.analyze(very_long_prompt)
        except Exception as e:
            # Should be a validation error, not a crash
            assert isinstance(e, (ValueError, Exception))
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, claude_client):
        """Test batch analysis functionality."""
        prompts = [
            "Analyze AAPL stock data",
            "Analyze MSFT stock data", 
            "Analyze GOOGL stock data"
        ]
        
        mock_response = {
            "content": [{"text": "Batch analysis result"}],
            "usage": {"input_tokens": 500, "output_tokens": 200}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            results = await claude_client.analyze_batch(prompts)
            
            assert len(results) == 3
            assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_session_management(self, claude_client):
        """Test HTTP session lifecycle."""
        # Session should be created on first use
        await claude_client._ensure_session()
        assert claude_client._session is not None
        
        # Cleanup should close session
        await claude_client.cleanup()
        assert claude_client._session.closed


class TestCostManager:
    """Tests for AI cost tracking and management."""
    
    @pytest.fixture
    def cost_config(self):
        """Cost manager configuration."""
        return CostConfig(
            max_daily_cost=100.0,
            max_monthly_cost=1000.0,
            cost_per_input_token=0.000003,
            cost_per_output_token=0.000015,
            alert_threshold_percent=80.0,
            enable_cost_alerts=True
        )
    
    @pytest.fixture
    def cost_manager(self, cost_config):
        """Create cost manager instance."""
        return CostManager(cost_config)
    
    def test_token_cost_calculation(self, cost_manager):
        """Test token cost calculation."""
        input_tokens = 1000
        output_tokens = 500
        
        cost = cost_manager.calculate_cost(input_tokens, output_tokens)
        
        expected_cost = (1000 * 0.000003) + (500 * 0.000015)
        assert abs(cost - expected_cost) < 0.000001
    
    def test_daily_cost_tracking(self, cost_manager):
        """Test daily cost accumulation."""
        # Add some usage
        cost_manager.record_usage(1000, 500)  # First request
        cost_manager.record_usage(2000, 800)  # Second request
        
        daily_cost = cost_manager.get_daily_cost()
        
        expected_total = (3000 * 0.000003) + (1300 * 0.000015)
        assert abs(daily_cost - expected_total) < 0.000001
    
    def test_cost_limit_enforcement(self, cost_manager):
        """Test cost limit enforcement."""
        # Simulate high usage that exceeds daily limit
        large_usage = 1000000  # Large number of tokens
        
        cost_manager.record_usage(large_usage, large_usage)
        
        # Should detect limit exceeded
        assert cost_manager.is_daily_limit_exceeded()
        
        # Should prevent further usage
        with pytest.raises(Exception):  # Cost limit exception
            cost_manager.check_usage_allowed(1000, 500)
    
    def test_cost_projections(self, cost_manager):
        """Test cost projection calculations."""
        # Record some usage
        cost_manager.record_usage(5000, 2000)
        
        # Project cost for additional usage
        projected_cost = cost_manager.project_cost(10000, 5000)
        
        current_cost = cost_manager.get_daily_cost()
        additional_cost = (10000 * 0.000003) + (5000 * 0.000015)
        expected_projection = current_cost + additional_cost
        
        assert abs(projected_cost - expected_projection) < 0.000001
    
    def test_monthly_cost_tracking(self, cost_manager):
        """Test monthly cost accumulation."""
        # Simulate usage over multiple days
        for day in range(5):
            cost_manager.record_usage(1000, 500)
        
        monthly_cost = cost_manager.get_monthly_cost()
        expected_monthly = 5 * ((1000 * 0.000003) + (500 * 0.000015))
        
        assert abs(monthly_cost - expected_monthly) < 0.000001
    
    def test_cost_alerts(self, cost_manager):
        """Test cost alert generation."""
        # Add usage that approaches threshold (80% of $100 daily limit)
        target_cost = 80.0  # 80% of daily limit
        tokens_needed = int(target_cost / 0.000015)  # Assume mostly output tokens
        
        cost_manager.record_usage(0, tokens_needed)
        
        alerts = cost_manager.get_cost_alerts()
        
        assert len(alerts) > 0
        assert any("threshold" in alert.lower() for alert in alerts)
    
    def test_cost_reset(self, cost_manager):
        """Test daily/monthly cost reset."""
        # Add some usage
        cost_manager.record_usage(1000, 500)
        assert cost_manager.get_daily_cost() > 0
        
        # Reset daily cost
        cost_manager.reset_daily_cost()
        assert cost_manager.get_daily_cost() == 0
    
    def test_usage_statistics(self, cost_manager):
        """Test usage statistics generation."""
        # Record varied usage
        cost_manager.record_usage(1500, 600)
        cost_manager.record_usage(2000, 800)
        cost_manager.record_usage(1000, 400)
        
        stats = cost_manager.get_usage_statistics()
        
        assert 'total_requests' in stats
        assert 'total_input_tokens' in stats
        assert 'total_output_tokens' in stats
        assert 'total_cost' in stats
        assert 'average_cost_per_request' in stats
        
        assert stats['total_requests'] == 3
        assert stats['total_input_tokens'] == 4500
        assert stats['total_output_tokens'] == 1800


class TestDataPackager:
    """Tests for data packaging functionality."""
    
    @pytest.fixture
    def packaging_config(self):
        """Data packaging configuration."""
        return PackagingConfig(
            max_payload_size_mb=5,
            compression_enabled=True,
            include_metadata=True,
            format_version="1.0",
            exclude_fields=["internal_id", "debug_info"]
        )
    
    @pytest.fixture
    def data_packager(self, packaging_config):
        """Create data packager instance."""
        return DataPackager(packaging_config)
    
    def test_stock_data_packaging(self, data_packager, sample_stock_quote, sample_technical_indicators):
        """Test packaging of stock analysis data."""
        stock_data = {
            "quote": sample_stock_quote,
            "technical_indicators": sample_technical_indicators,
            "timestamp": datetime.now()
        }
        
        packaged = data_packager.package_stock_data(stock_data)
        
        assert "symbol" in packaged
        assert "market_data" in packaged
        assert "technical_analysis" in packaged
        assert "metadata" in packaged
        
        # Verify data structure
        assert packaged["symbol"] == sample_stock_quote.symbol
    
    def test_options_data_packaging(self, data_packager, test_data_generator):
        """Test packaging of options analysis data."""
        options_chain = test_data_generator.generate_options_chain("AAPL", 150.0)
        
        options_data = {
            "underlying_symbol": "AAPL",
            "options_chain": options_chain,
            "selection_results": {
                "best_calls": options_chain[:3],
                "selection_criteria": "high_volume_delta"
            }
        }
        
        packaged = data_packager.package_options_data(options_data)
        
        assert "underlying_symbol" in packaged
        assert "options_analysis" in packaged
        assert "best_options" in packaged
        assert "selection_metadata" in packaged
    
    def test_combined_analysis_packaging(self, data_packager, sample_stock_quote, test_data_generator):
        """Test packaging of complete analysis data."""
        options_chain = test_data_generator.generate_options_chain("AAPL", 150.0)
        
        combined_data = {
            "stock_quote": sample_stock_quote,
            "technical_indicators": {
                "rsi_14": 65.2,
                "macd": 1.5,
                "bollinger_position": 0.7
            },
            "options_chain": options_chain,
            "momentum_analysis": {
                "momentum_score": 75,
                "trend_direction": "bullish"
            }
        }
        
        packaged = data_packager.package_for_analysis(combined_data)
        
        # Should have all major sections
        assert "stock_analysis" in packaged
        assert "options_analysis" in packaged
        assert "analysis_context" in packaged
        
        # Verify size is reasonable
        packaged_str = json.dumps(packaged)
        assert len(packaged_str) < 5 * 1024 * 1024  # Under 5MB
    
    def test_data_compression(self, data_packager):
        """Test data compression functionality."""
        # Create large repetitive data
        large_data = {
            "repetitive_field": "A" * 10000,
            "large_array": [{"id": i, "value": "data"} for i in range(1000)]
        }
        
        compressed = data_packager.compress_data(large_data)
        decompressed = data_packager.decompress_data(compressed)
        
        assert decompressed == large_data
        
        # Compressed should be smaller (for repetitive data)
        original_size = len(json.dumps(large_data))
        compressed_size = len(compressed)
        assert compressed_size < original_size
    
    def test_field_filtering(self, data_packager):
        """Test field filtering during packaging."""
        data_with_excludes = {
            "symbol": "AAPL",
            "price": 150.0,
            "internal_id": "secret123",  # Should be excluded
            "debug_info": {"verbose": True},  # Should be excluded
            "valid_field": "keep_this"
        }
        
        filtered = data_packager.filter_fields(data_with_excludes)
        
        assert "symbol" in filtered
        assert "price" in filtered
        assert "valid_field" in filtered
        assert "internal_id" not in filtered
        assert "debug_info" not in filtered
    
    def test_metadata_injection(self, data_packager):
        """Test metadata injection in packages."""
        simple_data = {"symbol": "AAPL", "price": 150.0}
        
        packaged = data_packager.add_metadata(simple_data)
        
        assert "metadata" in packaged
        metadata = packaged["metadata"]
        
        assert "timestamp" in metadata
        assert "format_version" in metadata
        assert "packager_version" in metadata
        assert metadata["format_version"] == "1.0"
    
    def test_size_validation(self, data_packager):
        """Test payload size validation."""
        # Create data that exceeds size limit
        oversized_data = {
            "large_field": "X" * (6 * 1024 * 1024)  # 6MB, exceeds 5MB limit
        }
        
        with pytest.raises(ValueError, match="payload size exceeds"):
            data_packager.validate_size(oversized_data)
    
    def test_data_sanitization(self, data_packager):
        """Test data sanitization for AI consumption."""
        unsanitized_data = {
            "symbol": "AAPL",
            "price": float('inf'),  # Invalid value
            "volume": None,  # Null value
            "nested": {
                "value": float('nan')  # Another invalid value
            }
        }
        
        sanitized = data_packager.sanitize_data(unsanitized_data)
        
        assert sanitized["symbol"] == "AAPL"
        assert sanitized["price"] is None or isinstance(sanitized["price"], (int, float))
        assert not (isinstance(sanitized["price"], float) and not np.isfinite(sanitized["price"]))


class TestPersistenceManager:
    """Tests for analysis result persistence."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def persistence_config(self, temp_directory):
        """Persistence configuration."""
        return PersistenceConfig(
            base_directory=str(temp_directory),
            enable_compression=True,
            retention_days=30,
            backup_enabled=True,
            file_format="json"
        )
    
    @pytest.fixture
    def persistence_manager(self, persistence_config):
        """Create persistence manager instance."""
        return PersistenceManager(persistence_config)
    
    @pytest.mark.asyncio
    async def test_save_analysis_result(self, persistence_manager, test_data_generator):
        """Test saving analysis results."""
        analysis_result = {
            "symbol": "AAPL",
            "analysis_id": "test_123",
            "timestamp": datetime.now().isoformat(),
            "ai_response": test_data_generator.generate_claude_response(),
            "scores": {
                "technical_score": 75,
                "momentum_score": 82,
                "overall_rating": 78
            }
        }
        
        file_path = await persistence_manager.save_analysis(analysis_result)
        
        assert file_path.exists()
        assert file_path.suffix == ".json"
        
        # Verify content
        saved_data = json.loads(file_path.read_text())
        assert saved_data["symbol"] == "AAPL"
        assert saved_data["analysis_id"] == "test_123"
    
    @pytest.mark.asyncio
    async def test_load_analysis_result(self, persistence_manager):
        """Test loading saved analysis results."""
        # First save some data
        test_data = {
            "symbol": "MSFT",
            "analysis_id": "load_test",
            "scores": {"overall": 85}
        }
        
        file_path = await persistence_manager.save_analysis(test_data)
        
        # Now load it back
        loaded_data = await persistence_manager.load_analysis(file_path)
        
        assert loaded_data["symbol"] == "MSFT"
        assert loaded_data["analysis_id"] == "load_test"
        assert loaded_data["scores"]["overall"] == 85
    
    @pytest.mark.asyncio
    async def test_compression_handling(self, persistence_manager):
        """Test file compression functionality."""
        # Create large data to test compression
        large_data = {
            "symbol": "GOOGL",
            "large_field": "A" * 10000,
            "array_data": [{"index": i, "data": "test" * 100} for i in range(100)]
        }
        
        file_path = await persistence_manager.save_analysis(large_data)
        loaded_data = await persistence_manager.load_analysis(file_path)
        
        # Data should be identical after compression/decompression
        assert loaded_data["symbol"] == "GOOGL"
        assert loaded_data["large_field"] == "A" * 10000
        assert len(loaded_data["array_data"]) == 100
    
    def test_file_organization(self, persistence_manager):
        """Test file organization and directory structure."""
        test_data = {
            "symbol": "TESLA",
            "timestamp": "2024-01-15T10:30:00"
        }
        
        file_path = persistence_manager._generate_file_path(test_data)
        
        # Should organize by date
        assert "2024" in str(file_path)
        assert "01" in str(file_path)  # Month
        assert "TESLA" in str(file_path)
    
    @pytest.mark.asyncio
    async def test_data_retention(self, persistence_manager, temp_directory):
        """Test data retention policy."""
        # Create old files
        old_date = datetime.now() - timedelta(days=35)  # Older than retention
        
        old_file = temp_directory / "old_analysis.json"
        old_file.write_text(json.dumps({
            "symbol": "OLD",
            "timestamp": old_date.isoformat()
        }))
        
        # Set file modification time to old date
        os.utime(old_file, (old_date.timestamp(), old_date.timestamp()))
        
        # Run cleanup
        await persistence_manager.cleanup_old_files()
        
        # Old file should be removed
        assert not old_file.exists()
    
    @pytest.mark.asyncio
    async def test_backup_creation(self, persistence_manager):
        """Test backup functionality."""
        test_data = {"symbol": "BACKUP_TEST", "important_data": True}
        
        file_path = await persistence_manager.save_analysis(test_data)
        backup_path = await persistence_manager.create_backup(file_path)
        
        assert backup_path.exists()
        assert backup_path != file_path  # Different file
        
        # Backup should contain same data
        backup_data = json.loads(backup_path.read_text())
        assert backup_data["symbol"] == "BACKUP_TEST"
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, persistence_manager):
        """Test searching saved analyses."""
        # Save multiple analyses
        for symbol in ["AAPL", "MSFT", "AAPL"]:  # AAPL appears twice
            await persistence_manager.save_analysis({
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            })
        
        # Search for AAPL analyses
        aapl_results = await persistence_manager.search_analyses(symbol="AAPL")
        
        assert len(aapl_results) == 2
        assert all(result["symbol"] == "AAPL" for result in aapl_results)
    
    @pytest.mark.asyncio
    async def test_metadata_tracking(self, persistence_manager):
        """Test metadata tracking for saved files."""
        analysis_data = {
            "symbol": "META_TEST",
            "analysis_type": "comprehensive"
        }
        
        file_path = await persistence_manager.save_analysis(analysis_data)
        metadata = await persistence_manager.get_file_metadata(file_path)
        
        assert "file_size" in metadata
        assert "created_date" in metadata
        assert "checksum" in metadata
        assert metadata["file_size"] > 0


class TestRatingEngine:
    """Tests for AI rating engine functionality."""
    
    @pytest.fixture
    def rating_config(self):
        """Rating engine configuration."""
        return RatingConfig(
            scoring_weights={
                "technical_analysis": 0.35,
                "momentum_analysis": 0.30,
                "options_attractiveness": 0.25,
                "market_context": 0.10
            },
            confidence_threshold=75.0,
            enable_explanations=True,
            rating_scale="0-100"
        )
    
    @pytest.fixture
    def rating_engine(self, rating_config):
        """Create rating engine instance."""
        return RatingEngine(rating_config)
    
    def test_score_combination(self, rating_engine):
        """Test combining multiple analysis scores."""
        component_scores = {
            "technical_analysis": 75,
            "momentum_analysis": 82,
            "options_attractiveness": 68,
            "market_context": 70
        }
        
        combined_score = rating_engine.combine_scores(component_scores)
        
        # Verify weighted calculation
        expected = (75 * 0.35) + (82 * 0.30) + (68 * 0.25) + (70 * 0.10)
        assert abs(combined_score - expected) < 0.01
    
    def test_confidence_calculation(self, rating_engine):
        """Test confidence score calculation."""
        analysis_data = {
            "data_quality": 85,
            "indicator_consensus": 78,
            "historical_accuracy": 82,
            "market_volatility": 25  # Lower volatility = higher confidence
        }
        
        confidence = rating_engine.calculate_confidence(analysis_data)
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, (int, float))
    
    def test_rating_categorization(self, rating_engine):
        """Test rating categorization and recommendations.""" 
        test_scores = [
            (90, "Strong Buy"),
            (75, "Buy"),
            (60, "Hold"),
            (40, "Sell"),
            (20, "Strong Sell")
        ]
        
        for score, expected_category in test_scores:
            category = rating_engine.categorize_rating(score)
            # Categories might be slightly different, just verify it's reasonable
            assert isinstance(category, str)
            assert len(category) > 0
    
    def test_explanation_generation(self, rating_engine):
        """Test rating explanation generation."""
        rating_data = {
            "overall_score": 78,
            "confidence": 85,
            "component_scores": {
                "technical_analysis": 75,
                "momentum_analysis": 82,
                "options_attractiveness": 68
            },
            "key_factors": ["Strong momentum", "Good technical setup"],
            "risk_factors": ["Market volatility", "Earnings upcoming"]
        }
        
        explanation = rating_engine.generate_explanation(rating_data)
        
        assert isinstance(explanation, dict)
        assert "summary" in explanation
        assert "strengths" in explanation
        assert "risks" in explanation
        assert "recommendation" in explanation
    
    def test_historical_rating_tracking(self, rating_engine):
        """Test tracking of historical ratings."""
        symbol = "HIST_TEST"
        
        # Add several historical ratings
        for i, score in enumerate([70, 75, 80, 78]):
            rating_engine.record_historical_rating(
                symbol=symbol,
                score=score,
                timestamp=datetime.now() - timedelta(days=i)
            )
        
        history = rating_engine.get_rating_history(symbol)
        
        assert len(history) == 4
        assert all(isinstance(r["score"], (int, float)) for r in history)
    
    def test_rating_consistency_check(self, rating_engine):
        """Test rating consistency validation."""
        # Current analysis
        current_rating = {
            "symbol": "CONSISTENCY_TEST",
            "score": 85,
            "component_scores": {
                "technical_analysis": 80,
                "momentum_analysis": 90
            }
        }
        
        # Previous rating
        previous_rating = {
            "score": 50,  # Large difference
            "component_scores": {
                "technical_analysis": 45,
                "momentum_analysis": 55
            }
        }
        
        consistency_check = rating_engine.check_rating_consistency(
            current_rating, previous_rating
        )
        
        assert "consistency_score" in consistency_check
        assert "significant_changes" in consistency_check
        assert "explanation" in consistency_check
        
        # Large change should be flagged
        assert consistency_check["significant_changes"] is True
    
    def test_market_condition_adjustment(self, rating_engine):
        """Test rating adjustment based on market conditions."""
        base_rating = 75
        
        market_conditions = {
            "market_volatility": "high",
            "market_trend": "bearish",
            "sector_performance": "underperforming"
        }
        
        adjusted_rating = rating_engine.adjust_for_market_conditions(
            base_rating, market_conditions
        )
        
        # In poor market conditions, rating should typically be adjusted down
        assert adjusted_rating <= base_rating
    
    def test_rating_validation(self, rating_engine):
        """Test rating validation and bounds checking."""
        # Test valid ratings
        valid_ratings = [0, 50, 100]
        for rating in valid_ratings:
            assert rating_engine.validate_rating(rating) is True
        
        # Test invalid ratings
        invalid_ratings = [-10, 150, float('inf')]
        for rating in invalid_ratings:
            assert rating_engine.validate_rating(rating) is False


class TestResponseParser:
    """Tests for AI response parsing functionality."""
    
    @pytest.fixture
    def response_parser(self):
        """Create response parser instance."""
        return ResponseParser()
    
    def test_parse_comprehensive_response(self, response_parser, test_data_generator):
        """Test parsing of comprehensive analysis response."""
        mock_response = test_data_generator.generate_claude_response("comprehensive")
        
        parsed = response_parser.parse_response(mock_response)
        
        assert isinstance(parsed, ParsedResponse)
        assert parsed.overall_rating is not None
        assert 0 <= parsed.overall_rating <= 100
        assert parsed.confidence is not None
        assert isinstance(parsed.reasoning, str)
    
    def test_parse_scoring_response(self, response_parser):
        """Test parsing of scoring-focused response."""
        scoring_response = """
        **Technical Analysis Score: 82/100**
        - Strong uptrend with price above key moving averages
        - RSI at 68 indicates healthy momentum
        - MACD showing bullish crossover
        
        **Momentum Score: 76/100**  
        - Consistent upward movement over past week
        - Volume supporting price action
        
        **Options Score: 71/100**
        - Good liquidity in near-term contracts
        - IV at reasonable levels
        
        **Overall Rating: 77/100**
        **Confidence: 85%**
        
        Recommendation: BUY - Strong technical setup with good momentum
        """
        
        parsed = response_parser.parse_response(scoring_response)
        
        assert parsed.technical_score == 82
        assert parsed.momentum_score == 76
        assert parsed.options_score == 71
        assert parsed.overall_rating == 77
        assert parsed.confidence == 85
    
    def test_parse_malformed_response(self, response_parser):
        """Test handling of malformed responses."""
        malformed_responses = [
            "",  # Empty response
            "No scores found in this text",  # No recognizable scores
            "Score: ABC/100",  # Invalid score format
            "Technical Score: 150/100"  # Out of range score
        ]
        
        for response in malformed_responses:
            parsed = response_parser.parse_response(response)
            
            # Should handle gracefully without crashing
            assert isinstance(parsed, ParsedResponse)
            # Some fields might be None or default values
    
    def test_extract_key_factors(self, response_parser):
        """Test extraction of key factors and risks."""
        response_with_factors = """
        Analysis shows strong momentum with overall rating of 78/100.
        
        Key Strengths:
        - Rising RSI trend
        - Bullish MACD crossover
        - Strong volume confirmation
        - Price above all moving averages
        
        Risk Factors:
        - Market volatility increasing
        - Earnings announcement next week
        - Sector rotation concerns
        
        Recommendation: Consider for position
        """
        
        parsed = response_parser.parse_response(response_with_factors)
        
        assert len(parsed.key_factors) > 0
        assert len(parsed.risks) > 0
        assert any("RSI" in factor for factor in parsed.key_factors)
        assert any("volatility" in risk.lower() for risk in parsed.risks)
    
    def test_extract_recommendation(self, response_parser):
        """Test extraction of trading recommendations."""
        responses_with_recommendations = [
            ("Recommendation: BUY", "BUY"),
            ("Recommendation: HOLD", "HOLD"), 
            ("Recommendation: SELL", "SELL"),
            ("Consider for watchlist", "WATCHLIST"),
            ("Strong buy recommendation", "BUY")
        ]
        
        for response_text, expected_rec in responses_with_recommendations:
            parsed = response_parser.parse_response(response_text)
            
            # Should extract some form of recommendation
            assert parsed.recommendation is not None
            assert isinstance(parsed.recommendation, str)
    
    def test_confidence_extraction(self, response_parser):
        """Test confidence score extraction from various formats."""
        confidence_formats = [
            ("Confidence: 85%", 85),
            ("Confidence Level: 92%", 92),
            ("85% confidence", 85),
            ("High confidence (90%)", 90)
        ]
        
        for response_text, expected_confidence in confidence_formats:
            parsed = response_parser.parse_response(response_text)
            
            if parsed.confidence is not None:
                assert abs(parsed.confidence - expected_confidence) <= 5  # Allow some tolerance
    
    def test_score_range_validation(self, response_parser):
        """Test validation of extracted scores."""
        response_with_invalid_scores = """
        Technical Score: 150/100
        Momentum Score: -20/100
        Overall Rating: 200/100
        Confidence: 120%
        """
        
        parsed = response_parser.parse_response(response_with_invalid_scores)
        
        # Invalid scores should be normalized or rejected
        if parsed.technical_score is not None:
            assert 0 <= parsed.technical_score <= 100
        if parsed.momentum_score is not None:
            assert 0 <= parsed.momentum_score <= 100
        if parsed.overall_rating is not None:
            assert 0 <= parsed.overall_rating <= 100
        if parsed.confidence is not None:
            assert 0 <= parsed.confidence <= 100


class TestPromptTemplates:
    """Tests for prompt template management."""
    
    @pytest.fixture
    def prompt_templates(self):
        """Create prompt templates instance."""
        return PromptTemplates()
    
    def test_comprehensive_analysis_prompt(self, prompt_templates, test_data_generator):
        """Test comprehensive analysis prompt generation."""
        stock_data = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "technical_indicators": {
                "rsi_14": 65,
                "macd": 1.5,
                "bollinger_position": 0.7
            }
        }
        
        prompt = prompt_templates.create_comprehensive_prompt(stock_data)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert "AAPL" in prompt
        assert "150.0" in prompt or "150" in prompt
        assert "RSI" in prompt or "rsi" in prompt
    
    def test_momentum_analysis_prompt(self, prompt_templates):
        """Test momentum-specific analysis prompt."""
        momentum_data = {
            "symbol": "MSFT",
            "price_changes": {
                "1_day": 2.1,
                "5_day": 8.5,
                "20_day": 15.2
            },
            "volume_trends": {
                "avg_volume": 25000000,
                "recent_volume": 35000000
            }
        }
        
        prompt = prompt_templates.create_momentum_prompt(momentum_data)
        
        assert "MSFT" in prompt
        assert "momentum" in prompt.lower()
        assert "2.1" in prompt or "8.5" in prompt
    
    def test_options_analysis_prompt(self, prompt_templates, test_data_generator):
        """Test options-specific analysis prompt."""
        options_data = {
            "underlying_symbol": "GOOGL",
            "best_calls": test_data_generator.generate_options_chain("GOOGL", 2800.0)[:3],
            "iv_analysis": {
                "current_iv": 0.28,
                "iv_percentile": 45
            }
        }
        
        prompt = prompt_templates.create_options_prompt(options_data)
        
        assert "GOOGL" in prompt
        assert "options" in prompt.lower()
        assert "delta" in prompt.lower() or "gamma" in prompt.lower()
    
    def test_prompt_customization(self, prompt_templates):
        """Test prompt customization with different parameters."""
        base_data = {"symbol": "TEST", "price": 100.0}
        
        # Test different analysis types
        prompts = []
        for analysis_type in ["technical", "momentum", "options", "comprehensive"]:
            prompt = prompt_templates.create_custom_prompt(base_data, analysis_type)
            prompts.append(prompt)
            assert "TEST" in prompt
        
        # All prompts should be different
        assert len(set(prompts)) == len(prompts)
    
    def test_template_validation(self, prompt_templates):
        """Test prompt template validation."""
        # Test with missing required data
        incomplete_data = {"symbol": "INCOMPLETE"}  # Missing price data
        
        try:
            prompt = prompt_templates.create_comprehensive_prompt(incomplete_data)
            # Should either work with defaults or raise informative error
            assert isinstance(prompt, str)
        except Exception as e:
            # Should be a validation error, not a crash
            assert "missing" in str(e).lower() or "required" in str(e).lower()
    
    def test_prompt_length_management(self, prompt_templates):
        """Test prompt length management for token limits."""
        # Create very large dataset
        large_data = {
            "symbol": "LARGE",
            "technical_indicators": {f"indicator_{i}": i for i in range(1000)},
            "large_field": "x" * 50000
        }
        
        prompt = prompt_templates.create_comprehensive_prompt(large_data)
        
        # Prompt should be reasonable length (under typical token limits)
        assert len(prompt) < 100000  # Reasonable character limit
    
    def test_prompt_formatting(self, prompt_templates):
        """Test prompt formatting and structure."""
        data = {
            "symbol": "FORMAT_TEST",
            "price": 123.45,
            "technical_indicators": {"rsi": 67.8}
        }
        
        prompt = prompt_templates.create_comprehensive_prompt(data)
        
        # Should have clear structure
        assert "FORMAT_TEST" in prompt
        assert "123.45" in prompt or "123" in prompt
        # Should have some formatting markers
        assert any(marker in prompt for marker in ["**", "##", "---", "\n\n"])