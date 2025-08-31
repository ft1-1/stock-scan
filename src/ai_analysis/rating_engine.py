"""
Rating Engine for Options Screening AI Integration

Orchestrates the complete workflow:
1. Data package creation from multiple sources
2. Cost validation and rate limiting
3. Claude AI analysis with 60-second intervals
4. Response parsing and validation
5. Final scoring and ranking
6. Comprehensive persistence and audit trails

Designed for one-opportunity-at-a-time processing with proper rate limiting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from .data_packager import DataPackager, DataCompletenessMetrics
from .claude_client import ClaudeClient, ClaudeConfig, create_claude_client, ClaudeResponse
from .prompt_templates import PromptTemplates, create_analysis_prompt
from .response_parser import ResponseParser, ClaudeAnalysis, parse_claude_response
from .cost_manager import CostManager, create_cost_manager, TokenCost
from .persistence import AnalysisPersistence, create_persistence_manager

logger = logging.getLogger(__name__)


@dataclass
class RatingConfig:
    """Configuration for the rating engine"""
    # Claude AI settings
    claude_api_key: str
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_daily_limit: float = 50.0
    
    # Data quality settings
    min_data_completeness: float = 60.0
    require_option_data: bool = True
    require_enhanced_data: bool = False
    
    # Rate limiting
    min_request_interval: float = 60.0  # 60 seconds between requests
    max_concurrent_requests: int = 1    # Process one at a time
    
    # Cost controls
    max_cost_per_request: float = 1.0   # $1 max per analysis
    enable_cost_warnings: bool = True
    
    # Persistence settings
    storage_path: Optional[Path] = None
    compress_storage: bool = True
    
    # Quality controls
    enable_response_validation: bool = True
    enable_score_correction: bool = True
    fallback_on_errors: bool = True
    
    # Mock mode for development
    use_mock_client: bool = False


@dataclass
class AnalysisResult:
    """Complete result of AI analysis for one opportunity"""
    # Basic info
    symbol: str
    success: bool
    
    # Analysis data
    ai_analysis: Optional[ClaudeAnalysis] = None
    original_score: Optional[float] = None
    combined_score: Optional[float] = None
    
    # Processing metadata
    data_completeness: Optional[float] = None
    processing_time: Optional[float] = None
    cost: Optional[float] = None
    tokens_used: Optional[int] = None
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    
    # Persistence
    record_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        result = {
            'symbol': self.symbol,
            'success': self.success,
            'combined_score': self.combined_score,
            'data_completeness': self.data_completeness,
            'processing_time': self.processing_time,
            'cost': self.cost,
            'tokens_used': self.tokens_used,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'fallback_used': self.fallback_used,
            'record_id': self.record_id
        }
        
        if self.ai_analysis:
            result['ai_analysis'] = self.ai_analysis.to_dict()
        
        return result


class RatingEngine:
    """
    Complete AI rating engine for options screening.
    
    Provides end-to-end workflow from data packaging through final scoring,
    with comprehensive error handling, cost controls, and audit trails.
    """
    
    def __init__(self, config: RatingConfig):
        """
        Initialize rating engine with configuration.
        
        Args:
            config: Rating engine configuration
        """
        self.config = config
        
        # Initialize components
        self.data_packager = DataPackager(config.min_data_completeness)
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        
        # Initialize Claude client
        claude_config = ClaudeConfig(
            api_key=config.claude_api_key,
            model=config.claude_model,
            daily_cost_limit=config.claude_daily_limit,
            min_request_interval=config.min_request_interval
        )
        
        self.claude_client = create_claude_client(
            api_key=config.claude_api_key,
            daily_cost_limit=config.claude_daily_limit,
            use_mock=config.use_mock_client
        )
        
        # Initialize cost manager
        self.cost_manager = create_cost_manager(
            daily_limit=config.claude_daily_limit,
            storage_path=config.storage_path / "cost_tracking" if config.storage_path else None
        )
        
        # Initialize persistence
        self.persistence = create_persistence_manager(
            storage_path=config.storage_path,
            compress_data=config.compress_storage
        )
        
        logger.info("Rating engine initialized successfully")
    
    async def analyze_single_opportunity(
        self,
        opportunity_data: Dict[str, Any],
        option_chain_data: Dict[str, Any],
        enhanced_eodhd_data: Dict[str, Any],
        analysis_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a single opportunity with full AI workflow.
        
        Args:
            opportunity_data: Locally calculated opportunity metrics
            option_chain_data: Selected option contract data
            enhanced_eodhd_data: EODHD enhanced data
            analysis_context: Optional additional context
            
        Returns:
            Complete analysis result
        """
        symbol = opportunity_data.get('symbol', 'UNKNOWN')
        start_time = datetime.now()
        
        logger.info(f"Starting AI analysis for {symbol}")
        
        try:
            # Step 1: Create comprehensive data package
            data_package = self.data_packager.create_analysis_package(
                opportunity_data, option_chain_data, enhanced_eodhd_data
            )
            
            data_completeness = data_package.get('data_quality', {}).get('overall_score', 0.0)
            
            # Check data quality requirements
            if data_completeness < self.config.min_data_completeness:
                logger.warning(f"Data completeness {data_completeness:.1f}% below threshold for {symbol}")
                if not self.config.fallback_on_errors:
                    return AnalysisResult(
                        symbol=symbol,
                        success=False,
                        data_completeness=data_completeness,
                        error_type="InsufficientData",
                        error_message=f"Data completeness {data_completeness:.1f}% below required {self.config.min_data_completeness}%"
                    )
            
            # Step 2: Generate analysis prompt
            prompt = self.prompt_templates.create_analysis_prompt(data_package, analysis_context)
            
            # Step 3: Cost validation
            estimated_cost = self.cost_manager.estimate_request_cost(prompt)
            can_afford, afford_reason, remaining_budget = self.cost_manager.can_afford_request(
                estimated_cost.total_cost
            )
            
            if not can_afford:
                logger.error(f"Cannot afford analysis for {symbol}: {afford_reason}")
                return AnalysisResult(
                    symbol=symbol,
                    success=False,
                    data_completeness=data_completeness,
                    cost=estimated_cost.total_cost,
                    error_type="CostLimit",
                    error_message=afford_reason
                )
            
            if estimated_cost.total_cost > self.config.max_cost_per_request:
                logger.error(f"Estimated cost ${estimated_cost.total_cost:.4f} exceeds per-request limit ${self.config.max_cost_per_request}")
                return AnalysisResult(
                    symbol=symbol,
                    success=False,
                    data_completeness=data_completeness,
                    cost=estimated_cost.total_cost,
                    error_type="CostLimit",
                    error_message=f"Request cost exceeds limit"
                )
            
            # Step 4: Claude AI analysis (with rate limiting)
            claude_response = await self.claude_client.analyze_opportunity(prompt, symbol)
            
            # Record actual cost
            if claude_response.tokens_used:
                actual_cost = self.cost_manager.record_actual_usage(
                    prompt=prompt,
                    response=claude_response.content or "",
                    symbol=symbol
                )
            else:
                actual_cost = estimated_cost
            
            # Step 5: Parse and validate response
            original_score = opportunity_data.get('overall_score')
            
            if claude_response.success and claude_response.parsed_json:
                # Parse successful response
                parsed_analysis = self.response_parser.parse_response(
                    claude_response.content,
                    symbol,
                    original_score
                )
                
                # Apply score corrections if enabled
                if self.config.enable_score_correction:
                    parsed_analysis = self.response_parser.validate_and_correct_scores(parsed_analysis)
                
            else:
                # Handle failed response
                if self.config.fallback_on_errors:
                    logger.warning(f"Claude analysis failed for {symbol}, using fallback: {claude_response.error}")
                    parsed_analysis = self.response_parser.parse_response(
                        "", symbol, original_score  # Empty response triggers fallback
                    )
                    parsed_analysis.notes = f"Fallback analysis. Error: {claude_response.error}"
                else:
                    return AnalysisResult(
                        symbol=symbol,
                        success=False,
                        data_completeness=data_completeness,
                        cost=actual_cost.total_cost,
                        tokens_used=actual_cost.total_tokens,
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        error_type="AnalysisFailure",
                        error_message=claude_response.error
                    )
            
            # Step 6: Calculate combined score (if original score available)
            combined_score = self._calculate_combined_score(
                original_score, parsed_analysis.rating
            )
            
            # Step 7: Save comprehensive record
            error_info = None
            if not claude_response.success:
                error_info = ("ClaudeAPIError", claude_response.error)
            
            record_id = self.persistence.save_analysis_record(
                symbol=symbol,
                data_package=data_package,
                prompt=prompt,
                raw_response=claude_response.content,
                parsed_analysis=parsed_analysis,
                token_cost=actual_cost,
                error_info=error_info,
                performance_metadata={
                    'response_time_seconds': claude_response.response_time,
                    'claude_model': self.config.claude_model,
                    'rate_limit_wait_seconds': 0.0  # Could track this from client
                }
            )
            
            # Create successful result
            result = AnalysisResult(
                symbol=symbol,
                success=True,
                ai_analysis=parsed_analysis,
                original_score=original_score,
                combined_score=combined_score,
                data_completeness=data_completeness,
                processing_time=(datetime.now() - start_time).total_seconds(),
                cost=actual_cost.total_cost,
                tokens_used=actual_cost.total_tokens,
                record_id=record_id,
                fallback_used=not parsed_analysis.is_valid
            )
            
            logger.info(f"Successfully analyzed {symbol}: rating={parsed_analysis.rating}, "
                       f"cost=${actual_cost.total_cost:.4f}, time={result.processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error analyzing {symbol}: {e}")
            
            # Save error record
            try:
                self.persistence.save_analysis_record(
                    symbol=symbol,
                    data_package=data_package if 'data_package' in locals() else {},
                    prompt=prompt if 'prompt' in locals() else "",
                    error_info=("UnexpectedError", str(e))
                )
            except:
                pass  # Don't fail on persistence errors
            
            return AnalysisResult(
                symbol=symbol,
                success=False,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error_type="UnexpectedError",
                error_message=str(e)
            )
    
    async def analyze_opportunity_batch(
        self,
        opportunities: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
        analysis_context: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[AnalysisResult]:
        """
        Analyze a batch of opportunities with proper rate limiting.
        
        Args:
            opportunities: List of (opportunity_data, option_data, enhanced_data) tuples
            analysis_context: Optional context for all analyses
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results
        """
        symbols = [opp[0].get('symbol', f'Unknown_{i}') for i, opp in enumerate(opportunities)]
        
        logger.info(f"Starting batch analysis of {len(opportunities)} opportunities")
        
        # Start batch session
        session_id = self.persistence.start_batch_session(symbols)
        
        # Pre-flight cost check
        sample_prompts = []
        for opportunity_data, option_data, enhanced_data in opportunities[:3]:  # Sample first 3
            try:
                package = self.data_packager.create_analysis_package(
                    opportunity_data, option_data, enhanced_data
                )
                prompt = self.prompt_templates.create_analysis_prompt(package)
                sample_prompts.append(prompt)
            except:
                continue
        
        if sample_prompts:
            batch_estimate = self.cost_manager.estimate_batch_cost(sample_prompts * (len(opportunities) // len(sample_prompts) + 1))
            logger.info(f"Estimated batch cost: ${batch_estimate['total_estimated_cost']:.2f}")
            
            if not batch_estimate['can_afford_all']:
                logger.warning(f"Batch cost exceeds budget: {batch_estimate['reason']}")
                for rec in batch_estimate['recommendations']:
                    logger.info(f"  {rec}")
        
        # Process opportunities one at a time with rate limiting
        results = []
        
        for i, (opportunity_data, option_data, enhanced_data) in enumerate(opportunities):
            # Progress callback
            if progress_callback:
                progress_callback(i, len(opportunities), opportunity_data.get('symbol', f'Unknown_{i}'))
            
            # Analyze single opportunity
            result = await self.analyze_single_opportunity(
                opportunity_data, option_data, enhanced_data, analysis_context
            )
            results.append(result)
            
            # Rate limiting (except for last request)
            if i < len(opportunities) - 1:
                logger.debug(f"Rate limiting: waiting {self.config.min_request_interval} seconds")
                await asyncio.sleep(self.config.min_request_interval)
        
        # Finish batch session
        session_summary = self.persistence.finish_batch_session()
        
        successful_count = sum(1 for r in results if r.success)
        total_cost = sum(r.cost or 0 for r in results)
        
        logger.info(f"Batch analysis completed: {successful_count}/{len(opportunities)} successful, "
                   f"total cost: ${total_cost:.4f}")
        
        return results
    
    def _calculate_combined_score(
        self,
        original_score: Optional[float],
        ai_score: int
    ) -> Optional[float]:
        """
        Calculate combined score from original quantitative score and AI rating.
        
        Uses 60% quantitative, 40% AI weighting as specified.
        """
        if original_score is None:
            return float(ai_score)
        
        # Ensure scores are in 0-100 range
        original_normalized = max(0, min(100, original_score))
        ai_normalized = max(0, min(100, ai_score))
        
        # 60% quantitative, 40% AI
        combined = (original_normalized * 0.6) + (ai_normalized * 0.4)
        
        return round(combined, 2)
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage and performance statistics"""
        cost_stats = self.cost_manager.get_usage_summary(7)  # Last 7 days
        claude_stats = self.claude_client.get_usage_stats()
        daily_summary = self.persistence.get_daily_summary()
        
        return {
            'cost_management': cost_stats,
            'claude_api': claude_stats,
            'daily_analysis': daily_summary,
            'configuration': {
                'daily_limit': self.config.claude_daily_limit,
                'min_data_completeness': self.config.min_data_completeness,
                'rate_limit_seconds': self.config.min_request_interval,
                'mock_mode': self.config.use_mock_client
            }
        }
    
    def can_analyze_now(self) -> Tuple[bool, str]:
        """
        Check if analysis can be performed right now.
        
        Returns:
            (can_analyze, reason_if_not)
        """
        # Check Claude client rate limits
        can_request, claude_reason = self.claude_client.can_make_request()
        if not can_request:
            return False, f"Claude rate limit: {claude_reason}"
        
        # Check cost limits
        daily_usage = self.cost_manager.get_daily_usage()
        remaining_budget = self.config.claude_daily_limit - daily_usage.total_cost
        
        if remaining_budget < 0.10:  # $0.10 minimum buffer
            return False, f"Daily budget nearly exhausted (${remaining_budget:.4f} remaining)"
        
        return True, "Ready for analysis"


def create_rating_engine(
    claude_api_key: str,
    daily_limit: float = 50.0,
    storage_path: Optional[Path] = None,
    use_mock: bool = False
) -> RatingEngine:
    """
    Factory function to create rating engine with sensible defaults.
    
    Args:
        claude_api_key: Anthropic API key
        daily_limit: Daily spending limit
        storage_path: Storage directory for persistence
        use_mock: Use mock Claude client for testing
        
    Returns:
        Configured RatingEngine instance
    """
    config = RatingConfig(
        claude_api_key=claude_api_key,
        claude_daily_limit=daily_limit,
        storage_path=storage_path,
        use_mock_client=use_mock
    )
    
    return RatingEngine(config)


async def main():
    """Example usage and testing"""
    # Create rating engine
    engine = create_rating_engine(
        claude_api_key="test-key",
        daily_limit=10.0,
        use_mock=True
    )
    
    # Check if ready
    can_analyze, reason = engine.can_analyze_now()
    print(f"Ready to analyze: {can_analyze} ({reason})")
    
    # Sample opportunity data
    sample_opportunity = {
        "symbol": "AAPL",
        "current_price": 175.50,
        "overall_score": 82.5,
        "technical_indicators": {"rsi": 58.2, "macd": 1.25},
        "momentum_analysis": {"21_day_return": 0.045}
    }
    
    sample_option = {
        "option_symbol": "AAPL240315C00170000",
        "strike": 170.0,
        "bid": 8.50,
        "ask": 8.75,
        "delta": 0.65,
        "implied_volatility": 0.28,
        "volume": 1000,
        "open_interest": 5000
    }
    
    sample_enhanced = {
        "fundamentals": {
            "company_info": {"name": "Apple Inc.", "sector": "Technology"},
            "financial_health": {"eps_ttm": 5.95, "profit_margin": 0.253}
        },
        "news": [{"title": "Apple announces new product", "date": "2024-01-15"}]
    }
    
    # Analyze single opportunity
    print("\nAnalyzing single opportunity...")
    result = await engine.analyze_single_opportunity(
        sample_opportunity, sample_option, sample_enhanced
    )
    
    print(f"Analysis result for {result.symbol}:")
    print(f"  Success: {result.success}")
    print(f"  AI Rating: {result.ai_analysis.rating if result.ai_analysis else 'N/A'}")
    print(f"  Combined Score: {result.combined_score}")
    print(f"  Cost: ${result.cost:.4f}")
    print(f"  Processing Time: {result.processing_time:.1f}s")
    
    if result.error_message:
        print(f"  Error: {result.error_message}")
    
    # Get usage statistics
    stats = engine.get_usage_statistics()
    print(f"\nUsage Statistics:")
    print(f"  Today's Cost: ${stats['cost_management']['today']['cost']}")
    print(f"  Today's Requests: {stats['claude_api']['requests_today']}")
    print(f"  Total Analyses: {stats['daily_analysis']['total_analyses']}")


if __name__ == "__main__":
    asyncio.run(main())