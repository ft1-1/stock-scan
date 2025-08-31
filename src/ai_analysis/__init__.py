"""
AI Analysis Module for Options Screening

Complete Claude AI integration with 0-100 rating system for options screening.

Core Components:
- DataPackager: Combines multiple data sources into comprehensive JSON packages
- ClaudeClient: API integration with rate limiting and cost controls  
- PromptTemplates: Structured prompts with detailed scoring rubric
- ResponseParser: JSON validation and fallback handling
- CostManager: Token counting and daily spending limits
- Persistence: Complete audit trail with request/response storage
- RatingEngine: End-to-end workflow orchestration

Key Features:
- 60-second rate limiting between requests
- $50/day cost limits with warnings
- Comprehensive data quality assessment
- Structured 0-100 scoring (Trend 35%, Options 20%, IV 15%, etc.)
- Complete audit trail and error recovery
- Mock mode for development and testing

Usage:
    from src.ai_analysis import create_rating_engine
    
    engine = create_rating_engine(
        claude_api_key="your-api-key",
        daily_limit=50.0
    )
    
    result = await engine.analyze_single_opportunity(
        opportunity_data, option_data, enhanced_data
    )
"""

# Import main classes
from .data_packager import DataPackager, DataCompletenessMetrics
from .claude_client import ClaudeClient, ClaudeConfig, ClaudeResponse, create_claude_client
from .prompt_templates import PromptTemplates, RatingCriteria, create_analysis_prompt
from .response_parser import (
    ResponseParser, ClaudeAnalysis, ComponentScores, OptionRecommendation,
    ConfidenceLevel, parse_claude_response
)
from .cost_manager import CostManager, CostConfig, TokenCost, DailyUsage, create_cost_manager
from .persistence import AnalysisPersistence, AnalysisRecord, BatchAnalysisSession, create_persistence_manager
from .rating_engine import RatingEngine, RatingConfig, AnalysisResult, create_rating_engine

# Version information
__version__ = "1.0.0"
__author__ = "AI Integration Architect"

# Export main factory functions for easy access
__all__ = [
    # Main factory functions
    "create_rating_engine",
    "create_claude_client", 
    "create_cost_manager",
    "create_persistence_manager",
    "create_analysis_prompt",
    "parse_claude_response",
    
    # Core classes
    "RatingEngine",
    "DataPackager",
    "ClaudeClient", 
    "PromptTemplates",
    "ResponseParser",
    "CostManager",
    "AnalysisPersistence",
    
    # Configuration classes
    "RatingConfig",
    "ClaudeConfig",
    "CostConfig",
    "RatingCriteria",
    
    # Data classes
    "AnalysisResult",
    "ClaudeAnalysis",
    "ClaudeResponse",
    "ComponentScores",
    "OptionRecommendation",
    "TokenCost",
    "DailyUsage",
    "AnalysisRecord",
    "BatchAnalysisSession",
    "DataCompletenessMetrics",
    
    # Enums
    "ConfidenceLevel",
]

# Module-level constants
DEFAULT_DAILY_LIMIT = 50.0
DEFAULT_RATE_LIMIT_SECONDS = 60.0
DEFAULT_MIN_DATA_COMPLETENESS = 60.0

# Rating system breakdown
RATING_BREAKDOWN = {
    "trend_momentum": 35,      # Price trend strength, momentum acceleration, relative strength
    "options_quality": 20,     # Liquidity/spread quality, Greeks profile optimization  
    "iv_value": 15,           # IV percentile assessment, IV/HV relationship
    "squeeze_volatility": 10,  # Volatility compression, breakout probability
    "fundamentals": 10,        # Financial strength, growth trajectory
    "event_news": 10          # Earnings safety, news sentiment
}


def get_module_info() -> dict:
    """Get information about the AI analysis module"""
    return {
        "version": __version__,
        "components": len(__all__),
        "rating_breakdown": RATING_BREAKDOWN,
        "total_rating_points": sum(RATING_BREAKDOWN.values()),
        "default_settings": {
            "daily_limit": DEFAULT_DAILY_LIMIT,
            "rate_limit_seconds": DEFAULT_RATE_LIMIT_SECONDS,
            "min_data_completeness": DEFAULT_MIN_DATA_COMPLETENESS
        }
    }


def validate_rating_breakdown():
    """Validate that rating breakdown sums to 100"""
    total = sum(RATING_BREAKDOWN.values())
    if total != 100:
        raise ValueError(f"Rating breakdown must sum to 100, got {total}")
    return True


# Validate on import
validate_rating_breakdown()