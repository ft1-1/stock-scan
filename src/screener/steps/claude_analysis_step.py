"""Claude Analysis step executor for AI-powered opportunity evaluation.

This module implements the ClaudeAnalysisExecutor that processes ONLY top-ranked
opportunities with comprehensive JSON packages for Claude AI analysis.

Integration points:
- DataEnrichmentExecutor provides enhanced data (Phase 1)
- TechnicalAnalysisExecutor calculates scores (Phase 2)  
- LocalRankingExecutor filters to top N opportunities (Phase 2)
- ClaudeClient provides rate limiting and cost controls
- Individual JSON persistence per opportunity
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

from src.screener.workflow_engine import WorkflowStepExecutor
from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStepResult,
    WorkflowExecutionContext
)
from src.ai_analysis.claude_client import ClaudeClient, ClaudeConfig, create_claude_client
from src.ai_analysis.data_packager import DataPackager, estimate_token_count
from src.ai_analysis.prompt_templates import create_analysis_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClaudeAnalysisExecutor(WorkflowStepExecutor):
    """
    Claude Analysis executor that:
    1. Processes ONLY top-ranked opportunities (proceed_to_ai: True)
    2. Creates comprehensive JSON packages with enhanced data
    3. Uses existing ClaudeClient with 60-second rate limiting
    4. Saves individual JSON files per opportunity with full context
    5. Tracks costs and enforces daily limits
    """
    
    def __init__(self, step: WorkflowStep):
        super().__init__(step)
        self.claude_client: Optional[ClaudeClient] = None
        self.data_packager: Optional[DataPackager] = None
        self.output_directory: Optional[Path] = None
        
    async def execute_step(
        self, 
        input_data: Any, 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute Claude analysis for top-ranked opportunities only.
        
        Args:
            input_data: Output from LocalRankingExecutor with top opportunities
            context: Workflow execution context
            
        Returns:
            Dict containing AI analysis results and processing metadata
        """
        # Input validation
        if not isinstance(input_data, dict) or 'top_opportunities' not in input_data:
            raise ValueError("Input must contain 'top_opportunities' from LocalRankingExecutor")
            
        top_opportunities = input_data['top_opportunities']
        if not isinstance(top_opportunities, list):
            raise ValueError("Top opportunities must be a list")
        
        # Filter to only opportunities marked for AI analysis
        ai_opportunities = [opp for opp in top_opportunities if opp.get('proceed_to_ai') is True]
        
        if not ai_opportunities:
            logger.warning("No opportunities marked for AI analysis (proceed_to_ai: True)")
            return {
                'ai_analysis_results': {},
                'opportunities_analyzed': 0,
                'opportunities_failed': 0,
                'total_cost': 0.0,
                'analysis_summary': {
                    'no_opportunities_for_ai': True,
                    'reason': 'No opportunities marked with proceed_to_ai: True'
                }
            }
        
        logger.info(f"Starting Claude analysis for {len(ai_opportunities)} top-ranked opportunities")
        
        # Initialize components
        await self._initialize_claude_client()
        await self._initialize_data_packager()
        await self._setup_output_directory()
        
        # Process opportunities sequentially with rate limiting
        analysis_results = {}
        opportunities_analyzed = 0
        opportunities_failed = 0
        total_cost = 0.0
        
        for i, opportunity in enumerate(ai_opportunities, 1):
            symbol = opportunity.get('symbol', f'UNKNOWN_{i}')
            logger.info(f"Analyzing opportunity {i}/{len(ai_opportunities)}: {symbol}")
            
            try:
                # Check if we can make request (cost and rate limits)
                can_request, reason = self.claude_client.can_make_request()
                if not can_request:
                    # Check if it's a rate limit we can wait for
                    if "Rate limit:" in reason and "remaining" in reason:
                        # Extract wait time from reason string
                        import re
                        match = re.search(r'(\d+\.?\d*)s remaining', reason)
                        if match:
                            wait_time = float(match.group(1))
                            logger.info(f"Rate limited for {symbol}. Waiting {wait_time:.1f}s...")
                            await asyncio.sleep(wait_time + 0.5)  # Add small buffer
                            # Now we can proceed
                        else:
                            logger.warning(f"Cannot analyze {symbol}: {reason}")
                            opportunities_failed += 1
                            analysis_results[symbol] = {
                                'success': False,
                                'error': f"Rate/cost limit: {reason}",
                                'timestamp': datetime.now().isoformat()
                            }
                            continue
                    else:
                        # It's a cost limit or other issue - skip
                        logger.warning(f"Cannot analyze {symbol}: {reason}")
                        opportunities_failed += 1
                        analysis_results[symbol] = {
                            'success': False,
                            'error': f"Rate/cost limit: {reason}",
                            'timestamp': datetime.now().isoformat()
                        }
                        continue
                
                # Process single opportunity
                result = await self._analyze_single_opportunity(opportunity, context)
                
                if result['success']:
                    analysis_results[symbol] = result
                    opportunities_analyzed += 1
                    total_cost += result.get('cost', 0.0)
                    
                    # Save individual JSON file
                    await self._save_opportunity_analysis(symbol, result)
                    
                    logger.info(f"Successfully analyzed {symbol} "
                               f"(cost: ${result.get('cost', 0.0):.4f})")
                else:
                    analysis_results[symbol] = result
                    opportunities_failed += 1
                    logger.error(f"Failed to analyze {symbol}: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                opportunities_failed += 1
                analysis_results[symbol] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Update context
        context.completed_symbols += opportunities_analyzed
        context.failed_symbols += opportunities_failed
        context.total_cost += total_cost
        
        # Get final usage stats
        usage_stats = self.claude_client.get_usage_stats()
        
        analysis_summary = {
            'opportunities_analyzed': opportunities_analyzed,
            'opportunities_failed': opportunities_failed,
            'success_rate': (opportunities_analyzed / len(ai_opportunities) * 100) if ai_opportunities else 0,
            'total_cost': total_cost,
            'avg_cost_per_opportunity': total_cost / opportunities_analyzed if opportunities_analyzed > 0 else 0,
            'claude_usage_stats': usage_stats,
            'analysis_timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_directory)
        }
        
        logger.info(f"Claude analysis completed: {opportunities_analyzed} successful, "
                   f"{opportunities_failed} failed, total cost: ${total_cost:.4f}")
        
        return {
            'ai_analysis_results': analysis_results,
            'opportunities_analyzed': opportunities_analyzed,
            'opportunities_failed': opportunities_failed,
            'total_cost': total_cost,
            'analysis_summary': analysis_summary,
            'original_ranking_summary': input_data.get('ranking_summary', {})
        }
    
    async def _analyze_single_opportunity(
        self, 
        opportunity: Dict[str, Any], 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Analyze a single opportunity with Claude AI.
        
        Creates comprehensive data package and processes through Claude.
        """
        symbol = opportunity.get('symbol', 'UNKNOWN')
        
        try:
            # 1. Create comprehensive data package
            data_package = await self._create_data_package(opportunity)
            
            if not data_package:
                return {
                    'success': False,
                    'error': 'Failed to create data package',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            
            # 2. Generate Claude prompt
            prompt = self._create_claude_prompt(data_package)
            
            # 3. Estimate cost
            estimated_cost = self.claude_client.estimate_request_cost(prompt)
            logger.debug(f"Estimated cost for {symbol}: ${estimated_cost:.4f}")
            
            # 4. Submit to Claude with rate limiting
            claude_response = await self.claude_client.analyze_opportunity(
                prompt=prompt,
                opportunity_symbol=symbol
            )
            
            # 5. Process response
            if claude_response.success:
                # Parse and validate Claude response
                parsed_analysis = self._parse_claude_response(claude_response)
                
                # Create comprehensive result
                result = {
                    'success': True,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'data_package': data_package,
                    'claude_submission': {
                        'prompt': prompt,
                        'model': 'claude-3-5-sonnet',
                        'timestamp': claude_response.timestamp.isoformat() if claude_response.timestamp else None,
                        'estimated_tokens': estimate_token_count({'prompt': prompt})
                    },
                    'claude_response': {
                        'raw_response': claude_response.content,
                        'parsed_analysis': parsed_analysis,
                        'tokens_used': claude_response.tokens_used,
                        'cost': claude_response.cost_estimate,
                        'response_time': claude_response.response_time,
                        'success': claude_response.success
                    },
                    'cost': claude_response.cost_estimate or 0.0,
                    'processing_metadata': {
                        'data_completeness_score': data_package.get('data_quality', {}).get('overall_score', 0),
                        'prompt_length': len(prompt),
                        'response_length': len(claude_response.content) if claude_response.content else 0
                    }
                }
                
                return result
            else:
                return {
                    'success': False,
                    'symbol': symbol,
                    'error': claude_response.error,
                    'timestamp': datetime.now().isoformat(),
                    'cost': claude_response.cost_estimate or 0.0
                }
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _create_data_package(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create comprehensive data package for Claude analysis.
        
        Combines local analysis data with enhanced EODHD data.
        """
        try:
            symbol = opportunity.get('symbol')
            
            # Extract local analysis data (technical, momentum, squeeze, scoring)
            # Extract enhanced EODHD data first to use in opportunity_data
            enhanced_data = opportunity.get('enhanced_data', {})
            
            # Build technical indicators from enhanced data
            technical_indicators = {}
            
            # First, get from technical_indicators (EODHD technical endpoint)
            if enhanced_data.get('technical_indicators'):
                tech_ind = enhanced_data['technical_indicators']
                # Extract latest values from each indicator
                if tech_ind.get('rsi') and isinstance(tech_ind['rsi'], list) and len(tech_ind['rsi']) > 0:
                    technical_indicators['rsi'] = tech_ind['rsi'][0].get('rsi')
                if tech_ind.get('macd') and isinstance(tech_ind['macd'], list) and len(tech_ind['macd']) > 0:
                    technical_indicators['macd'] = tech_ind['macd'][0].get('macd')
                if tech_ind.get('sma_20') and isinstance(tech_ind['sma_20'], list) and len(tech_ind['sma_20']) > 0:
                    technical_indicators['sma_20'] = tech_ind['sma_20'][0].get('sma')
                if tech_ind.get('sma_50') and isinstance(tech_ind['sma_50'], list) and len(tech_ind['sma_50']) > 0:
                    technical_indicators['sma_50'] = tech_ind['sma_50'][0].get('sma')
                if tech_ind.get('bbands') and isinstance(tech_ind['bbands'], list) and len(tech_ind['bbands']) > 0:
                    technical_indicators['bollinger_bands'] = tech_ind['bbands'][0]
            
            # Also check 'technicals' field (from fundamentals or other sources)
            if enhanced_data.get('technicals'):
                tech = enhanced_data['technicals']
                # Override with any values from technicals
                technical_indicators.update({
                    'rsi': technical_indicators.get('rsi') or tech.get('rsi'),
                    'sma_20': technical_indicators.get('sma_20') or tech.get('sma_20'),
                    'sma_50': technical_indicators.get('sma_50') or tech.get('sma_50'),
                    'macd': technical_indicators.get('macd') or tech.get('macd'),
                    'bollinger_bands': technical_indicators.get('bollinger_bands') or tech.get('bollinger_bands')
                })
            
            # Also extract SMA values from fundamentals if available
            if enhanced_data.get('fundamentals') and enhanced_data['fundamentals'].get('stock_technicals'):
                stock_tech = enhanced_data['fundamentals']['stock_technicals']
                if stock_tech.get('50_day_ma'):
                    technical_indicators['sma_50'] = technical_indicators.get('sma_50') or stock_tech['50_day_ma']
                if stock_tech.get('200_day_ma'):
                    technical_indicators['sma_200'] = stock_tech['200_day_ma']
            
            # Get current price from multiple possible sources
            current_price = (
                opportunity.get('current_price') or 
                enhanced_data.get('quote', {}).get('close') or
                enhanced_data.get('quote', {}).get('last_price') or
                enhanced_data.get('live_price', {}).get('close') or
                enhanced_data.get('live_price', {}).get('price')
            )
            
            opportunity_data = {
                'symbol': symbol,
                'current_price': current_price,
                'overall_score': opportunity.get('composite_score'),
                'confidence_level': 'high' if opportunity.get('composite_score', 0) >= 80 else 'medium',
                'strategy_type': 'options_screening',
                'technical_indicators': technical_indicators,
                'momentum_analysis': opportunity.get('momentum_data', {}),
                'squeeze_analysis': opportunity.get('squeeze_data', {}),
                'score_breakdown': opportunity.get('score_breakdown', {}),
                'risk_metrics': {
                    'warnings': opportunity.get('warnings', []),
                    'ranking': opportunity.get('ranking'),
                    'percentile': opportunity.get('score_percentile')
                }
            }
            
            # Extract best option contract data
            option_data = opportunity.get('best_call', {})
            
            # If no best_call, try to get from options_chain in enhanced_data
            if not option_data and enhanced_data.get('options_chain'):
                logger.debug(f"No best_call found, checking options_chain with {len(enhanced_data['options_chain'])} contracts")
                # Get the first SHORT_CALL option (or any call option) as example
                for contract in enhanced_data['options_chain']:
                    # Check for our custom contract_type or standard option_type
                    contract_type = contract.get('contract_type', '').upper()
                    option_type = contract.get('option_type', '').lower()
                    type_field = contract.get('type', '').lower()
                    
                    # Accept SHORT_CALL, LEAP, or standard 'call' type
                    if contract_type in ['SHORT_CALL', 'LEAP'] or option_type == 'call' or type_field == 'call':
                        option_data = contract
                        logger.debug(f"Found option: type={contract_type}, strike={contract.get('strike')}, exp={contract.get('expiration')}")
                        break
                if not option_data:
                    logger.warning(f"No suitable options found in {len(enhanced_data['options_chain'])} contracts")
            
            if option_data:
                # Standardize option data structure
                option_contract_data = {
                    'option_symbol': option_data.get('symbol'),
                    'underlying_symbol': symbol,
                    'strike': option_data.get('strike'),
                    'expiration': option_data.get('expiration'),
                    'dte': option_data.get('dte'),
                    'option_type': 'call',
                    'bid': option_data.get('bid'),
                    'ask': option_data.get('ask'),
                    'last': option_data.get('last'),
                    'delta': option_data.get('delta'),
                    'gamma': option_data.get('gamma'),
                    'theta': option_data.get('theta'),
                    'vega': option_data.get('vega'),
                    'rho': option_data.get('rho'),
                    'implied_volatility': option_data.get('implied_volatility') or option_data.get('iv'),
                    'volume': option_data.get('volume'),
                    'open_interest': option_data.get('open_interest'),
                    'option_score': option_data.get('score'),
                    'selection_reason': f"Best call option (score: {option_data.get('score', 'N/A')})"
                }
            else:
                option_contract_data = {}
            
            # Debug: Log what we have
            if enhanced_data:
                logger.info(f"Enhanced data for {symbol} contains: {list(enhanced_data.keys())}")
                # Check sentiment format
                if 'sentiment' in enhanced_data and enhanced_data['sentiment']:
                    sentiment = enhanced_data['sentiment']
                    logger.debug(f"Sentiment data type: {type(sentiment)}, sample: {str(sentiment)[:100]}")
                # Check specific fields
                if 'fundamentals' in enhanced_data and enhanced_data['fundamentals']:
                    logger.info(f"Fundamentals present with {len(enhanced_data['fundamentals'])} fields")
                if 'news' in enhanced_data and enhanced_data['news']:
                    logger.info(f"News present with {len(enhanced_data['news'])} articles")
                if 'risk_metrics' in enhanced_data and enhanced_data['risk_metrics']:
                    logger.info(f"Risk metrics present: {list(enhanced_data['risk_metrics'].keys())}")
            else:
                logger.warning(f"No enhanced data found for {symbol}")
            
            # Create package using DataPackager
            package = self.data_packager.create_analysis_package(
                opportunity_data=opportunity_data,
                option_chain_data=option_contract_data,
                enhanced_eodhd_data=enhanced_data
            )
            
            logger.debug(f"Created data package for {symbol} with "
                        f"{package['data_quality']['overall_score']:.1f}% completeness")
            
            return package
            
        except Exception as e:
            logger.error(f"Error creating data package for {opportunity.get('symbol', 'UNKNOWN')}: {e}")
            return None
    
    def _create_claude_prompt(self, data_package: Dict[str, Any]) -> str:
        """Create Claude prompt from data package using prompt templates."""
        try:
            # Add analysis context based on data quality
            completeness_score = data_package.get('data_quality', {}).get('overall_score', 0)
            
            analysis_context = None
            if completeness_score < 70:
                analysis_context = (
                    f"Note: Data completeness is {completeness_score:.1f}%. "
                    f"Some analysis components may have limited data. "
                    f"Please adjust confidence levels accordingly and note any data gaps in your analysis."
                )
            
            prompt = create_analysis_prompt(
                data_package=data_package,
                analysis_context=analysis_context
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            raise
    
    def _parse_claude_response(self, claude_response) -> Dict[str, Any]:
        """Parse and validate Claude's JSON response."""
        try:
            if claude_response.parsed_json:
                # Response was successfully parsed as JSON
                analysis = claude_response.parsed_json
                
                # Validate required fields
                required_fields = ['symbol', 'rating', 'confidence', 'thesis']
                missing_fields = [field for field in required_fields if field not in analysis]
                
                if missing_fields:
                    logger.warning(f"Claude response missing fields: {missing_fields}")
                
                # Ensure rating is within bounds
                rating = analysis.get('rating', 0)
                if not isinstance(rating, (int, float)) or rating < 0 or rating > 100:
                    logger.warning(f"Invalid rating: {rating}, setting to 50")
                    analysis['rating'] = 50
                
                # Validate component scores sum up correctly
                component_scores = analysis.get('component_scores', {})
                if component_scores:
                    total_components = sum(component_scores.values())
                    if abs(total_components - rating) > 5:  # Allow small rounding differences
                        logger.warning(f"Component scores ({total_components}) don't match rating ({rating})")
                
                return analysis
            else:
                # Try to extract any usable information from raw response
                logger.warning("Failed to parse Claude response as JSON, extracting basic info")
                return {
                    'symbol': 'UNKNOWN',
                    'rating': 50,
                    'confidence': 'low',
                    'thesis': 'Failed to parse structured response',
                    'error': 'JSON parsing failed',
                    'raw_content': claude_response.content[:500] if claude_response.content else None
                }
                
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return {
                'symbol': 'UNKNOWN',
                'rating': 0,
                'confidence': 'low',
                'thesis': 'Error parsing response',
                'error': str(e)
            }
    
    async def _save_opportunity_analysis(self, symbol: str, analysis_result: Dict[str, Any]) -> None:
        """Save individual opportunity analysis to JSON file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timestamp}.json"
            filepath = self.output_directory / filename
            
            # Ensure directory exists
            self.output_directory.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, default=str, ensure_ascii=False)
            
            logger.debug(f"Saved analysis for {symbol} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving analysis for {symbol}: {e}")
    
    async def _initialize_claude_client(self) -> None:
        """Initialize Claude client with settings from workflow configuration."""
        if self.claude_client is not None:
            return
        
        try:
            # Get API key from settings
            api_key = getattr(self.settings, 'claude_api_key', None)
            if not api_key:
                # Try environment variable
                api_key = os.getenv('CLAUDE_API_KEY')
            
            if not api_key:
                raise ValueError("Claude API key not found in settings or environment")
            
            # Get daily cost limit from settings or use default
            daily_cost_limit = getattr(self.settings, 'claude_daily_cost_limit', 50.0)
            
            # Determine if we should use mock client
            use_mock = getattr(self.settings, 'claude_use_mock', False)
            
            self.claude_client = create_claude_client(
                api_key=api_key,
                daily_cost_limit=daily_cost_limit,
                use_mock=use_mock
            )
            
            logger.info("Claude client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    async def _initialize_data_packager(self) -> None:
        """Initialize data packager with quality threshold."""
        if self.data_packager is not None:
            return
        
        # Get minimum completeness threshold from settings or use default
        min_threshold = getattr(self.settings, 'claude_min_data_completeness', 60.0)
        
        self.data_packager = DataPackager(min_completeness_threshold=min_threshold)
        logger.info(f"Data packager initialized with {min_threshold}% completeness threshold")
    
    async def _setup_output_directory(self) -> None:
        """Setup output directory for individual analysis files."""
        try:
            # Create date-based directory structure
            today = datetime.now().strftime('%Y%m%d')
            base_output_dir = Path('data/ai_analysis')
            self.output_directory = base_output_dir / today
            
            # Create directory if it doesn't exist
            self.output_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Output directory setup: {self.output_directory}")
            
        except Exception as e:
            logger.error(f"Error setting up output directory: {e}")
            # Fallback to current directory
            self.output_directory = Path('.')
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data from LocalRankingExecutor."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'top_opportunities' not in input_data:
            raise ValueError("Input must contain 'top_opportunities' key")
        
        top_opportunities = input_data['top_opportunities']
        if not isinstance(top_opportunities, list):
            raise ValueError("Top opportunities must be a list")
        
        # Check for properly structured opportunities
        for i, opportunity in enumerate(top_opportunities[:3]):  # Check first 3
            if not isinstance(opportunity, dict):
                raise ValueError(f"Opportunity {i} must be a dictionary")
            
            required_fields = ['symbol', 'composite_score', 'ranking']
            missing_fields = [field for field in required_fields if field not in opportunity]
            if missing_fields:
                raise ValueError(f"Opportunity {i} missing fields: {missing_fields}")
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from Claude analysis."""
        if not isinstance(output_data, dict):
            raise ValueError("Output data must be a dictionary")
        
        required_keys = ['ai_analysis_results', 'opportunities_analyzed', 'opportunities_failed', 'total_cost']
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")
        
        # Validate analysis results structure
        analysis_results = output_data['ai_analysis_results']
        if not isinstance(analysis_results, dict):
            raise ValueError("AI analysis results must be a dictionary")
        
        # Check that successful analyses have proper structure
        for symbol, result in analysis_results.items():
            if result.get('success'):
                required_result_fields = ['symbol', 'timestamp', 'claude_response']
                missing_fields = [field for field in required_result_fields if field not in result]
                if missing_fields:
                    raise ValueError(f"Analysis result for {symbol} missing fields: {missing_fields}")
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, dict):
            return output_data.get('opportunities_analyzed', 0)
        return 0


# Convenience functions for external usage

def create_claude_analysis_executor(
    top_n_for_ai: Optional[int] = None,
    min_data_completeness: float = 60.0
) -> ClaudeAnalysisExecutor:
    """
    Factory function to create ClaudeAnalysisExecutor with common settings.
    
    Args:
        top_n_for_ai: Maximum number of opportunities to send to AI (uses config if None)
        min_data_completeness: Minimum data completeness score threshold
        
    Returns:
        Configured ClaudeAnalysisExecutor
    """
    from config.settings import get_settings
    settings = get_settings()
    
    step = WorkflowStep.AI_ANALYSIS
    executor = ClaudeAnalysisExecutor(step)
    
    # Store settings for later use, using config default if not provided
    executor._top_n_for_ai = top_n_for_ai if top_n_for_ai is not None else settings.ai_analysis_max_opportunities
    executor._min_data_completeness = min_data_completeness
    
    return executor


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    from src.models import WorkflowExecutionContext, WorkflowConfig
    
    async def test_claude_analysis_executor():
        """Test the Claude analysis executor with sample data."""
        
        # Create sample top opportunities data (from LocalRankingExecutor)
        sample_input = {
            'top_opportunities': [
                {
                    'symbol': 'AAPL',
                    'composite_score': 85.5,
                    'ranking': 1,
                    'proceed_to_ai': True,
                    'current_price': 175.50,
                    'technical_data': {'rsi': 58.2, 'macd': 1.25},
                    'momentum_data': {'momentum_21d': 0.045},
                    'squeeze_data': {'is_squeeze': True},
                    'score_breakdown': {'technical': 85, 'momentum': 80},
                    'best_call': {
                        'symbol': 'AAPL240315C00170000',
                        'strike': 170.0,
                        'bid': 8.50,
                        'ask': 8.75,
                        'delta': 0.65,
                        'score': 78
                    },
                    'enhanced_data': {
                        'fundamentals': {'company_info': {'name': 'Apple Inc.'}},
                        'news': [{'title': 'Apple announces...', 'date': '2024-01-15'}]
                    }
                }
            ],
            'ranking_summary': {
                'total_analyzed': 50,
                'top_n_selected': 1,
                'score_threshold': 85.5
            }
        }
        
        # Create execution context
        context = WorkflowExecutionContext(
            config=WorkflowConfig(enable_ai_analysis=True)
        )
        
        # Test executor
        executor = create_claude_analysis_executor()
        
        try:
            # Validate input
            await executor.validate_input(sample_input, context)
            
            # Execute (this will use mock client unless real API key is provided)
            result = await executor.execute_step(sample_input, context)
            
            # Validate output
            await executor.validate_output(result, context)
            
            print("✓ ClaudeAnalysisExecutor test completed successfully")
            print(f"Analyzed {result['opportunities_analyzed']} opportunities")
            print(f"Total cost: ${result['total_cost']:.4f}")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    # Run test
    asyncio.run(test_claude_analysis_executor())