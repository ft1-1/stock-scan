"""Local ranking step executor for filtering and ranking quantitative opportunities.

This module implements the LocalRankingExecutor that ranks all opportunities by
composite score and filters to the top N highest-scored opportunities before
sending to Claude for AI analysis.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import traceback

from src.screener.workflow_engine import WorkflowStepExecutor
from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStepResult,
    WorkflowExecutionContext
)

from src.utils.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class LocalRankingExecutor(WorkflowStepExecutor):
    """
    Local Ranking executor that:
    1. Ranks all opportunities by composite quantitative score
    2. Filters to top N highest-scored opportunities
    3. Adds ranking metadata and score rationale
    4. Only sends the best opportunities to Claude for AI analysis
    """
    
    def __init__(self, step: WorkflowStep, top_n: Optional[int] = None):
        super().__init__(step)
        self.settings = get_settings()
        # Use setting from config if not explicitly provided
        self.top_n = top_n if top_n is not None else self.settings.ai_analysis_max_opportunities
        
    async def execute_step(
        self, 
        input_data: Any, 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute local ranking and filtering of analyzed opportunities.
        
        Args:
            input_data: Output from TechnicalAnalysisExecutor with analyzed opportunities
            context: Workflow execution context
            
        Returns:
            Dict containing ranked and filtered opportunities for AI analysis
        """
        # Input validation
        if not isinstance(input_data, dict) or 'analyzed_opportunities' not in input_data:
            raise ValueError("Input must be analyzed opportunities from TechnicalAnalysisExecutor")
            
        analyzed_opportunities = input_data['analyzed_opportunities']
        if not analyzed_opportunities:
            logger.warning("No analyzed opportunities provided for ranking")
            return {
                'top_opportunities': [],
                'ranking_summary': {
                    'total_analyzed': 0,
                    'top_n_selected': 0,
                    'score_threshold': 0.0,
                    'ranking_timestamp': datetime.now().isoformat()
                },
                'all_ranked_opportunities': []
            }
        
        logger.info(f"Starting local ranking for {len(analyzed_opportunities)} opportunities")
        
        # 1. Rank all opportunities by composite score
        ranked_opportunities = await self._rank_all_opportunities(analyzed_opportunities)
        
        # 2. Calculate ranking statistics
        ranking_stats = self._calculate_ranking_statistics(ranked_opportunities)
        
        # 3. Filter to top N opportunities
        top_opportunities = await self._filter_top_opportunities(
            ranked_opportunities, self.top_n, ranking_stats
        )
        
        # 4. Add ranking metadata to each opportunity
        for i, opportunity in enumerate(top_opportunities):
            opportunity['ranking'] = i + 1
            opportunity['proceed_to_ai'] = True
            opportunity['selection_rationale'] = self._generate_selection_rationale(
                opportunity, ranking_stats
            )
        
        # Collect enhanced data for qualified stocks
        if top_opportunities:
            logger.info(f"Collecting enhanced data for {len(top_opportunities)} qualified stocks")
            try:
                # Import data enrichment step for enhanced collection
                from src.screener.steps.data_enrichment_step import DataEnrichmentExecutor
                from src.models import WorkflowStep
                
                # Use the DATA_COLLECTION step enum for the executor initialization
                enrichment_executor = DataEnrichmentExecutor(WorkflowStep.DATA_COLLECTION)
                
                # Collect enhanced data only for stocks that passed threshold
                top_opportunities = await enrichment_executor.collect_enhanced_data_for_qualified_stocks(
                    top_opportunities, context
                )
                logger.info(f"Enhanced data collection completed for {len(top_opportunities)} stocks")
            except Exception as e:
                logger.error(f"Failed to collect enhanced data: {e}")
                # Continue with existing data if enhancement fails
        
        # Update context
        context.completed_symbols = len(top_opportunities)
        
        ranking_summary = {
            'total_analyzed': len(analyzed_opportunities),
            'top_n_selected': len(top_opportunities),
            'top_n_target': self.top_n,
            'score_threshold': top_opportunities[-1]['composite_score'] if top_opportunities else 0.0,
            'average_score': ranking_stats['average_score'],
            'median_score': ranking_stats['median_score'],
            'score_distribution': ranking_stats['score_distribution'],
            'ranking_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Local ranking completed: {len(top_opportunities)} opportunities selected "
                   f"from {len(analyzed_opportunities)} total (threshold: {ranking_summary['score_threshold']:.1f})")
        
        return {
            'top_opportunities': top_opportunities,
            'ranking_summary': ranking_summary,
            'all_ranked_opportunities': ranked_opportunities,
            'original_analysis_summary': input_data.get('analysis_summary', {})
        }
    
    async def _rank_all_opportunities(self, opportunities: Dict[str, Dict]) -> List[Dict]:
        """
        Rank all opportunities by composite score and add ranking details.
        
        Args:
            opportunities: Dictionary of analyzed opportunities by symbol
            
        Returns:
            List of opportunities sorted by composite score (descending)
        """
        logger.info(f"Ranking {len(opportunities)} opportunities by composite score")
        
        ranked_list = []
        
        for symbol, opportunity in opportunities.items():
            # Ensure opportunity has required fields
            if not isinstance(opportunity, dict):
                logger.warning(f"Invalid opportunity data for {symbol}")
                continue
            
            composite_score = opportunity.get('composite_score', 0.0)
            if not isinstance(composite_score, (int, float)):
                logger.warning(f"Invalid composite score for {symbol}: {composite_score}")
                continue
            
            # Add ranking metadata
            ranked_opportunity = opportunity.copy()
            ranked_opportunity['symbol'] = symbol
            ranked_opportunity['composite_score'] = float(composite_score)
            
            # Add score validation and normalization
            if ranked_opportunity['composite_score'] < 0:
                ranked_opportunity['composite_score'] = 0.0
            elif ranked_opportunity['composite_score'] > 100:
                ranked_opportunity['composite_score'] = 100.0
            
            ranked_list.append(ranked_opportunity)
        
        # Sort by composite score (highest first)
        ranked_list.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add global ranking position
        for i, opportunity in enumerate(ranked_list):
            opportunity['global_ranking'] = i + 1
        
        logger.info(f"Ranking completed. Score range: {ranked_list[0]['composite_score']:.1f} - "
                   f"{ranked_list[-1]['composite_score']:.1f}")
        
        return ranked_list
    
    def _calculate_ranking_statistics(self, ranked_opportunities: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for the ranked opportunities."""
        if not ranked_opportunities:
            return {
                'average_score': 0.0,
                'median_score': 0.0,
                'score_distribution': {},
                'total_count': 0
            }
        
        scores = [opp['composite_score'] for opp in ranked_opportunities]
        
        # Calculate basic statistics
        average_score = np.mean(scores)
        median_score = np.median(scores)
        
        # Score distribution buckets
        score_distribution = {
            'excellent': len([s for s in scores if s >= 80]),    # 80-100
            'very_good': len([s for s in scores if 70 <= s < 80]),  # 70-79
            'good': len([s for s in scores if 60 <= s < 70]),       # 60-69
            'average': len([s for s in scores if 50 <= s < 60]),    # 50-59
            'below_average': len([s for s in scores if 40 <= s < 50]),  # 40-49
            'poor': len([s for s in scores if s < 40])              # 0-39
        }
        
        return {
            'average_score': float(average_score),
            'median_score': float(median_score),
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
            'score_distribution': score_distribution,
            'total_count': len(ranked_opportunities)
        }
    
    async def _filter_top_opportunities(self, ranked_opportunities: List[Dict], 
                                      top_n: int, ranking_stats: Dict) -> List[Dict]:
        """
        Filter to top N opportunities with additional quality checks.
        
        Args:
            ranked_opportunities: All opportunities ranked by score
            top_n: Number of top opportunities to select
            ranking_stats: Ranking statistics for context
            
        Returns:
            List of top N opportunities with quality validation
        """
        if not ranked_opportunities:
            return []
        
        # Apply minimum score threshold - configurable via environment variable
        min_score_threshold = self.settings.ai_analysis_min_score
        
        # Filter opportunities that meet minimum quality standards
        quality_filtered = []
        for opportunity in ranked_opportunities:
            score = opportunity['composite_score']
            
            # Basic score threshold
            if score < min_score_threshold:
                opportunity['rejection_reason'] = f"Score {score:.1f} below threshold {min_score_threshold:.1f}"
                logger.info(f"Rejecting {opportunity.get('symbol', 'unknown')}: Score {score:.1f} < {min_score_threshold:.1f} threshold")
                continue
            
            # Additional quality checks
            warnings = opportunity.get('warnings', [])
            if len(warnings) > 3:  # Too many analysis warnings
                opportunity['rejection_reason'] = f"Too many analysis warnings: {len(warnings)}"
                continue
            
            # Check for critical missing data
            # Temporarily disabled to allow all symbols through for testing
            # score_breakdown = opportunity.get('score_breakdown', {})
            # if all(score <= 10 for score in score_breakdown.values()):
            #     opportunity['rejection_reason'] = "All component scores too low"
            #     continue
            
            quality_filtered.append(opportunity)
        
        logger.info(f"Quality filtering: {len(quality_filtered)} of {len(ranked_opportunities)} "
                   f"opportunities passed (threshold: {min_score_threshold:.1f})")
        
        if len(quality_filtered) == 0 and len(ranked_opportunities) > 0:
            logger.warning(f"All {len(ranked_opportunities)} opportunities rejected - none met the {min_score_threshold:.1f} score threshold for Claude analysis")
        
        # Select top N from quality-filtered opportunities
        top_opportunities = quality_filtered[:top_n]
        
        # Add selection metadata
        for i, opportunity in enumerate(top_opportunities):
            opportunity['selection_rank'] = i + 1
            opportunity['passed_quality_filter'] = True
            opportunity['score_percentile'] = self._calculate_percentile(
                opportunity['composite_score'], 
                [opp['composite_score'] for opp in ranked_opportunities]
            )
        
        logger.info(f"Selected top {len(top_opportunities)} opportunities for AI analysis")
        
        return top_opportunities
    
    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile rank of a score within all scores."""
        if not all_scores:
            return 50.0
        
        below_count = sum(1 for s in all_scores if s < score)
        total_count = len(all_scores)
        
        percentile = (below_count / total_count) * 100
        return round(percentile, 1)
    
    def _generate_selection_rationale(self, opportunity: Dict, ranking_stats: Dict) -> str:
        """Generate human-readable rationale for why this opportunity was selected."""
        symbol = opportunity['symbol']
        score = opportunity['composite_score']
        rank = opportunity['global_ranking']
        total = ranking_stats['total_count']
        
        # Score interpretation
        if score >= 80:
            score_desc = "excellent"
        elif score >= 70:
            score_desc = "very good"
        elif score >= 60:
            score_desc = "good"
        else:
            score_desc = "acceptable"
        
        rationale_parts = [
            f"Ranked #{rank} of {total} opportunities with {score_desc} composite score of {score:.1f}"
        ]
        
        # Highlight strong components
        score_breakdown = opportunity.get('score_breakdown', {})
        strong_components = []
        for component, component_score in score_breakdown.items():
            if component_score >= 70:  # Strong component
                strong_components.append(component.replace('_', ' ').title())
        
        if strong_components:
            rationale_parts.append(f"Strong {', '.join(strong_components)} signals")
        
        # Add specific highlights
        highlights = []
        
        # Check for squeeze
        squeeze_data = opportunity.get('squeeze_data', {})
        if squeeze_data.get('is_squeeze'):
            highlights.append("TTM Squeeze detected")
        
        # Check for options availability
        best_call = opportunity.get('best_call')
        if best_call and best_call.get('score', 0) >= 70:
            highlights.append(f"High-quality call option available (score: {best_call['score']:.0f})")
        
        # Check for momentum
        momentum_data = opportunity.get('momentum_data', {})
        momentum_21d = momentum_data.get('momentum_21d')
        if momentum_21d and momentum_21d > 10:
            highlights.append(f"Strong momentum ({momentum_21d:.1f}%)")
        
        if highlights:
            rationale_parts.append(". ".join(highlights))
        
        return ". ".join(rationale_parts) + "."
    
    def _validate_opportunity_structure(self, opportunity: Dict, symbol: str) -> Tuple[bool, List[str]]:
        """Validate that an opportunity has the expected structure."""
        errors = []
        
        # Required fields
        required_fields = ['composite_score', 'score_breakdown']
        for field in required_fields:
            if field not in opportunity:
                errors.append(f"Missing required field: {field}")
        
        # Validate score breakdown
        score_breakdown = opportunity.get('score_breakdown', {})
        expected_components = ['technical', 'momentum', 'squeeze', 'quality']
        for component in expected_components:
            if component not in score_breakdown:
                errors.append(f"Missing score component: {component}")
            elif not isinstance(score_breakdown[component], (int, float)):
                errors.append(f"Invalid score type for {component}")
        
        # Validate composite score
        composite_score = opportunity.get('composite_score')
        if not isinstance(composite_score, (int, float)):
            errors.append("Invalid composite score type")
        elif composite_score < 0 or composite_score > 100:
            errors.append(f"Composite score out of range: {composite_score}")
        
        return len(errors) == 0, errors
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data from TechnicalAnalysisExecutor."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'analyzed_opportunities' not in input_data:
            raise ValueError("Input must contain 'analyzed_opportunities' key")
        
        opportunities = input_data['analyzed_opportunities']
        if not isinstance(opportunities, dict):
            raise ValueError("Analyzed opportunities must be a dictionary")
        
        # Validate a few sample opportunities
        sample_symbols = list(opportunities.keys())[:3]
        for symbol in sample_symbols:
            opportunity = opportunities[symbol]
            is_valid, errors = self._validate_opportunity_structure(opportunity, symbol)
            if not is_valid:
                raise ValueError(f"Invalid opportunity structure for {symbol}: {'; '.join(errors)}")
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from local ranking."""
        if not isinstance(output_data, dict):
            raise ValueError("Output data must be a dictionary")
        
        required_keys = ['top_opportunities', 'ranking_summary']
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")
        
        # Validate top opportunities structure
        top_opportunities = output_data['top_opportunities']
        if not isinstance(top_opportunities, list):
            raise ValueError("Top opportunities must be a list")
        
        # Check that ranking metadata was added
        for i, opportunity in enumerate(top_opportunities):
            if 'ranking' not in opportunity:
                raise ValueError(f"Opportunity {i} missing ranking metadata")
            if 'proceed_to_ai' not in opportunity:
                raise ValueError(f"Opportunity {i} missing proceed_to_ai flag")
            if 'selection_rationale' not in opportunity:
                raise ValueError(f"Opportunity {i} missing selection rationale")
        
        # Validate ranking summary
        ranking_summary = output_data['ranking_summary']
        required_summary_keys = ['total_analyzed', 'top_n_selected', 'score_threshold']
        for key in required_summary_keys:
            if key not in ranking_summary:
                raise ValueError(f"Missing ranking summary key: {key}")
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, dict):
            return len(output_data.get('top_opportunities', []))
        return 0