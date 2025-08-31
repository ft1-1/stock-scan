"""Result processing step executor for final workflow aggregation and output.

This module implements the ResultProcessingExecutor that aggregates all analysis results,
creates final summary reports, and handles output formatting for the workflow.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.screener.workflow_engine import WorkflowStepExecutor
from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStepResult,
    WorkflowExecutionContext
)
from src.utils.logging_config import get_logger, print_info, print_success, print_warning
from src.notifications.mailgun_client import create_mailgun_client

logger = get_logger(__name__)


class ResultProcessingExecutor(WorkflowStepExecutor):
    """
    Result processing executor that handles final workflow output and aggregation.
    
    This executor:
    - Aggregates results from all previous workflow steps
    - Creates summary statistics and rankings
    - Saves timestamped results to JSON files
    - Formats output for external consumption
    - Handles both successful and failed analysis results
    """
    
    def __init__(self, step: WorkflowStep):
        super().__init__(step)
        
    async def execute_step(
        self, 
        input_data: Any, 
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute result processing and create final workflow output.
        
        Args:
            input_data: AI analysis results from previous step
            context: Workflow execution context with all step results
            
        Returns:
            Dict containing final aggregated results and summary
        """
        logger.debug("Starting result processing and final aggregation")
        
        # Extract AI analysis results
        ai_results = input_data or {}
        if not isinstance(ai_results, dict):
            logger.warning(f"Unexpected AI results format: {type(input_data)}")
            ai_results = {}
        
        # Collect all workflow data
        workflow_data = await self._collect_workflow_data(context, ai_results)
        
        # Create final results structure
        final_results = await self._create_final_results(workflow_data, context)
        
        # Save results to file
        results_file = await self._save_results(final_results, context)
        
        # Send email notification for top opportunities
        await self._send_email_notification(final_results)
        
        # Log summary
        await self._log_summary(final_results)
        
        logger.debug(f"Result processing completed successfully: {len(final_results.get('results', []))} opportunities")
        
        return final_results
    
    async def _collect_workflow_data(
        self, 
        context: WorkflowExecutionContext,
        ai_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect data from all workflow steps."""
        workflow_data = {
            'screening_data': {},
            'enriched_data': {},
            'technical_data': {},
            'ranking_data': {},
            'ai_analysis': ai_results
        }
        
        # Extract data from each completed step
        step_results = context.step_results
        
        # Stock screening results
        if WorkflowStep.STOCK_SCREENING in step_results:
            screening_result = step_results[WorkflowStep.STOCK_SCREENING]
            if screening_result.status == WorkflowStepStatus.COMPLETED:
                workflow_data['screening_data'] = {
                    'symbols': screening_result.output_data or [],
                    'total_symbols': len(screening_result.output_data or [])
                }
        
        # Data enrichment results
        if WorkflowStep.DATA_COLLECTION in step_results:
            enrichment_result = step_results[WorkflowStep.DATA_COLLECTION]
            if enrichment_result.status == WorkflowStepStatus.COMPLETED:
                workflow_data['enriched_data'] = enrichment_result.output_data or {}
        
        # Technical analysis results
        if WorkflowStep.TECHNICAL_CALCULATION in step_results:
            technical_result = step_results[WorkflowStep.TECHNICAL_CALCULATION]
            if technical_result.status == WorkflowStepStatus.COMPLETED:
                workflow_data['technical_data'] = technical_result.output_data or {}
        
        # Local ranking results
        if WorkflowStep.OPTION_SELECTION in step_results:
            ranking_result = step_results[WorkflowStep.OPTION_SELECTION]
            if ranking_result.status == WorkflowStepStatus.COMPLETED:
                workflow_data['ranking_data'] = ranking_result.output_data or {}
        
        return workflow_data
    
    async def _create_final_results(
        self, 
        workflow_data: Dict[str, Any],
        context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """Create the final results structure."""
        
        # Get ranking data for opportunities
        ranking_data = workflow_data.get('ranking_data', {})
        opportunities = ranking_data.get('top_opportunities', [])
        
        # Get AI analysis results
        ai_analysis = workflow_data.get('ai_analysis', {})
        # The AI analysis returns a dictionary keyed by symbol
        ai_analysis_results = ai_analysis.get('ai_analysis_results', {})
        
        # Convert to list format for backward compatibility
        ai_opportunities = []
        for symbol, result in ai_analysis_results.items():
            if result.get('success'):
                parsed = result.get('claude_response', {}).get('parsed_analysis', {})
                ai_opportunities.append({
                    'symbol': symbol,
                    'rating': parsed.get('rating', 0),
                    'reasoning': parsed.get('thesis', ''),
                    'confidence': parsed.get('confidence', 'medium'),
                    'thesis': parsed.get('thesis', ''),
                    'opportunities': parsed.get('opportunities', []),
                    'risks': parsed.get('risks', []),
                    'option_contract': parsed.get('option_contract', {}),
                    'red_flags': parsed.get('red_flags', []),
                    'notes': parsed.get('notes', ''),
                    'analysis_file': f"data/ai_analysis/{datetime.now().strftime('%Y%m%d')}/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                })
        
        # Merge opportunities with AI analysis
        final_opportunities = []
        for opportunity in opportunities:
            symbol = opportunity.get('symbol')
            if not symbol:
                continue
            
            # Find corresponding AI analysis
            ai_result = None
            for ai_opp in ai_opportunities:
                if ai_opp.get('symbol') == symbol:
                    ai_result = ai_opp
                    break
            
            # Create merged opportunity result
            merged_opportunity = {
                'symbol': symbol,
                'local_score': opportunity.get('score', 0),
                'momentum_score': opportunity.get('momentum_score', 0),
                'squeeze_score': opportunity.get('squeeze_score', 0),
                'best_call': opportunity.get('best_call'),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Add AI analysis if available
            if ai_result:
                merged_opportunity.update({
                    'ai_rating': ai_result.get('rating', 0),
                    'ai_reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('confidence', 0),
                    'analysis_file': ai_result.get('analysis_file'),
                    # Add detailed AI analysis for email
                    'ai_analysis': {
                        'thesis': ai_result.get('thesis', ''),
                        'opportunities': ai_result.get('opportunities', []),
                        'risks': ai_result.get('risks', []),
                        'option_contract': ai_result.get('option_contract', {}),
                        'red_flags': ai_result.get('red_flags', []),
                        'notes': ai_result.get('notes', ''),
                        'confidence': ai_result.get('confidence', 'medium')
                    }
                })
            else:
                merged_opportunity.update({
                    'ai_rating': None,
                    'ai_reasoning': 'AI analysis not completed',
                    'ai_confidence': None,
                    'analysis_file': None,
                    'ai_analysis': {}
                })
            
            final_opportunities.append(merged_opportunity)
        
        # Sort by combined score (local score + AI rating)
        def combined_score(opp):
            local = opp.get('local_score', 0)
            ai_rating = opp.get('ai_rating', 0) or 0
            return (local * 0.1) + (ai_rating * 0.9)  # Weight AI rating 90%, local 10%
        
        final_opportunities.sort(key=combined_score, reverse=True)
        
        # Calculate execution statistics
        total_execution_time = sum(
            result.duration_seconds or 0 
            for result in context.step_results.values()
        )
        
        # Calculate costs
        total_cost = context.total_cost or 0
        
        # Create final results structure
        final_results = {
            'workflow_id': context.workflow_id,
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {
                'total_execution_time_seconds': total_execution_time,
                'symbols_screened': workflow_data.get('screening_data', {}).get('total_symbols', 0),
                'opportunities_found': len(opportunities),
                'ai_analyses_completed': len(ai_opportunities),
                'total_api_calls': context.total_api_calls or 0,
                'total_cost_usd': total_cost,
                'success_rate': (context.completed_symbols / max(context.total_symbols, 1)) if context.total_symbols else 1.0
            },
            'screening_criteria': self._extract_screening_criteria(context),
            'results': final_opportunities,
            'step_details': self._extract_step_details(context),
            'warnings': context.warnings or [],
            'errors': context.errors or []
        }
        
        return final_results
    
    def _extract_screening_criteria(self, context: WorkflowExecutionContext) -> Dict[str, Any]:
        """Extract screening criteria from workflow context."""
        try:
            # The screening criteria should be in the initial workflow config
            # For now, return basic info - could be enhanced with actual criteria
            return {
                'screening_method': 'automated',
                'timestamp': context.started_at.isoformat(),
                'workflow_id': context.workflow_id
            }
        except Exception as e:
            logger.warning(f"Could not extract screening criteria: {e}")
            return {}
    
    def _extract_step_details(self, context: WorkflowExecutionContext) -> List[Dict[str, Any]]:
        """Extract detailed information about each workflow step."""
        step_details = []
        
        for step, result in context.step_results.items():
            detail = {
                'step': step.value,
                'status': result.status.value,
                'duration_seconds': result.duration_seconds,
                'records_processed': result.records_processed,
                'started_at': result.started_at.isoformat() if result.started_at else None,
                'completed_at': result.completed_at.isoformat() if result.completed_at else None
            }
            
            if result.error_message:
                detail['error_message'] = result.error_message
            
            if result.retry_count:
                detail['retry_count'] = result.retry_count
            
            step_details.append(detail)
        
        return step_details
    
    async def _save_results(
        self, 
        final_results: Dict[str, Any],
        context: WorkflowExecutionContext
    ) -> str:
        """Save final results to a timestamped JSON file."""
        try:
            # Create output directory
            output_dir = Path(self.settings.output_directory) / "workflow_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_results_{timestamp}.json"
            results_file = output_dir / filename
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Final results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Return a fallback path
            return f"ERROR: Could not save results - {str(e)}"
    
    async def _log_summary(self, final_results: Dict[str, Any]) -> None:
        """Log workflow execution summary."""
        summary = final_results.get('execution_summary', {})
        results = final_results.get('results', [])
        
        # Suppress execution summary - already shown in run_production.py
        logger.debug(f"Workflow execution summary: {summary}")
        
        # Suppress top opportunities display - shown in run_production.py
        if results:
            logger.debug(f"Top {len(results[:5])} opportunities processed")
        
        # Log warnings and errors
        warnings = final_results.get('warnings', [])
        errors = final_results.get('errors', [])
        
        if warnings:
            print_warning(f"Warnings encountered: {len(warnings)}")
            for warning in warnings[-3:]:  # Show last 3 warnings
                print_warning(f"  - {warning}")
        
        if errors:
            print_warning(f"Errors encountered: {len(errors)}")
            for error in errors[-3:]:  # Show last 3 errors
                print_warning(f"  - {error}")
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data for the result processing step."""
        # Result processing can handle any input format
        # AI analysis results are optional (workflow might complete without AI)
        pass
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from the result processing step."""
        if not isinstance(output_data, dict):
            raise ValueError("Output data must be a dictionary")
        
        required_keys = ['workflow_id', 'timestamp', 'execution_summary', 'results']
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")
        
        # Validate execution summary
        exec_summary = output_data['execution_summary']
        if not isinstance(exec_summary, dict):
            raise ValueError("execution_summary must be a dictionary")
        
        # Validate results array
        results = output_data['results']
        if not isinstance(results, list):
            raise ValueError("results must be a list")
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, dict) and 'results' in output_data:
            return len(output_data['results'])
        return 0
    
    async def _send_email_notification(self, final_results: Dict[str, Any]) -> None:
        """Send email notification with top 10 opportunities."""
        try:
            # Check if email notifications are enabled
            if os.getenv('ENABLE_EMAIL_NOTIFICATIONS', 'false').lower() != 'true':
                logger.info("Email notifications disabled")
                return
            
            # Get email recipients from environment
            recipients_str = os.getenv('EMAIL_RECIPIENTS')
            if not recipients_str:
                logger.info("No email recipients configured")
                return
            
            recipients = [email.strip() for email in recipients_str.split(',')]
            
            # Get top 10 opportunities
            opportunities = final_results.get('results', [])[:10]
            
            if not opportunities:
                logger.info("No opportunities to email")
                return
            
            # Create Mailgun client
            mailgun_client = create_mailgun_client()
            if not mailgun_client:
                logger.warning("Mailgun client not configured - skipping email")
                return
            
            # Send email
            execution_summary = final_results.get('execution_summary', {})
            success = await mailgun_client.send_opportunity_alert(
                recipients=recipients,
                opportunities=opportunities,
                execution_summary=execution_summary
            )
            
            if success:
                print_success(f"📧 Email sent to {len(recipients)} recipients with top {len(opportunities)} opportunities")
            else:
                print_warning("Failed to send email notification")
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            print_warning(f"Could not send email notification: {e}")