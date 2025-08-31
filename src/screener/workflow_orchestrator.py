"""Workflow orchestrator that wires up the complete 7-step screening workflow.

This module provides the WorkflowOrchestrator class that:
1. Initializes and configures the WorkflowEngine
2. Registers all step executors with proper workflow step mapping
3. Handles the complete end-to-end workflow execution
4. Manages configuration and error handling
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from src.screener.workflow_engine import WorkflowEngine, WorkflowStepExecutor
from src.screener.steps import (
    StockScreeningExecutor,
    DataEnrichmentExecutor,
    TechnicalAnalysisExecutor,
    LocalRankingExecutor,
    ClaudeAnalysisExecutor,
    ResultProcessingExecutor
)
from src.models import (
    WorkflowStep,
    ScreeningCriteria,
    WorkflowConfig,
    WorkflowResult
)
from src.utils.logging_config import get_logger, print_info, print_success, print_error
from config.settings import get_settings

logger = get_logger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates the complete 7-step options screening workflow.
    
    This class handles:
    - WorkflowEngine initialization and configuration
    - Step executor registration with proper workflow step mapping
    - End-to-end workflow execution with error handling
    - Result aggregation and output formatting
    """
    
    def __init__(self, workflow_config: Optional[WorkflowConfig] = None):
        """
        Initialize the workflow orchestrator.
        
        Args:
            workflow_config: Optional workflow configuration
        """
        self.settings = get_settings()
        self.workflow_config = workflow_config or self._create_default_config()
        self.workflow_engine = WorkflowEngine()
        self._executors_registered = False
        
    def _create_default_config(self) -> WorkflowConfig:
        """Create default workflow configuration."""
        return WorkflowConfig(
            max_retry_attempts=3,
            step_timeout_seconds=1800,  # 30 minutes per step
            checkpoint_interval=10,  # Save checkpoint every 10 symbols
            max_concurrent_stocks=5,  # Process 5 stocks concurrently
            enable_caching=True,
            enable_validation=True
        )
    
    async def initialize(self) -> None:
        """Initialize the workflow orchestrator and register all step executors."""
        if self._executors_registered:
            return
        
        logger.debug("Initializing workflow orchestrator...")
        # Suppress setup message for cleaner output
        
        try:
            # Register all step executors with correct workflow step mapping
            await self._register_step_executors()
            
            self._executors_registered = True
            logger.debug("Workflow orchestrator initialized successfully")
            # Suppress success message for cleaner output
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow orchestrator: {e}")
            print_error(f"Initialization failed: {e}")
            raise
    
    async def _register_step_executors(self) -> None:
        """Register all step executors with the workflow engine."""
        
        # Step mapping: WorkflowStep -> Executor Class
        step_mappings = {
            # Step 1: Stock Screening & Filtering
            WorkflowStep.STOCK_SCREENING: StockScreeningExecutor,
            
            # Step 2: Market Data Collection  
            WorkflowStep.DATA_COLLECTION: DataEnrichmentExecutor,
            
            # Step 3: Technical Indicator Calculation
            WorkflowStep.TECHNICAL_CALCULATION: TechnicalAnalysisExecutor,
            
            # Step 4: Option Selection & Analysis (Local Ranking)
            WorkflowStep.OPTION_SELECTION: LocalRankingExecutor,
            
            # Step 5: AI Data Package Preparation (handled by Claude step)
            # This step is implicit in ClaudeAnalysisExecutor - no separate executor needed
            
            # Step 6: AI Rating & Analysis
            WorkflowStep.AI_ANALYSIS: ClaudeAnalysisExecutor,
            
            # Step 7: Result Processing & Output
            WorkflowStep.RESULT_PROCESSING: ResultProcessingExecutor
        }
        
        # Register each executor
        for workflow_step, executor_class in step_mappings.items():
            try:
                # Create executor instance
                executor = executor_class(workflow_step)
                
                # Register with workflow engine
                self.workflow_engine.register_step_executor(workflow_step, executor)
                
                logger.debug(f"Registered {executor_class.__name__} for step {workflow_step.value}")
                
            except Exception as e:
                logger.error(f"Failed to register {executor_class.__name__} for step {workflow_step.value}: {e}")
                raise
        
        # Special handling for LLM_PACKAGING step
        # This step is handled implicitly by ClaudeAnalysisExecutor, so we create a pass-through executor
        self.workflow_engine.register_step_executor(
            WorkflowStep.LLM_PACKAGING, 
            PassThroughExecutor(WorkflowStep.LLM_PACKAGING)
        )
        
        logger.info(f"Registered {len(step_mappings) + 1} step executors")
    
    async def execute_workflow(
        self, 
        criteria: ScreeningCriteria,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute the complete 7-step screening workflow.
        
        Args:
            criteria: Screening criteria configuration
            workflow_id: Optional workflow identifier for tracking
            
        Returns:
            WorkflowResult containing complete execution results
        """
        # Ensure orchestrator is initialized
        await self.initialize()
        
        logger.debug(f"Starting workflow execution with criteria: {criteria}")
        # Suppress verbose output - already shown in run_production.py
        
        try:
            # Execute workflow through engine
            result = await self.workflow_engine.execute_workflow(
                criteria=criteria,
                config=self.workflow_config,
                workflow_id=workflow_id
            )
            
            logger.debug(f"Workflow execution completed: {result.workflow_id}")
            # Success message shown in run_production.py
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            print_error(f"Workflow failed: {e}")
            raise
    
    async def get_workflow_status(self) -> Dict[str, any]:
        """Get current workflow status information."""
        if not self._executors_registered:
            return {"status": "not_initialized"}
        
        running_workflows = await self.workflow_engine.get_running_workflows()
        
        return {
            "status": "ready",
            "executors_registered": self._executors_registered,
            "running_workflows": len(running_workflows),
            "config": self.workflow_config.dict() if hasattr(self.workflow_config, 'dict') else str(self.workflow_config)
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow by ID."""
        if not self._executors_registered:
            return False
        
        return await self.workflow_engine.cancel_workflow(workflow_id)


class PassThroughExecutor(WorkflowStepExecutor):
    """
    Pass-through executor for LLM_PACKAGING step.
    
    The LLM packaging functionality is handled implicitly by ClaudeAnalysisExecutor,
    so this executor simply passes data through without modification.
    """
    
    def __init__(self, step: WorkflowStep):
        super().__init__(step)
        
    async def execute_step(self, input_data, context):
        """Pass data through unchanged."""
        logger.debug(f"Pass-through executor for step {self.step.value}")
        return input_data
    
    async def validate_input(self, input_data, context):
        """No input validation needed for pass-through."""
        pass
    
    async def validate_output(self, output_data, context):
        """No output validation needed for pass-through."""
        pass
    
    def get_records_processed(self, output_data):
        """Return count from input data if available."""
        if isinstance(output_data, dict) and 'top_opportunities' in output_data:
            return len(output_data['top_opportunities'])
        elif isinstance(output_data, list):
            return len(output_data)
        return 0


# Convenience function for simple workflow execution
async def run_screening_workflow(
    criteria: ScreeningCriteria,
    workflow_config: Optional[WorkflowConfig] = None,
    workflow_id: Optional[str] = None
) -> WorkflowResult:
    """
    Convenience function to run the complete screening workflow.
    
    Args:
        criteria: Screening criteria
        workflow_config: Optional workflow configuration
        workflow_id: Optional workflow identifier
        
    Returns:
        WorkflowResult with complete execution results
    """
    orchestrator = WorkflowOrchestrator(workflow_config)
    return await orchestrator.execute_workflow(criteria, workflow_id)