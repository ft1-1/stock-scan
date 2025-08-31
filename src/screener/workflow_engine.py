"""Core workflow engine for orchestrating the 7-step screening process."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

from src.models import (
    WorkflowStep,
    WorkflowStepStatus,
    WorkflowStatus,
    WorkflowStepResult,
    WorkflowExecutionContext,
    WorkflowConfig,
    WorkflowResult,
    CheckpointData,
    ScreeningCriteria,
    ScreeningResult
)
from src.providers.exceptions import ProviderError
from src.utils.logging_config import (
    get_logger, 
    workflow_progress,
    print_step_details,
    print_symbol_progress,
    print_info,
    print_warning,
    print_error,
    print_success,
    console
)
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))
from config.settings import get_settings

logger = get_logger(__name__)


class WorkflowStepExecutor:
    """Base class for executing individual workflow steps."""
    
    def __init__(self, step: WorkflowStep):
        self.step = step
        self.settings = get_settings()
    
    async def execute(
        self, 
        context: WorkflowExecutionContext,
        input_data: Any = None
    ) -> WorkflowStepResult:
        """Execute the workflow step."""
        logger.info(f"Starting execution of step: {self.step.value}")
        
        start_time = datetime.now()
        
        try:
            # Validate input
            await self.validate_input(input_data, context)
            
            # Execute step logic
            output_data = await self.execute_step(input_data, context)
            
            # Validate output
            await self.validate_output(output_data, context)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return WorkflowStepResult(
                step=self.step,
                status=WorkflowStepStatus.COMPLETED,
                started_at=start_time,
                completed_at=end_time,
                duration_seconds=duration,
                output_data=output_data,
                records_processed=self.get_records_processed(output_data)
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Step {self.step.value} failed: {str(e)}")
            
            return WorkflowStepResult(
                step=self.step,
                status=WorkflowStepStatus.FAILED,
                started_at=start_time,
                completed_at=end_time,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    async def validate_input(self, input_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate input data for the step."""
        pass
    
    async def execute_step(self, input_data: Any, context: WorkflowExecutionContext) -> Any:
        """Execute the actual step logic - to be implemented by subclasses."""
        raise NotImplementedError(f"Step {self.step.value} execution not implemented")
    
    async def validate_output(self, output_data: Any, context: WorkflowExecutionContext) -> None:
        """Validate output data from the step."""
        pass
    
    def get_records_processed(self, output_data: Any) -> Optional[int]:
        """Get number of records processed by this step."""
        if isinstance(output_data, list):
            return len(output_data)
        elif isinstance(output_data, dict) and 'count' in output_data:
            return output_data['count']
        return None


class CheckpointManager:
    """Manages workflow checkpointing for recovery."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.settings = get_settings()
        self.checkpoint_dir = checkpoint_dir or Path(self.settings.cache_directory) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_checkpoint(self, context: WorkflowExecutionContext) -> None:
        """Save workflow checkpoint."""
        if not self.settings.enable_checkpointing:
            return
        
        checkpoint_data = CheckpointData(
            workflow_id=context.workflow_id,
            completed_steps=list(context.step_results.keys()),
            current_step=context.current_step,
            intermediate_results={
                step.value: result.output_data 
                for step, result in context.step_results.items()
                if result.status == WorkflowStepStatus.COMPLETED
            },
            context_data={
                "total_symbols": context.total_symbols,
                "completed_symbols": context.completed_symbols,
                "failed_symbols": context.failed_symbols
            },
            workflow_config=context.config
        )
        
        checkpoint_file = self.checkpoint_dir / f"{context.workflow_id}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data.dict(), f, default=str, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def load_checkpoint(self, workflow_id: str) -> Optional[CheckpointData]:
        """Load workflow checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            return CheckpointData(**data)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {workflow_id}: {e}")
            return None
    
    async def cleanup_checkpoint(self, workflow_id: str) -> None:
        """Clean up checkpoint file after successful completion."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Checkpoint cleaned up: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint: {e}")


class WorkflowEngine:
    """Orchestrates the complete 7-step options screening workflow."""
    
    def __init__(self):
        self.settings = get_settings()
        self.checkpoint_manager = CheckpointManager()
        self.step_executors: Dict[WorkflowStep, WorkflowStepExecutor] = {}
        self._running_workflows: Dict[str, WorkflowExecutionContext] = {}
    
    def register_step_executor(self, step: WorkflowStep, executor: WorkflowStepExecutor):
        """Register an executor for a workflow step."""
        self.step_executors[step] = executor
        logger.info(f"Registered executor for step: {step.value}")
    
    async def execute_workflow(
        self,
        criteria: ScreeningCriteria,
        config: Optional[WorkflowConfig] = None,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """Execute the complete screening workflow."""
        
        # Initialize workflow context
        context = WorkflowExecutionContext(
            workflow_id=workflow_id or str(uuid.uuid4()),
            config=config or WorkflowConfig()
        )
        
        # Suppress detailed workflow configuration for cleaner output
        logger.debug(f"Workflow ID: {context.workflow_id}")
        if criteria.specific_symbols:
            logger.debug(f"Screening specific symbols: {', '.join(criteria.specific_symbols)}")
        
        # Check for existing checkpoint
        checkpoint = await self.checkpoint_manager.load_checkpoint(context.workflow_id)
        if checkpoint:
            print_warning(f"Resuming workflow from checkpoint: {context.workflow_id}")
            logger.info(f"Resuming workflow from checkpoint: {context.workflow_id}")
            context = await self._restore_from_checkpoint(context, checkpoint)
        
        # Register running workflow
        self._running_workflows[context.workflow_id] = context
        
        # Suppress progress tracking for cleaner output
        # workflow_progress.start_workflow(total_steps=7)
        
        try:
            context.status = WorkflowStatus.RUNNING
            logger.info(f"Starting workflow execution: {context.workflow_id}")
            
            # Execute workflow steps
            result = await self._execute_workflow_steps(context, criteria)
            
            # Mark as completed
            context.status = WorkflowStatus.COMPLETED
            
            # Cleanup checkpoint
            await self.checkpoint_manager.cleanup_checkpoint(context.workflow_id)
            
            # workflow_progress.finish_workflow(len(result.qualifying_results))
            logger.info(f"Workflow completed successfully: {context.workflow_id}")
            return result
            
        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.errors.append(str(e))
            
            logger.error(f"Workflow failed: {context.workflow_id} - {str(e)}")
            
            # Create failure result
            return WorkflowResult(
                workflow_id=context.workflow_id,
                started_at=context.started_at,
                completed_at=datetime.now(),
                screening_criteria=criteria,
                total_stocks_screened=context.completed_symbols,
                qualifying_results=[],
                execution_time_seconds=context.duration_seconds,
                errors_encountered=context.errors
            )
            
        finally:
            # Unregister workflow
            self._running_workflows.pop(context.workflow_id, None)
    
    async def _execute_workflow_steps(
        self, 
        context: WorkflowExecutionContext,
        criteria: ScreeningCriteria
    ) -> WorkflowResult:
        """Execute all workflow steps in sequence."""
        
        # Define step execution order with display names
        steps = [
            (WorkflowStep.STOCK_SCREENING, "Stock Screening & Filtering"),
            (WorkflowStep.DATA_COLLECTION, "Market Data Collection"),
            (WorkflowStep.TECHNICAL_CALCULATION, "Technical Indicator Calculation"),
            (WorkflowStep.OPTION_SELECTION, "Option Selection & Analysis"),
            (WorkflowStep.LLM_PACKAGING, "AI Data Package Preparation"),
            (WorkflowStep.AI_ANALYSIS, "AI Rating & Analysis"),
            (WorkflowStep.RESULT_PROCESSING, "Result Processing & Output")
        ]
        
        step_input = criteria
        
        for step_num, (step, step_name) in enumerate(steps, 1):
            # Skip if already completed (checkpoint recovery)
            if step in context.step_results:
                step_result = context.step_results[step]
                if step_result.status == WorkflowStepStatus.COMPLETED:
                    step_input = step_result.output_data
                    # workflow_progress.update_main(f"{step_name} (cached)", step_num)
                    continue
            
            # Update progress (suppressed for cleaner output)
            # workflow_progress.update_main(step_name, step_num)
            
            # Execute step
            context.current_step = step
            logger.debug(f"Executing step {step_num}/7: {step_name}")
            
            step_result = await self._execute_single_step(step, step_input, context)
            
            # Store result
            context.step_results[step] = step_result
            
            # Log step completion (only if records processed or important)
            if step_result.status == WorkflowStepStatus.COMPLETED:
                # Only show important steps with actual results
                if step_result.records_processed and step_result.records_processed > 0:
                    if step_num in [4, 6]:  # Only show ranking and AI analysis steps
                        print_success(f"Step {step_num}: {step_result.records_processed} records")
                if step_result.duration_seconds:
                    logger.debug(f"Step duration: {step_result.duration_seconds:.2f}s")
            
            # Handle step failure
            if step_result.status == WorkflowStepStatus.FAILED:
                print_error(f"Step {step_num} failed: {step_result.error_message}")
                if self._is_critical_step(step):
                    raise Exception(f"Critical step {step.value} failed: {step_result.error_message}")
                else:
                    print_warning(f"Non-critical step {step.value} failed, continuing")
                    logger.warning(f"Non-critical step {step.value} failed, continuing")
                    continue
            
            # Update input for next step
            step_input = step_result.output_data
            
            # Save checkpoint
            if context.completed_symbols % context.config.checkpoint_interval == 0:
                await self.checkpoint_manager.save_checkpoint(context)
                logger.debug(f"Checkpoint saved at {context.completed_symbols} symbols")
        
        # Compile final results
        return self._compile_workflow_result(context, criteria)
    
    async def _execute_single_step(
        self,
        step: WorkflowStep,
        input_data: Any,
        context: WorkflowExecutionContext
    ) -> WorkflowStepResult:
        """Execute a single workflow step with retries."""
        
        executor = self.step_executors.get(step)
        if not executor:
            raise Exception(f"No executor registered for step: {step.value}")
        
        retry_count = 0
        max_retries = context.config.max_retry_attempts
        step_start_time = datetime.now()
        
        while retry_count <= max_retries:
            try:
                # Check step-specific timeout (not total workflow duration)
                step_duration = (datetime.now() - step_start_time).total_seconds()
                # AI analysis step gets 2 hours, all other steps get the configured timeout
                timeout_seconds = 7200 if step == WorkflowStep.AI_ANALYSIS else context.config.step_timeout_seconds
                if step_duration > timeout_seconds:
                    raise TimeoutError(f"Step {step.value} exceeded timeout")
                
                # Execute step
                result = await executor.execute(context, input_data)
                
                if result.status == WorkflowStepStatus.COMPLETED:
                    return result
                elif result.status == WorkflowStepStatus.FAILED and retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"Step {step.value} failed, retrying ({retry_count}/{max_retries})")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    return result
                    
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    return WorkflowStepResult(
                        step=step,
                        status=WorkflowStepStatus.FAILED,
                        started_at=datetime.now(),
                        error_message=str(e),
                        retry_count=retry_count - 1
                    )
                
                logger.warning(f"Step {step.value} exception, retrying ({retry_count}/{max_retries}): {e}")
                await asyncio.sleep(2 ** retry_count)
        
        # Should not reach here
        raise Exception(f"Step {step.value} failed after {max_retries} retries")
    
    def _is_critical_step(self, step: WorkflowStep) -> bool:
        """Determine if a step is critical for workflow success."""
        critical_steps = {
            WorkflowStep.STOCK_SCREENING,
            WorkflowStep.DATA_COLLECTION,
            WorkflowStep.RESULT_PROCESSING
        }
        return step in critical_steps
    
    def _compile_workflow_result(
        self, 
        context: WorkflowExecutionContext,
        criteria: ScreeningCriteria
    ) -> WorkflowResult:
        """Compile final workflow result."""
        
        # Get final results from last step
        result_processing_step = context.step_results.get(WorkflowStep.RESULT_PROCESSING)
        # For now, use empty list to avoid validation errors
        # TODO: Convert results to proper ScreeningResult objects
        qualifying_results = []
        
        return WorkflowResult(
            workflow_id=context.workflow_id,
            started_at=context.started_at,
            completed_at=datetime.now(),
            screening_criteria=criteria,
            total_stocks_screened=context.total_symbols,
            qualifying_results=qualifying_results,
            execution_time_seconds=context.duration_seconds,
            api_calls_made=context.total_api_calls,
            total_cost=context.total_cost,
            success_rate=min(1.0, context.completed_symbols / max(context.total_symbols, 1)),
            errors_encountered=context.errors,
            warnings=context.warnings
        )
    
    async def _restore_from_checkpoint(
        self, 
        context: WorkflowExecutionContext,
        checkpoint: CheckpointData
    ) -> WorkflowExecutionContext:
        """Restore workflow context from checkpoint."""
        
        # Restore basic context
        context.total_symbols = checkpoint.context_data.get('total_symbols', 0)
        context.completed_symbols = checkpoint.context_data.get('completed_symbols', 0) 
        context.failed_symbols = checkpoint.context_data.get('failed_symbols', 0)
        
        # Restore step results
        for step_name, step_data in checkpoint.intermediate_results.items():
            try:
                step = WorkflowStep(step_name)
                context.step_results[step] = WorkflowStepResult(
                    step=step,
                    status=WorkflowStepStatus.COMPLETED,
                    started_at=checkpoint.cached_at,
                    completed_at=checkpoint.cached_at,
                    output_data=step_data
                )
            except ValueError:
                logger.warning(f"Unknown step in checkpoint: {step_name}")
        
        return context
    
    async def get_running_workflows(self) -> List[WorkflowExecutionContext]:
        """Get list of currently running workflows."""
        return list(self._running_workflows.values())
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        context = self._running_workflows.get(workflow_id)
        if context:
            context.status = WorkflowStatus.CANCELLED
            logger.info(f"Workflow cancelled: {workflow_id}")
            return True
        return False