"""Workflow-specific data models."""

from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class WorkflowStep(str, Enum):
    """Individual steps in the options screening workflow."""
    STOCK_SCREENING = "stock_screening"
    DATA_COLLECTION = "data_collection"
    TECHNICAL_CALCULATION = "technical_calculation"
    OPTION_SELECTION = "option_selection"
    LLM_PACKAGING = "llm_packaging"
    AI_ANALYSIS = "ai_analysis"
    RESULT_PROCESSING = "result_processing"


class WorkflowStepStatus(str, Enum):
    """Status of individual workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class WorkflowStepResult(BaseModel):
    """Result of executing a single workflow step."""
    
    step: WorkflowStep = Field(..., description="The workflow step")
    status: WorkflowStepStatus = Field(..., description="Step execution status")
    
    # Timing information
    started_at: datetime = Field(..., description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Step execution duration")
    
    # Results
    output_data: Optional[Any] = Field(None, description="Step output data")
    records_processed: Optional[int] = Field(None, ge=0, description="Number of records processed")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if step failed")
    retry_count: int = Field(0, ge=0, description="Number of retries attempted")
    
    # Performance metrics
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage")
    cpu_time_seconds: Optional[float] = Field(None, ge=0, description="CPU time consumed")
    
    # Provider statistics (if applicable)
    api_calls_made: Optional[int] = Field(None, ge=0, description="API calls made during step")
    api_cost: Optional[float] = Field(None, ge=0, description="API costs incurred")


class CheckpointData(BaseModel):
    """Data for workflow checkpointing and recovery."""
    
    workflow_id: str = Field(..., description="Workflow identifier")
    checkpoint_time: datetime = Field(default_factory=datetime.now, description="Checkpoint timestamp")
    
    # Progress information
    completed_steps: List[WorkflowStep] = Field(default_factory=list, description="Completed steps")
    current_step: Optional[WorkflowStep] = Field(None, description="Currently executing step")
    
    # Intermediate data
    intermediate_results: Dict[WorkflowStep, Any] = Field(default_factory=dict, description="Step results")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Workflow context")
    
    # Configuration
    workflow_config: Optional['WorkflowConfig'] = Field(None, description="Workflow configuration")
    
    # State information
    symbols_to_process: List[str] = Field(default_factory=list, description="Remaining symbols")
    symbols_completed: List[str] = Field(default_factory=list, description="Completed symbols")
    
    # Recovery information
    is_recovery: bool = Field(False, description="Whether this is a recovery checkpoint")
    recovery_reason: Optional[str] = Field(None, description="Reason for recovery")


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""
    
    # Execution settings
    max_concurrent_stocks: int = Field(50, gt=0, le=500, description="Max concurrent stock processing")
    max_retry_attempts: int = Field(3, ge=0, le=10, description="Max retry attempts per step")
    step_timeout_seconds: int = Field(300, gt=0, description="Timeout per step in seconds")
    
    # Error handling
    continue_on_errors: bool = Field(True, description="Continue processing on non-critical errors")
    error_threshold_percent: float = Field(10.0, ge=0, le=100, description="Max error rate before abort")
    
    # Performance settings
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl_seconds: int = Field(3600, gt=0, description="Cache TTL in seconds")
    
    # Checkpoint settings
    checkpoint_interval: int = Field(10, gt=0, description="Save checkpoint every N symbols")
    enable_validation: bool = Field(True, description="Enable step validation")
    
    # AI settings
    enable_ai_analysis: bool = Field(False, description="Enable AI analysis step")
    ai_batch_size: int = Field(10, gt=0, le=100, description="AI analysis batch size")
    max_ai_cost_dollars: float = Field(50.0, gt=0, description="Max AI cost per workflow")
    
    # Output settings
    save_intermediate_results: bool = Field(False, description="Save intermediate step results")
    output_format: str = Field("json", description="Output format")
    
    # Quality settings
    min_data_quality_score: float = Field(80.0, ge=0, le=100, description="Minimum data quality score")
    require_options_data: bool = Field(True, description="Require options data for inclusion")


class WorkflowExecutionContext(BaseModel):
    """Runtime context for workflow execution."""
    
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workflow ID")
    started_at: datetime = Field(default_factory=datetime.now, description="Workflow start time")
    
    # Configuration
    config: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    
    # Current state
    current_step: Optional[WorkflowStep] = Field(None, description="Currently executing step")
    status: WorkflowStatus = Field(WorkflowStatus.CREATED, description="Overall workflow status")
    
    # Progress tracking
    total_symbols: int = Field(0, ge=0, description="Total symbols to process")
    completed_symbols: int = Field(0, ge=0, description="Completed symbols")
    failed_symbols: int = Field(0, ge=0, description="Failed symbols")
    
    # Step tracking
    step_results: Dict[WorkflowStep, WorkflowStepResult] = Field(
        default_factory=dict, 
        description="Results for each step"
    )
    
    # Performance metrics
    total_api_calls: int = Field(0, ge=0, description="Total API calls made")
    total_cost: float = Field(0.0, ge=0, description="Total execution cost")
    peak_memory_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage")
    
    # Error tracking
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings encountered")
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.completed_symbols / self.total_symbols) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total_processed = self.completed_symbols + self.failed_symbols
        if total_processed == 0:
            return 0.0
        return (self.failed_symbols / total_processed) * 100
    
    @property
    def duration_seconds(self) -> float:
        """Calculate total execution duration."""
        return (datetime.now() - self.started_at).total_seconds()


class WorkflowMetrics(BaseModel):
    """Comprehensive workflow performance metrics."""
    
    workflow_id: str = Field(..., description="Workflow identifier")
    
    # Timing metrics
    total_duration_seconds: float = Field(..., ge=0, description="Total execution time")
    step_durations: Dict[WorkflowStep, float] = Field(default_factory=dict, description="Duration per step")
    
    # Throughput metrics
    symbols_per_second: float = Field(..., ge=0, description="Symbols processed per second")
    api_calls_per_second: float = Field(..., ge=0, description="API calls per second")
    
    # Resource usage
    peak_memory_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage")
    total_cpu_seconds: Optional[float] = Field(None, ge=0, description="Total CPU time")
    network_bytes_transferred: Optional[int] = Field(None, ge=0, description="Network data transferred")
    
    # Quality metrics
    success_rate: float = Field(..., ge=0, le=1, description="Overall success rate")
    data_quality_average: Optional[float] = Field(None, ge=0, le=100, description="Average data quality")
    
    # Cost metrics
    total_cost_dollars: float = Field(..., ge=0, description="Total execution cost")
    cost_per_symbol: float = Field(..., ge=0, description="Cost per symbol processed")
    
    # Provider-specific metrics
    provider_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Performance metrics by provider"
    )
    
    # Comparison metrics (if available)
    vs_previous_run: Optional[Dict[str, float]] = Field(None, description="Comparison to previous run")
    vs_baseline: Optional[Dict[str, float]] = Field(None, description="Comparison to baseline")
    
    metrics_generated_at: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")