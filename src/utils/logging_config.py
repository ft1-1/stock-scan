"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import sys
from pathlib import Path
# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import get_settings

# Global console and progress objects
console = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console,
    transient=False
)


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru sinks."""
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    # Remove default loguru logger
    logger.remove()
    
    # Ensure log directory exists
    log_path = Path(settings.log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console logging
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logging - Application logs
    logger.add(
        log_path / "application.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        enqueue=True
    )
    
    # File logging - Error logs
    logger.add(
        log_path / "errors.log", 
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="50 MB",
        retention="90 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True
    )
    
    # Debug logging (only in debug mode)
    if settings.debug:
        logger.add(
            log_path / "debug.log",
            level="DEBUG", 
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="7 days",
            compression="gz",
            enqueue=True
        )
    
    # Performance logging
    logger.add(
        log_path / "performance.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]} | {message}",
        filter=lambda record: "performance" in record["extra"],
        rotation="100 MB", 
        retention="30 days",
        compression="gz",
        enqueue=True
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configure third-party loggers
    for name in ["aiohttp", "urllib3", "requests", "asyncio"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    
    # Only log configuration completed if in DEBUG mode
    if settings.debug:
        logger.info("Logging configuration completed")


def get_logger(name: str) -> "logger":
    """Get a logger instance for a module."""
    return logger.bind(module=name)


def log_performance(component: str, operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    logger.bind(
        component=component,
        performance=True
    ).info(
        f"Performance: {operation} took {duration:.3f}s",
        extra={
            "component": component,
            "operation": operation,
            "duration_seconds": duration,
            **kwargs
        }
    )


def log_api_call(provider: str, endpoint: str, response_time: float, success: bool, **kwargs):
    """Log API call metrics."""
    level = "INFO" if success else "WARNING"
    logger.bind(
        component=f"api_{provider}",
        performance=True
    ).log(
        level,
        f"API Call: {provider}.{endpoint} - {response_time:.3f}s - {'Success' if success else 'Failed'}",
        extra={
            "component": f"api_{provider}",
            "provider": provider,
            "endpoint": endpoint,
            "response_time_seconds": response_time,
            "success": success,
            **kwargs
        }
    )


# Progress tracking functions
class WorkflowProgress:
    """Manages workflow progress display."""
    
    def __init__(self):
        self.main_task = None
        self.sub_tasks = {}
        
    def start_workflow(self, total_steps: int = 7):
        """Start the main workflow progress."""
        console.print(Panel.fit(
            "[bold cyan]Options Screening Workflow Started[/bold cyan]",
            border_style="cyan"
        ))
        progress.start()
        self.main_task = progress.add_task(
            "[cyan]Overall Progress", 
            total=total_steps
        )
        
    def update_main(self, step_name: str, step_num: int):
        """Update main workflow progress."""
        progress.update(
            self.main_task,
            description=f"[cyan]Step {step_num}/7: {step_name}",
            completed=step_num
        )
        # Suppress detailed step messages for cleaner output
        logger.debug(f"Starting Step {step_num}: {step_name}")
        
    def add_subtask(self, name: str, total: int):
        """Add a subtask progress bar."""
        task_id = progress.add_task(f"  └─ {name}", total=total)
        self.sub_tasks[name] = task_id
        return task_id
        
    def update_subtask(self, name: str, completed: int, description: str = None):
        """Update a subtask progress."""
        if name in self.sub_tasks:
            if description:
                progress.update(self.sub_tasks[name], completed=completed, description=f"  └─ {description}")
            else:
                progress.update(self.sub_tasks[name], completed=completed)
                
    def complete_subtask(self, name: str):
        """Mark a subtask as complete."""
        if name in self.sub_tasks:
            task = self.sub_tasks[name]
            total = progress.tasks[task].total
            progress.update(task, completed=total)
            console.print(f"    [green]✓[/green] {name} completed")
            
    def finish_workflow(self, results_count: int):
        """Finish the workflow and display results."""
        progress.stop()
        console.print(Panel.fit(
            f"[bold green]Workflow Completed Successfully[/bold green]\n"
            f"Found [bold yellow]{results_count}[/bold yellow] opportunities",
            border_style="green"
        ))


# Global workflow progress instance
workflow_progress = WorkflowProgress()


def print_step_details(step_name: str, details: dict):
    """Print detailed information about a step."""
    table = Table(title=f"{step_name} Details", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=30)
    table.add_column("Value", style="yellow")
    
    for key, value in details.items():
        table.add_row(str(key), str(value))
    
    console.print(table)


def print_symbol_progress(current: int, total: int, symbol: str, status: str = "Processing"):
    """Print progress for symbol processing."""
    console.print(f"  [{current}/{total}] {status} [bold cyan]{symbol}[/bold cyan]")


def print_results_summary(results: list):
    """Print a summary table of results."""
    if not results:
        console.print("[yellow]No qualifying opportunities found[/yellow]")
        return
        
    table = Table(title="Top Opportunities", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=5)
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Score", style="yellow", width=10)
    table.add_column("Price", style="green", width=10)
    table.add_column("Option", style="blue", width=20)
    table.add_column("AI Rating", style="magenta", width=10)
    
    for i, result in enumerate(results[:10], 1):
        table.add_row(
            str(i),
            result.get("symbol", "N/A"),
            f"{result.get('score', 0):.1f}",
            f"${result.get('price', 0):.2f}",
            result.get('option_contract', 'N/A'),
            f"{result.get('ai_rating', 'N/A')}/100"
        )
    
    console.print(table)


def print_error(message: str, exception: Exception = None):
    """Print an error message with formatting."""
    console.print(f"[bold red]❌ Error:[/bold red] {message}")
    if exception:
        console.print(f"[dim]Details: {str(exception)}[/dim]")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[bold blue]ℹ Info:[/bold blue] {message}")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[bold green]✓ Success:[/bold green] {message}")