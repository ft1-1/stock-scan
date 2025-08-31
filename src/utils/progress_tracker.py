"""Progress tracking utilities for detailed workflow monitoring."""

import time
from typing import Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
import logging

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class StepMetrics:
    """Metrics for a workflow step."""
    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    api_calls_made: int = 0
    api_calls_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def duration_seconds(self) -> float:
        """Calculate step duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.records_processed + self.records_failed
        if total > 0:
            return (self.records_processed / total) * 100
        return 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            return (self.cache_hits / total) * 100
        return 0.0


class DetailedProgressTracker:
    """Detailed progress tracking for workflow execution."""
    
    def __init__(self):
        self.console = Console()
        self.step_metrics: Dict[str, StepMetrics] = {}
        self.current_step: Optional[str] = None
        self.symbol_progress: Dict[str, str] = {}
        self.live_display: Optional[Live] = None
        
    def start_tracking(self):
        """Start live progress tracking display."""
        layout = self._create_layout()
        self.live_display = Live(layout, console=self.console, refresh_per_second=2)
        self.live_display.start()
    
    def stop_tracking(self):
        """Stop live progress tracking."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
    
    def start_step(self, step_name: str):
        """Start tracking a new step."""
        self.current_step = step_name
        if step_name not in self.step_metrics:
            self.step_metrics[step_name] = StepMetrics(name=step_name)
        self.step_metrics[step_name].started_at = datetime.now()
        logger.info(f"Started step: {step_name}")
        self._update_display()
    
    def complete_step(self, step_name: str):
        """Complete tracking for a step."""
        if step_name in self.step_metrics:
            self.step_metrics[step_name].completed_at = datetime.now()
            logger.info(f"Completed step: {step_name} in {self.step_metrics[step_name].duration_seconds:.2f}s")
        self._update_display()
    
    def update_symbol_progress(self, symbol: str, status: str):
        """Update progress for a specific symbol."""
        self.symbol_progress[symbol] = status
        if self.current_step and self.current_step in self.step_metrics:
            if status == "completed":
                self.step_metrics[self.current_step].records_processed += 1
            elif status == "failed":
                self.step_metrics[self.current_step].records_failed += 1
        self._update_display()
    
    def record_api_call(self, success: bool, cached: bool = False):
        """Record an API call."""
        if self.current_step and self.current_step in self.step_metrics:
            metrics = self.step_metrics[self.current_step]
            if cached:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1
                metrics.api_calls_made += 1
                if not success:
                    metrics.api_calls_failed += 1
        self._update_display()
    
    def _create_layout(self) -> Layout:
        """Create the display layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=10),
            Layout(name="symbols", size=10),
            Layout(name="footer", size=3)
        )
        return layout
    
    def _update_display(self):
        """Update the live display."""
        if not self.live_display:
            return
            
        layout = self._create_layout()
        
        # Header
        header = Panel(
            Text("Options Screening Workflow Progress", style="bold cyan", justify="center"),
            border_style="cyan"
        )
        layout["header"].update(header)
        
        # Metrics table
        metrics_table = self._create_metrics_table()
        layout["metrics"].update(Panel(metrics_table, title="Step Metrics", border_style="green"))
        
        # Symbol progress
        symbols_panel = self._create_symbols_panel()
        layout["symbols"].update(Panel(symbols_panel, title="Symbol Processing", border_style="yellow"))
        
        # Footer
        footer = self._create_footer()
        layout["footer"].update(footer)
        
        self.live_display.update(layout)
    
    def _create_metrics_table(self) -> Table:
        """Create metrics table."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Step", style="cyan", width=30)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Duration", style="green", width=10)
        table.add_column("Processed", style="blue", width=10)
        table.add_column("Failed", style="red", width=10)
        table.add_column("API Calls", style="magenta", width=10)
        table.add_column("Cache Hit %", style="cyan", width=12)
        
        for step_name, metrics in self.step_metrics.items():
            status = "🔄 Running" if metrics.completed_at is None else "✅ Complete"
            if self.current_step == step_name and metrics.completed_at is None:
                status = "▶️ Active"
                
            table.add_row(
                step_name,
                status,
                f"{metrics.duration_seconds:.1f}s",
                str(metrics.records_processed),
                str(metrics.records_failed),
                f"{metrics.api_calls_made}/{metrics.api_calls_failed}",
                f"{metrics.cache_hit_rate:.1f}%"
            )
        
        return table
    
    def _create_symbols_panel(self) -> str:
        """Create symbols progress panel."""
        if not self.symbol_progress:
            return "No symbols being processed"
        
        # Show last 10 symbols
        recent_symbols = list(self.symbol_progress.items())[-10:]
        lines = []
        for symbol, status in recent_symbols:
            emoji = {
                "processing": "🔄",
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️"
            }.get(status, "❓")
            lines.append(f"{emoji} {symbol}: {status}")
        
        return "\n".join(lines)
    
    def _create_footer(self) -> Panel:
        """Create footer with summary."""
        total_processed = sum(m.records_processed for m in self.step_metrics.values())
        total_failed = sum(m.records_failed for m in self.step_metrics.values())
        total_api_calls = sum(m.api_calls_made for m in self.step_metrics.values())
        overall_cache_hits = sum(m.cache_hits for m in self.step_metrics.values())
        overall_cache_total = overall_cache_hits + sum(m.cache_misses for m in self.step_metrics.values())
        cache_rate = (overall_cache_hits / overall_cache_total * 100) if overall_cache_total > 0 else 0
        
        summary = (
            f"Total Processed: {total_processed} | "
            f"Failed: {total_failed} | "
            f"API Calls: {total_api_calls} | "
            f"Cache Hit Rate: {cache_rate:.1f}%"
        )
        
        return Panel(Text(summary, style="bold white", justify="center"), border_style="blue")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tracking summary."""
        return {
            "steps_completed": len([m for m in self.step_metrics.values() if m.completed_at]),
            "total_processed": sum(m.records_processed for m in self.step_metrics.values()),
            "total_failed": sum(m.records_failed for m in self.step_metrics.values()),
            "total_api_calls": sum(m.api_calls_made for m in self.step_metrics.values()),
            "cache_hit_rate": sum(m.cache_hits for m in self.step_metrics.values()) / 
                              max(1, sum(m.cache_hits + m.cache_misses for m in self.step_metrics.values())) * 100,
            "step_metrics": {
                name: {
                    "duration": metrics.duration_seconds,
                    "processed": metrics.records_processed,
                    "failed": metrics.records_failed,
                    "success_rate": metrics.success_rate,
                    "cache_hit_rate": metrics.cache_hit_rate
                }
                for name, metrics in self.step_metrics.items()
            }
        }


# Global progress tracker instance
detailed_progress = DetailedProgressTracker()


def track_symbol_processing(symbol: str, operation: str):
    """Decorator to track symbol processing."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            detailed_progress.update_symbol_progress(symbol, "processing")
            try:
                result = await func(*args, **kwargs)
                detailed_progress.update_symbol_progress(symbol, "completed")
                return result
            except Exception as e:
                detailed_progress.update_symbol_progress(symbol, "failed")
                raise
        return wrapper
    return decorator


def track_api_call(provider: str):
    """Decorator to track API calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                # Check if result is from cache (simplified check)
                cached = kwargs.get('from_cache', False)
                result = await func(*args, **kwargs)
                detailed_progress.record_api_call(success=True, cached=cached)
                return result
            except Exception as e:
                detailed_progress.record_api_call(success=False, cached=False)
                raise
        return wrapper
    return decorator