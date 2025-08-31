"""Main application entry point for the options screening application."""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional
import click
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
# Add parent for config
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import (
    setup_logging, 
    get_logger,
    print_results_summary,
    print_info,
    print_error,
    print_success,
    console
)
from src.models import ScreeningCriteria, WorkflowConfig
from src.screener import WorkflowEngine, ScreeningCoordinator
from config.settings import get_settings

# Setup logging first
setup_logging()
logger = get_logger(__name__)


class OptionsScreenerApp:
    """Main application class for options screening."""
    
    def __init__(self):
        self.settings = get_settings()
        self.workflow_engine = WorkflowEngine()
        self.coordinator = None
        
    async def initialize(self):
        """Initialize the application and its components."""
        logger.info("Initializing Options Screening Application")
        
        # Create necessary directories
        self.settings.create_directories()
        
        # Initialize coordinator (will be implemented by specialists)
        # self.coordinator = ScreeningCoordinator(self.workflow_engine)
        
        logger.info("Application initialization completed")
    
    async def run_screening(
        self,
        criteria: Optional[ScreeningCriteria] = None,
        config: Optional[WorkflowConfig] = None
    ):
        """Run the complete options screening workflow."""
        logger.info("Starting options screening workflow")
        
        try:
            # Use default criteria if none provided
            if criteria is None:
                criteria = self._create_default_criteria()
            
            # Use default config if none provided  
            if config is None:
                config = WorkflowConfig(
                    enable_ai_analysis=bool(self.settings.claude_api_key),
                    max_ai_cost_dollars=self.settings.claude_daily_cost_limit
                )
            
            # Execute workflow
            result = await self.workflow_engine.execute_workflow(criteria, config)
            
            logger.info(f"Screening completed: {len(result.qualifying_results)} opportunities found")
            return result
            
        except Exception as e:
            logger.error(f"Screening workflow failed: {e}")
            raise
    
    def _create_default_criteria(self) -> ScreeningCriteria:
        """Create default screening criteria from settings."""
        return ScreeningCriteria(
            specific_symbols=self.settings.symbols_list,  # Use specific symbols if provided
            min_market_cap=self.settings.min_market_cap,
            max_market_cap=self.settings.max_market_cap,
            min_price=self.settings.min_stock_price,
            max_price=self.settings.max_stock_price,
            min_volume=self.settings.min_daily_volume,
            min_option_volume=self.settings.min_option_volume,
            min_open_interest=self.settings.min_open_interest,
            max_days_to_expiration=self.settings.max_days_to_expiration,
            exclude_sectors=self.settings.excluded_sectors,
            exclude_symbols=self.settings.excluded_symbols
        )
    
    async def health_check(self) -> dict:
        """Perform application health check."""
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "components": {}
        }
        
        try:
            # Check configuration
            health_info["components"]["configuration"] = {
                "status": "healthy",
                "api_keys_configured": {
                    "eodhd": bool(self.settings.eodhd_api_key),
                    "marketdata": bool(self.settings.marketdata_api_key), 
                    "claude": bool(self.settings.claude_api_key)
                }
            }
            
            # Check directories
            health_info["components"]["storage"] = {
                "status": "healthy",
                "directories": {
                    "output": self.settings.output_path.exists(),
                    "logs": self.settings.log_path.exists(), 
                    "cache": self.settings.cache_path.exists()
                }
            }
            
            # TODO: Add provider health checks when implemented
            # health_info["components"]["providers"] = await self._check_providers()
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            
        return health_info


# CLI interface
@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config-file', help='Path to configuration file')
@click.pass_context
def cli(ctx, debug, config_file):
    """Options Screening Application CLI."""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['config_file'] = config_file


@cli.command()
@click.option('--symbols', help='Comma-separated list of symbols to screen')
@click.option('--min-price', type=float, help='Minimum stock price')
@click.option('--max-price', type=float, help='Maximum stock price')
@click.option('--enable-ai', is_flag=True, help='Enable AI analysis')
@click.pass_context
async def screen(ctx, symbols, min_price, max_price, enable_ai):
    """Run options screening workflow."""
    app = OptionsScreenerApp()
    await app.initialize()
    
    # Build criteria from CLI options or use defaults from settings
    criteria = app._create_default_criteria()
    
    # Override with CLI options if provided
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        criteria.specific_symbols = symbol_list
        logger.info(f"Screening specific symbols: {symbol_list}")
    elif criteria.specific_symbols:
        logger.info(f"Using symbols from environment: {criteria.specific_symbols}")
    
    if min_price:
        criteria.min_price = min_price
    if max_price:
        criteria.max_price = max_price
    
    # Build config
    config = WorkflowConfig(enable_ai_analysis=enable_ai)
    
    try:
        result = await app.run_screening(criteria, config)
        
        # Display results summary
        if result.qualifying_results:
            print_success(f"Screening completed: {len(result.qualifying_results)} opportunities found")
            
            # Convert results to dict format for display
            results_for_display = []
            for opportunity in result.qualifying_results[:10]:
                results_for_display.append({
                    "symbol": opportunity.symbol,
                    "score": opportunity.overall_score,
                    "price": opportunity.current_price,
                    "option_contract": f"{opportunity.selected_option.strike}C {opportunity.selected_option.expiration}" if opportunity.selected_option else "N/A",
                    "ai_rating": opportunity.ai_rating if hasattr(opportunity, 'ai_rating') else "N/A"
                })
            
            print_results_summary(results_for_display)
            
            # Save results to file
            output_file = Path(app.settings.output_directory) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump([r.dict() for r in result.qualifying_results], f, indent=2, default=str)
            
            print_info(f"Results saved to: {output_file}")
        else:
            print_info("No qualifying opportunities found based on criteria")
                
    except Exception as e:
        print_error(f"Screening failed", exception=e)
        logger.exception("Screening workflow failed")
        sys.exit(1)


@cli.command()
@click.pass_context
async def health(ctx):
    """Check application health status."""
    app = OptionsScreenerApp()
    await app.initialize()
    
    health_info = await app.health_check()
    
    status = health_info["status"]
    click.echo(f"Application Status: {status.upper()}")
    
    for component, info in health_info["components"].items():
        click.echo(f"  {component}: {info['status']}")
    
    if status != "healthy":
        sys.exit(1)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize application configuration and directories."""
    settings = get_settings()
    
    click.echo("Initializing Options Screening Application...")
    
    # Create directories
    settings.create_directories()
    click.echo(f"✓ Created output directory: {settings.output_path}")
    click.echo(f"✓ Created log directory: {settings.log_path}")
    click.echo(f"✓ Created cache directory: {settings.cache_path}")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        example_file = Path(".env.example")
        if example_file.exists():
            click.echo(f"⚠ Please copy {example_file} to {env_file} and configure your API keys")
        else:
            click.echo("⚠ No .env.example file found. Please create a .env file with your API keys")
    else:
        click.echo("✓ Configuration file (.env) exists")
    
    click.echo("\nNext steps:")
    click.echo("1. Configure your API keys in the .env file")
    click.echo("2. Run 'python src/main.py health' to verify configuration")
    click.echo("3. Run 'python src/main.py screen --help' to see screening options")


if __name__ == "__main__":
    # Handle async CLI commands
    import inspect
    
    # Patch click to handle async functions
    original_main = cli.main
    
    def async_main(*args, **kwargs):
        # Get the command function
        ctx = click.get_current_context()
        if ctx.invoked_subcommand:
            cmd = cli.commands[ctx.invoked_subcommand]
            if inspect.iscoroutinefunction(cmd.callback):
                # Run async command
                return asyncio.run(original_main(*args, **kwargs, standalone_mode=False))
        
        return original_main(*args, **kwargs)
    
    cli.main = async_main
    cli()