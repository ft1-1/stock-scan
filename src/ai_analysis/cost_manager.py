"""
Cost Manager for Claude AI Integration

Provides comprehensive cost tracking and management:
- Token counting and estimation
- Daily spending limits enforcement ($50/day default)
- Usage analytics and reporting
- Cost forecasting for batch operations
- Budget alerts and notifications
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math

logger = logging.getLogger(__name__)


@dataclass
class TokenCost:
    """Token usage and cost for a single request"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DailyUsage:
    """Daily usage statistics"""
    date: str
    requests_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    first_request: Optional[datetime] = None
    last_request: Optional[datetime] = None
    
    def add_usage(self, token_cost: TokenCost):
        """Add usage from a request"""
        self.requests_count += 1
        self.total_tokens += token_cost.total_tokens
        self.total_cost += token_cost.total_cost
        self.input_tokens += token_cost.input_tokens
        self.output_tokens += token_cost.output_tokens
        self.input_cost += token_cost.input_cost
        self.output_cost += token_cost.output_cost
        
        if self.first_request is None:
            self.first_request = token_cost.timestamp
        self.last_request = token_cost.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            **asdict(self),
            'first_request': self.first_request.isoformat() if self.first_request else None,
            'last_request': self.last_request.isoformat() if self.last_request else None
        }


@dataclass
class CostConfig:
    """Cost management configuration"""
    daily_limit: float = 50.0
    warning_threshold: float = 0.8  # 80% of daily limit
    model: str = "claude-3-5-sonnet-20241022"
    
    # Claude 3.5 Sonnet pricing (per 1M tokens)
    input_cost_per_million: float = 3.0    # $3 per 1M input tokens
    output_cost_per_million: float = 15.0  # $15 per 1M output tokens
    
    # Token estimation parameters
    chars_per_token: float = 4.0           # Rough estimation
    response_token_estimate: int = 500     # Average expected response size
    
    def calculate_input_cost(self, tokens: int) -> float:
        """Calculate input token cost"""
        return (tokens / 1_000_000) * self.input_cost_per_million
    
    def calculate_output_cost(self, tokens: int) -> float:
        """Calculate output token cost"""
        return (tokens / 1_000_000) * self.output_cost_per_million


class CostManager:
    """
    Comprehensive cost management for Claude AI API.
    
    Features:
    - Token counting and cost calculation
    - Daily spending limits with warnings
    - Usage persistence and analytics
    - Batch cost estimation
    - Budget forecasting
    """
    
    def __init__(
        self,
        config: CostConfig,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize cost manager.
        
        Args:
            config: Cost configuration
            storage_path: Path to persist usage data (optional)
        """
        self.config = config
        self.storage_path = storage_path or Path("data/cost_tracking")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current session tracking
        self.current_session: List[TokenCost] = []
        
        # Daily usage cache
        self._daily_usage_cache: Dict[str, DailyUsage] = {}
        
        # Load existing data
        self._load_usage_history()
        
        logger.info(f"Cost manager initialized with ${config.daily_limit}/day limit")
    
    def estimate_request_cost(
        self,
        prompt: str,
        expected_response_tokens: Optional[int] = None
    ) -> TokenCost:
        """
        Estimate cost for a request before making it.
        
        Args:
            prompt: The prompt to be sent
            expected_response_tokens: Expected response size (uses default if None)
            
        Returns:
            Estimated TokenCost object
        """
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = expected_response_tokens or self.config.response_token_estimate
        total_tokens = input_tokens + output_tokens
        
        input_cost = self.config.calculate_input_cost(input_tokens)
        output_cost = self.config.calculate_output_cost(output_tokens)
        total_cost = input_cost + output_cost
        
        return TokenCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
    
    def record_actual_usage(
        self,
        prompt: str,
        response: str,
        symbol: Optional[str] = None
    ) -> TokenCost:
        """
        Record actual usage after request completion.
        
        Args:
            prompt: The prompt that was sent
            response: The response received
            symbol: Stock symbol for tracking (optional)
            
        Returns:
            Actual TokenCost object
        """
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = self._estimate_tokens(response)
        total_tokens = input_tokens + output_tokens
        
        input_cost = self.config.calculate_input_cost(input_tokens)
        output_cost = self.config.calculate_output_cost(output_tokens)
        total_cost = input_cost + output_cost
        
        token_cost = TokenCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        # Record in current session
        self.current_session.append(token_cost)
        
        # Update daily usage
        today = date.today().isoformat()
        if today not in self._daily_usage_cache:
            self._daily_usage_cache[today] = DailyUsage(date=today)
        
        self._daily_usage_cache[today].add_usage(token_cost)
        
        # Persist usage
        self._save_usage_record(token_cost, symbol)
        
        logger.info(f"Recorded usage: {total_tokens} tokens, ${total_cost:.4f}")
        
        # Check for warnings
        self._check_usage_warnings(today)
        
        return token_cost
    
    def can_afford_request(
        self,
        estimated_cost: float,
        date_override: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if a request can be afforded within daily limits.
        
        Args:
            estimated_cost: Estimated cost of the request
            date_override: Check for specific date (defaults to today)
            
        Returns:
            (can_afford, reason, remaining_budget)
        """
        check_date = date_override or date.today().isoformat()
        
        daily_usage = self._daily_usage_cache.get(
            check_date, 
            DailyUsage(date=check_date)
        )
        
        remaining_budget = self.config.daily_limit - daily_usage.total_cost
        
        if estimated_cost > remaining_budget:
            return (
                False, 
                f"Request cost ${estimated_cost:.4f} exceeds remaining budget ${remaining_budget:.4f}",
                remaining_budget
            )
        
        if daily_usage.total_cost + estimated_cost > self.config.daily_limit:
            return (
                False,
                f"Request would exceed daily limit of ${self.config.daily_limit}",
                remaining_budget
            )
        
        return (True, "OK", remaining_budget)
    
    def get_daily_usage(self, date_str: Optional[str] = None) -> DailyUsage:
        """Get usage statistics for a specific date"""
        check_date = date_str or date.today().isoformat()
        return self._daily_usage_cache.get(check_date, DailyUsage(date=check_date))
    
    def get_usage_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get usage summary for the last N days.
        
        Args:
            days_back: Number of days to include
            
        Returns:
            Usage summary with statistics
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back - 1)
        
        total_cost = 0.0
        total_requests = 0
        total_tokens = 0
        daily_details = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            daily_usage = self._daily_usage_cache.get(date_str, DailyUsage(date=date_str))
            
            total_cost += daily_usage.total_cost
            total_requests += daily_usage.requests_count
            total_tokens += daily_usage.total_tokens
            
            daily_details.append({
                'date': date_str,
                'requests': daily_usage.requests_count,
                'tokens': daily_usage.total_tokens,
                'cost': daily_usage.total_cost,
                'percentage_of_limit': (daily_usage.total_cost / self.config.daily_limit) * 100
            })
            
            current_date += timedelta(days=1)
        
        today_usage = self._daily_usage_cache.get(
            end_date.isoformat(), 
            DailyUsage(date=end_date.isoformat())
        )
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days_back
            },
            'totals': {
                'total_cost': round(total_cost, 4),
                'total_requests': total_requests,
                'total_tokens': total_tokens,
                'average_daily_cost': round(total_cost / days_back, 4),
                'average_cost_per_request': round(total_cost / total_requests, 4) if total_requests > 0 else 0
            },
            'today': {
                'cost': round(today_usage.total_cost, 4),
                'requests': today_usage.requests_count,
                'remaining_budget': round(self.config.daily_limit - today_usage.total_cost, 4),
                'percentage_used': round((today_usage.total_cost / self.config.daily_limit) * 100, 2)
            },
            'daily_breakdown': daily_details,
            'limits': {
                'daily_limit': self.config.daily_limit,
                'warning_threshold': self.config.daily_limit * self.config.warning_threshold
            }
        }
    
    def estimate_batch_cost(
        self,
        prompts: List[str],
        expected_response_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate cost for a batch of requests.
        
        Args:
            prompts: List of prompts to estimate
            expected_response_tokens: Expected response size per request
            
        Returns:
            Batch cost estimation with recommendations
        """
        estimates = []
        total_cost = 0.0
        total_tokens = 0
        
        for i, prompt in enumerate(prompts):
            estimate = self.estimate_request_cost(prompt, expected_response_tokens)
            estimates.append({
                'request_index': i,
                'tokens': estimate.total_tokens,
                'cost': estimate.total_cost
            })
            total_cost += estimate.total_cost
            total_tokens += estimate.total_tokens
        
        # Check affordability
        can_afford, reason, remaining_budget = self.can_afford_request(total_cost)
        
        # Calculate recommendations
        max_affordable = int(remaining_budget / (total_cost / len(prompts))) if prompts else 0
        
        # Estimate processing time (with 60-second intervals)
        processing_time_minutes = (len(prompts) * 1.5) if prompts else 0  # 1.5 min per request with overhead
        
        return {
            'batch_size': len(prompts),
            'total_estimated_cost': round(total_cost, 4),
            'total_estimated_tokens': total_tokens,
            'average_cost_per_request': round(total_cost / len(prompts), 4) if prompts else 0,
            'can_afford_all': can_afford,
            'reason': reason,
            'remaining_budget': round(remaining_budget, 4),
            'max_affordable_requests': max_affordable,
            'estimated_processing_time_minutes': processing_time_minutes,
            'request_estimates': estimates[:5],  # Show first 5 for summary
            'recommendations': self._generate_batch_recommendations(
                len(prompts), total_cost, remaining_budget, can_afford
            )
        }
    
    def _generate_batch_recommendations(
        self,
        batch_size: int,
        total_cost: float,
        remaining_budget: float,
        can_afford: bool
    ) -> List[str]:
        """Generate recommendations for batch processing"""
        recommendations = []
        
        if can_afford:
            recommendations.append("✓ Full batch can be processed within budget")
            if remaining_budget - total_cost < total_cost * 0.5:
                recommendations.append("⚠ Consider monitoring remaining budget closely")
        else:
            affordable_count = int(remaining_budget / (total_cost / batch_size)) if batch_size > 0 else 0
            recommendations.append(f"❌ Batch exceeds budget - can only afford {affordable_count} requests")
            recommendations.append("💡 Consider processing in smaller batches over multiple days")
            
            if affordable_count > 0:
                recommendations.append(f"📅 Process {affordable_count} today, remaining tomorrow")
        
        # Processing time recommendations
        processing_hours = (batch_size * 1.5) / 60  # 1.5 minutes per request
        if processing_hours > 2:
            recommendations.append(f"⏱ Batch will take ~{processing_hours:.1f} hours to complete")
            recommendations.append("🔄 Consider running overnight or during off-hours")
        
        return recommendations
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        if not text:
            return 0
        return max(1, int(len(text) / self.config.chars_per_token))
    
    def _check_usage_warnings(self, date_str: str):
        """Check for usage warnings and log alerts"""
        daily_usage = self._daily_usage_cache.get(date_str)
        if not daily_usage:
            return
        
        percentage_used = daily_usage.total_cost / self.config.daily_limit
        
        if percentage_used >= 1.0:
            logger.error(f"BUDGET EXCEEDED: Used ${daily_usage.total_cost:.2f} of ${self.config.daily_limit} daily limit")
        elif percentage_used >= self.config.warning_threshold:
            logger.warning(f"BUDGET WARNING: Used {percentage_used:.1%} of daily limit (${daily_usage.total_cost:.2f}/${self.config.daily_limit})")
    
    def _load_usage_history(self):
        """Load usage history from persistent storage"""
        usage_file = self.storage_path / "daily_usage.json"
        
        if usage_file.exists():
            try:
                with open(usage_file, 'r') as f:
                    data = json.load(f)
                
                for date_str, usage_data in data.items():
                    # Convert back from dict to DailyUsage object
                    usage = DailyUsage(
                        date=usage_data['date'],
                        requests_count=usage_data['requests_count'],
                        total_tokens=usage_data['total_tokens'],
                        total_cost=usage_data['total_cost'],
                        input_tokens=usage_data.get('input_tokens', 0),
                        output_tokens=usage_data.get('output_tokens', 0),
                        input_cost=usage_data.get('input_cost', 0.0),
                        output_cost=usage_data.get('output_cost', 0.0)
                    )
                    
                    if usage_data.get('first_request'):
                        usage.first_request = datetime.fromisoformat(usage_data['first_request'])
                    if usage_data.get('last_request'):
                        usage.last_request = datetime.fromisoformat(usage_data['last_request'])
                    
                    self._daily_usage_cache[date_str] = usage
                
                logger.info(f"Loaded usage history for {len(data)} days")
                
            except Exception as e:
                logger.error(f"Error loading usage history: {e}")
    
    def _save_usage_record(self, token_cost: TokenCost, symbol: Optional[str] = None):
        """Save individual usage record and update daily totals"""
        # Save individual request record
        request_file = self.storage_path / f"requests_{date.today().isoformat()}.jsonl"
        
        record = {
            **token_cost.to_dict(),
            'symbol': symbol,
            'model': self.config.model
        }
        
        try:
            with open(request_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error saving request record: {e}")
        
        # Update daily usage file
        self._save_daily_usage()
    
    def _save_daily_usage(self):
        """Save daily usage summary to persistent storage"""
        usage_file = self.storage_path / "daily_usage.json"
        
        try:
            # Convert DailyUsage objects to dict for JSON serialization
            data = {}
            for date_str, usage in self._daily_usage_cache.items():
                data[date_str] = usage.to_dict()
            
            with open(usage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving daily usage: {e}")
    
    def reset_daily_usage(self, date_str: Optional[str] = None):
        """Reset usage for a specific date (mainly for testing)"""
        reset_date = date_str or date.today().isoformat()
        self._daily_usage_cache[reset_date] = DailyUsage(date=reset_date)
        self._save_daily_usage()
        logger.info(f"Reset usage for {reset_date}")
    
    def export_usage_report(self, days_back: int = 30) -> str:
        """Export detailed usage report as formatted string"""
        summary = self.get_usage_summary(days_back)
        
        report = f"""
CLAUDE AI COST MANAGEMENT REPORT
Generated: {datetime.now().isoformat()}

SUMMARY ({days_back} days):
• Total Cost: ${summary['totals']['total_cost']}
• Total Requests: {summary['totals']['total_requests']}
• Total Tokens: {summary['totals']['total_tokens']:,}
• Average Daily Cost: ${summary['totals']['average_daily_cost']}
• Average Cost/Request: ${summary['totals']['average_cost_per_request']}

TODAY'S USAGE:
• Cost: ${summary['today']['cost']} / ${self.config.daily_limit}
• Requests: {summary['today']['requests']}
• Budget Used: {summary['today']['percentage_used']}%
• Remaining: ${summary['today']['remaining_budget']}

DAILY BREAKDOWN:
"""
        
        for day in summary['daily_breakdown'][-7:]:  # Show last 7 days
            report += f"• {day['date']}: ${day['cost']:6.2f} ({day['percentage_of_limit']:5.1f}%) - {day['requests']} requests\n"
        
        return report


def create_cost_manager(
    daily_limit: float = 50.0,
    storage_path: Optional[Path] = None
) -> CostManager:
    """
    Factory function to create cost manager with sensible defaults.
    
    Args:
        daily_limit: Daily spending limit in USD
        storage_path: Path for persistent storage
        
    Returns:
        Configured CostManager instance
    """
    config = CostConfig(daily_limit=daily_limit)
    return CostManager(config, storage_path)


if __name__ == "__main__":
    # Example usage and testing
    manager = create_cost_manager(daily_limit=10.0)  # $10 for testing
    
    # Test cost estimation
    sample_prompt = "Analyze this options opportunity..." * 50  # Longer prompt
    estimate = manager.estimate_request_cost(sample_prompt)
    print(f"Estimated cost: ${estimate.total_cost:.4f}")
    print(f"Estimated tokens: {estimate.total_tokens}")
    
    # Test affordability check
    can_afford, reason, remaining = manager.can_afford_request(estimate.total_cost)
    print(f"Can afford: {can_afford} ({reason})")
    print(f"Remaining budget: ${remaining:.4f}")
    
    # Test batch estimation
    prompts = [sample_prompt] * 5
    batch_estimate = manager.estimate_batch_cost(prompts)
    print(f"\nBatch estimate for {len(prompts)} requests:")
    print(f"Total cost: ${batch_estimate['total_estimated_cost']}")
    print(f"Can afford all: {batch_estimate['can_afford_all']}")
    
    for rec in batch_estimate['recommendations']:
        print(f"  {rec}")
    
    # Test usage summary
    summary = manager.get_usage_summary(7)
    print(f"\nUsage summary:")
    print(f"Today's cost: ${summary['today']['cost']}")
    print(f"Budget used: {summary['today']['percentage_used']}%")