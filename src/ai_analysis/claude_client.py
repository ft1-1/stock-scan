"""
Claude API Client for Options Screening AI Integration

Implements:
- 60-second rate limiting between requests
- Exponential backoff retry logic  
- Daily cost limits and monitoring
- Token counting and estimation
- Structured error handling
"""

import asyncio
import logging
import time
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message
    from anthropic._exceptions import (
        APIError, 
        RateLimitError, 
        APIStatusError,
        APIConnectionError
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Mock classes for development without anthropic SDK
    class Anthropic: pass
    class AsyncAnthropic: pass
    class Message: pass
    class APIError(Exception): pass
    class RateLimitError(Exception): pass
    class APIStatusError(Exception): pass
    class APIConnectionError(Exception): pass

logger = logging.getLogger(__name__)


@dataclass
class ClaudeConfig:
    """Configuration for Claude API client"""
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout_seconds: int = 60
    
    # Rate limiting
    min_request_interval: float = 60.0  # 60 seconds between requests
    
    # Cost controls
    daily_cost_limit: float = 50.0  # $50 per day
    max_requests_per_hour: int = 50
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_max_delay: float = 300.0  # 5 minutes max


@dataclass
class ClaudeResponse:
    """Standardized Claude response object"""
    success: bool
    content: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    response_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return asdict(self)


@dataclass
class UsageStats:
    """Track Claude API usage statistics"""
    requests_today: int = 0
    tokens_used_today: int = 0
    cost_today: float = 0.0
    last_request_time: Optional[float] = None
    requests_this_hour: int = 0
    hour_start_time: Optional[float] = None
    last_reset_date: Optional[str] = None


class RateLimitError(Exception):
    """Custom rate limit exception"""
    pass


class CostLimitExceededError(Exception):
    """Raised when daily cost limit is exceeded"""
    pass


class ClaudeClient:
    """
    Claude API client with comprehensive rate limiting and cost controls.
    
    Features:
    - 60-second minimum intervals between requests
    - Daily cost monitoring and limits
    - Exponential backoff retry logic
    - Token counting and cost estimation
    - Usage statistics tracking
    """
    
    def __init__(self, config: ClaudeConfig):
        """
        Initialize Claude client.
        
        Args:
            config: Claude configuration object
        """
        self.config = config
        self.usage_stats = UsageStats()
        self.usage_stats.last_reset_date = date.today().isoformat()
        
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic SDK not available - Claude client will not function")
            self.client = None
            return
        
        # Initialize Anthropic client
        try:
            self.client = Anthropic(
                api_key=config.api_key,
                timeout=config.timeout_seconds
            )
            logger.info("Claude client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self.client = None
    
    async def analyze_opportunity(
        self, 
        prompt: str,
        opportunity_symbol: str
    ) -> ClaudeResponse:
        """
        Analyze single opportunity with Claude AI.
        
        Args:
            prompt: Formatted analysis prompt
            opportunity_symbol: Stock symbol for logging
            
        Returns:
            ClaudeResponse object with results or error information
        """
        if not self.client:
            return ClaudeResponse(
                success=False,
                error="Claude client not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Check rate limits and cost controls
            self._check_rate_limits()
            self._check_cost_limits(prompt)
            
            # Wait for rate limit interval if needed
            await self._wait_for_rate_limit()
            
            logger.info(f"Analyzing opportunity {opportunity_symbol} with Claude")
            
            # Make API request with retry logic
            response = await self._make_request_with_retry(prompt)
            
            # Process response
            if response:
                # Extract content string from response
                content_str = self._extract_content_string(response)
                
                tokens_used = self._estimate_tokens_used(prompt, content_str)
                cost_estimate = self._calculate_cost(tokens_used)
                
                # Update usage statistics
                self._update_usage_stats(tokens_used, cost_estimate)
                
                # Parse JSON response
                parsed_json = self._parse_json_response(content_str)
                
                claude_response = ClaudeResponse(
                    success=True,
                    content=content_str,
                    parsed_json=parsed_json,
                    tokens_used=tokens_used,
                    cost_estimate=cost_estimate,
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Successfully analyzed {opportunity_symbol} "
                           f"(tokens: {tokens_used}, cost: ${cost_estimate:.4f})")
                
                return claude_response
            else:
                return ClaudeResponse(
                    success=False,
                    error="No response received from Claude",
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
                
        except CostLimitExceededError as e:
            logger.error(f"Cost limit exceeded for {opportunity_symbol}: {e}")
            return ClaudeResponse(
                success=False,
                error=f"Daily cost limit exceeded: {e}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for {opportunity_symbol}: {e}")
            return ClaudeResponse(
                success=False,
                error=f"Rate limit exceeded: {e}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {opportunity_symbol}: {e}")
            return ClaudeResponse(
                success=False,
                error=str(e),
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def _make_request_with_retry(self, prompt: str) -> Optional[Message]:
        """Make API request with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Create message
                message = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                return message
                
            except RateLimitError as e:
                last_exception = e
                wait_time = min(
                    self.config.retry_backoff_factor ** attempt * 30,  # Start with 30s
                    self.config.retry_max_delay
                )
                logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                continue
                
            except (APIConnectionError, APIStatusError) as e:
                last_exception = e
                wait_time = min(
                    self.config.retry_backoff_factor ** attempt * 10,  # Start with 10s
                    self.config.retry_max_delay
                )
                logger.warning(f"API error, retrying in {wait_time}s (attempt {attempt + 1}): {e}")
                await asyncio.sleep(wait_time)
                continue
                
            except APIError as e:
                # Don't retry on general API errors
                logger.error(f"Non-retryable API error: {e}")
                raise e
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retries exhausted without specific error")
    
    def _check_rate_limits(self):
        """Check if rate limits allow new request"""
        current_time = time.time()
        
        # Reset hourly counter if needed
        if (self.usage_stats.hour_start_time is None or 
            current_time - self.usage_stats.hour_start_time >= 3600):
            self.usage_stats.requests_this_hour = 0
            self.usage_stats.hour_start_time = current_time
        
        # Check hourly limit
        if self.usage_stats.requests_this_hour >= self.config.max_requests_per_hour:
            raise RateLimitError("Hourly request limit exceeded")
    
    def _check_cost_limits(self, prompt: str):
        """Check if daily cost limit allows new request"""
        # Reset daily stats if needed
        today = date.today().isoformat()
        if self.usage_stats.last_reset_date != today:
            self._reset_daily_stats()
        
        # Estimate cost of this request
        estimated_tokens = self._estimate_input_tokens(prompt)
        estimated_cost = self._calculate_cost(estimated_tokens)
        
        if (self.usage_stats.cost_today + estimated_cost) > self.config.daily_cost_limit:
            raise CostLimitExceededError(
                f"Request would exceed daily cost limit "
                f"(${self.usage_stats.cost_today:.2f} + ${estimated_cost:.2f} > "
                f"${self.config.daily_cost_limit})"
            )
    
    async def _wait_for_rate_limit(self):
        """Wait for minimum interval between requests"""
        if self.usage_stats.last_request_time:
            time_since_last = time.time() - self.usage_stats.last_request_time
            if time_since_last < self.config.min_request_interval:
                wait_time = self.config.min_request_interval - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        self.usage_stats.last_request_time = time.time()
    
    def _extract_content_string(self, response) -> str:
        """Extract content string from Claude response."""
        try:
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
                
                # If content is a list (new API format)
                if isinstance(content, list):
                    # Look for text content blocks
                    text_blocks = []
                    for block in content:
                        if hasattr(block, 'text'):
                            text_blocks.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            text_blocks.append(block['text'])
                        elif isinstance(block, str):
                            text_blocks.append(block)
                    return '\n'.join(text_blocks)
                
                # If content is already a string
                elif isinstance(content, str):
                    return content
                    
                # Fallback: try to get text attribute or convert to string
                else:
                    if hasattr(content, 'text'):
                        return content.text
                    else:
                        return str(content)
            else:
                # Fallback for unexpected response structure
                return str(response)
                
        except Exception as e:
            logger.error(f"Error extracting content from response: {e}")
            return str(response)
    
    def _update_usage_stats(self, tokens_used: int, cost_estimate: float):
        """Update usage statistics"""
        self.usage_stats.requests_today += 1
        self.usage_stats.tokens_used_today += tokens_used
        self.usage_stats.cost_today += cost_estimate
        self.usage_stats.requests_this_hour += 1
    
    def _reset_daily_stats(self):
        """Reset daily usage statistics"""
        today = date.today().isoformat()
        logger.info(f"Resetting daily usage stats for {today}")
        
        self.usage_stats.requests_today = 0
        self.usage_stats.tokens_used_today = 0
        self.usage_stats.cost_today = 0.0
        self.usage_stats.last_reset_date = today
    
    def _estimate_input_tokens(self, prompt: str) -> int:
        """Estimate input tokens for cost calculation"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(prompt) // 4
    
    def _estimate_tokens_used(self, prompt: str, response: str) -> int:
        """Estimate total tokens used (input + output)"""
        input_tokens = self._estimate_input_tokens(prompt)
        output_tokens = len(response) // 4  # Same estimation for output
        return input_tokens + output_tokens
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """
        Calculate cost based on token usage.
        
        Claude 3.5 Sonnet pricing (as of 2024):
        - Input: $3 per 1M tokens
        - Output: $15 per 1M tokens
        
        Using conservative estimate: $10 per 1M tokens average
        """
        return (tokens_used / 1_000_000) * 10.0
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from Claude"""
        if not content:
            return None
        
        try:
            # Look for JSON content - Claude might include explanation text
            content = content.strip()
            
            # Find JSON block
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)
            else:
                # Try parsing entire content as JSON
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "requests_today": self.usage_stats.requests_today,
            "tokens_used_today": self.usage_stats.tokens_used_today,
            "cost_today": round(self.usage_stats.cost_today, 4),
            "daily_limit": self.config.daily_cost_limit,
            "remaining_budget": round(
                self.config.daily_cost_limit - self.usage_stats.cost_today, 4
            ),
            "requests_this_hour": self.usage_stats.requests_this_hour,
            "hourly_limit": self.config.max_requests_per_hour,
            "last_request_ago": (
                time.time() - self.usage_stats.last_request_time
                if self.usage_stats.last_request_time else None
            ),
            "next_request_available": (
                self.config.min_request_interval - (time.time() - self.usage_stats.last_request_time)
                if self.usage_stats.last_request_time else 0
            )
        }
    
    def can_make_request(self) -> tuple[bool, str]:
        """
        Check if a request can be made right now.
        
        Returns:
            (can_make_request, reason_if_not)
        """
        try:
            self._check_rate_limits()
            
            # Check daily cost (with small buffer for estimation errors)
            if self.usage_stats.cost_today >= (self.config.daily_cost_limit * 0.95):
                return False, "Daily cost limit nearly reached"
            
            # Check if we need to wait for rate limit
            if self.usage_stats.last_request_time:
                time_since_last = time.time() - self.usage_stats.last_request_time
                if time_since_last < self.config.min_request_interval:
                    return False, f"Rate limit: {self.config.min_request_interval - time_since_last:.1f}s remaining"
            
            return True, "OK"
            
        except RateLimitError as e:
            return False, str(e)
        except CostLimitExceededError as e:
            return False, str(e)
    
    def estimate_request_cost(self, prompt: str) -> float:
        """Estimate cost of a request without making it"""
        estimated_tokens = self._estimate_input_tokens(prompt) + 500  # Add estimated response
        return self._calculate_cost(estimated_tokens)


class MockClaudeClient(ClaudeClient):
    """
    Mock Claude client for testing and development when API key is not available.
    
    Returns realistic-looking mock responses for testing the integration.
    """
    
    def __init__(self, config: ClaudeConfig):
        """Initialize mock client"""
        self.config = config
        self.usage_stats = UsageStats()
        self.usage_stats.last_reset_date = date.today().isoformat()
        logger.info("Mock Claude client initialized (no API calls will be made)")
    
    async def analyze_opportunity(
        self, 
        prompt: str,
        opportunity_symbol: str
    ) -> ClaudeResponse:
        """Return mock analysis response"""
        # Simulate processing time
        await asyncio.sleep(2.0)
        
        # Create mock response
        mock_analysis = {
            "symbol": opportunity_symbol,
            "rating": 75 + hash(opportunity_symbol) % 25,  # Consistent but varied ratings
            "confidence": "high",
            "thesis": f"Mock analysis for {opportunity_symbol} showing moderate bullish outlook",
            "opportunities": [
                "Strong technical setup",
                "Favorable volatility environment",
                "Good liquidity conditions"
            ],
            "risks": [
                "Market volatility risk",
                "Earnings event proximity",
                "General market conditions"
            ],
            "option_contract": {
                "recommendation": f"Consider {opportunity_symbol} call options",
                "entry_timing": "Current levels attractive",
                "risk_management": "Standard position sizing recommended"
            },
            "red_flags": [],
            "notes": "Mock analysis - not real Claude output"
        }
        
        mock_response = ClaudeResponse(
            success=True,
            content=json.dumps(mock_analysis, indent=2),
            parsed_json=mock_analysis,
            tokens_used=800,
            cost_estimate=0.008,
            response_time=2.0,
            timestamp=datetime.now()
        )
        
        # Update mock usage stats
        self._update_usage_stats(800, 0.008)
        
        logger.info(f"Mock analysis completed for {opportunity_symbol}")
        return mock_response


def create_claude_client(
    api_key: str,
    daily_cost_limit: float = 50.0,
    use_mock: bool = False
) -> ClaudeClient:
    """
    Factory function to create Claude client with sensible defaults.
    
    Args:
        api_key: Anthropic API key
        daily_cost_limit: Daily spending limit in USD
        use_mock: Use mock client for testing
        
    Returns:
        Configured Claude client
    """
    config = ClaudeConfig(
        api_key=api_key,
        daily_cost_limit=daily_cost_limit
    )
    
    if use_mock or not api_key or not ANTHROPIC_AVAILABLE:
        logger.info("Creating mock Claude client")
        return MockClaudeClient(config)
    else:
        logger.info("Creating live Claude client")
        return ClaudeClient(config)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_client():
        """Test Claude client functionality"""
        client = create_claude_client("test-key", use_mock=True)
        
        # Test usage stats
        stats = client.get_usage_stats()
        print("Initial usage stats:", json.dumps(stats, indent=2))
        
        # Test request check
        can_request, reason = client.can_make_request()
        print(f"Can make request: {can_request} ({reason})")
        
        # Test analysis
        test_prompt = "Analyze AAPL options opportunity..."
        response = await client.analyze_opportunity(test_prompt, "AAPL")
        
        print(f"Analysis success: {response.success}")
        if response.success:
            print(f"Rating: {response.parsed_json.get('rating')}")
        else:
            print(f"Error: {response.error}")
        
        # Check updated stats
        stats = client.get_usage_stats()
        print("Updated usage stats:", json.dumps(stats, indent=2))
    
    asyncio.run(test_client())