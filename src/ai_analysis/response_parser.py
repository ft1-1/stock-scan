"""
Response Parser for Claude AI Analysis

Validates and processes Claude's JSON responses to ensure:
- Proper JSON structure and schema compliance
- Score validation (component scores sum to total)
- Data type validation and sanitization
- Error handling for malformed responses
- Fallback scoring for failed responses
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Valid confidence levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"


class ValidationError(Exception):
    """Raised when response validation fails"""
    pass


@dataclass
class ComponentScores:
    """Component scores from Claude analysis"""
    trend_momentum: int
    options_quality: int
    iv_value: int
    squeeze_volatility: int
    fundamentals: int
    event_news: int
    
    def get_total(self) -> int:
        """Calculate total of all component scores"""
        return (self.trend_momentum + self.options_quality + self.iv_value + 
                self.squeeze_volatility + self.fundamentals + self.event_news)
    
    def validate_ranges(self) -> bool:
        """Validate each component is within expected range"""
        ranges = {
            'trend_momentum': (0, 35),
            'options_quality': (0, 20),
            'iv_value': (0, 15),
            'squeeze_volatility': (0, 10),
            'fundamentals': (0, 10),
            'event_news': (0, 10)
        }
        
        for component, (min_val, max_val) in ranges.items():
            value = getattr(self, component)
            if not (min_val <= value <= max_val):
                return False
        return True


@dataclass 
class OptionRecommendation:
    """Option contract recommendation details"""
    recommendation: str
    entry_timing: str
    risk_management: str


@dataclass
class ClaudeAnalysis:
    """Structured Claude analysis result"""
    symbol: str
    rating: int
    component_scores: ComponentScores
    confidence: ConfidenceLevel
    thesis: str
    opportunities: List[str]
    risks: List[str]
    option_contract: OptionRecommendation
    red_flags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Metadata
    parsed_at: datetime = field(default_factory=datetime.now)
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime and enum to strings
        result['parsed_at'] = self.parsed_at.isoformat()
        result['confidence'] = self.confidence.value
        result['component_scores'] = asdict(self.component_scores)
        result['option_contract'] = asdict(self.option_contract)
        return result


@dataclass
class FallbackAnalysis:
    """Fallback analysis when Claude response fails"""
    symbol: str
    rating: int = 50  # Default neutral rating
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    thesis: str = "Analysis unavailable - using fallback scoring"
    opportunities: List[str] = field(default_factory=lambda: ["Limited analysis available"])
    risks: List[str] = field(default_factory=lambda: ["AI analysis unavailable"])
    option_contract: OptionRecommendation = field(
        default_factory=lambda: OptionRecommendation(
            recommendation="Manual analysis required",
            entry_timing="Cannot determine timing",
            risk_management="Standard risk management applies"
        )
    )
    red_flags: List[str] = field(default_factory=lambda: ["AI analysis failed"])
    notes: str = "Fallback analysis due to AI response failure"
    error_reason: str = "Unknown error"
    
    def to_claude_analysis(self) -> ClaudeAnalysis:
        """Convert to ClaudeAnalysis format"""
        # Create balanced component scores that sum to rating
        total_score = self.rating
        component_scores = ComponentScores(
            trend_momentum=int(total_score * 0.35),
            options_quality=int(total_score * 0.20), 
            iv_value=int(total_score * 0.15),
            squeeze_volatility=int(total_score * 0.10),
            fundamentals=int(total_score * 0.10),
            event_news=int(total_score * 0.10)
        )
        
        return ClaudeAnalysis(
            symbol=self.symbol,
            rating=self.rating,
            component_scores=component_scores,
            confidence=self.confidence,
            thesis=self.thesis,
            opportunities=self.opportunities,
            risks=self.risks,
            option_contract=self.option_contract,
            red_flags=self.red_flags,
            notes=f"{self.notes} (Error: {self.error_reason})",
            validation_errors=[f"Fallback analysis: {self.error_reason}"],
            is_valid=False
        )


class ResponseParser:
    """
    Parser and validator for Claude AI responses.
    
    Handles JSON parsing, schema validation, score validation,
    and fallback analysis generation.
    """
    
    def __init__(self):
        self.required_fields = {
            'symbol', 'rating', 'component_scores', 'confidence',
            'thesis', 'opportunities', 'risks', 'option_contract'
        }
        
        self.component_score_fields = {
            'trend_momentum', 'options_quality', 'iv_value',
            'squeeze_volatility', 'fundamentals', 'event_news'
        }
        
        self.option_contract_fields = {
            'recommendation', 'entry_timing', 'risk_management'
        }
    
    def parse_response(
        self, 
        response_content: str,
        symbol: str,
        original_rating: Optional[float] = None
    ) -> ClaudeAnalysis:
        """
        Parse and validate Claude response.
        
        Args:
            response_content: Raw response from Claude
            symbol: Stock symbol for context
            original_rating: Original quantitative rating for fallback
            
        Returns:
            Validated ClaudeAnalysis object or fallback
        """
        try:
            # Parse JSON
            data = self._parse_json(response_content)
            if not data:
                return self._create_fallback(symbol, "Failed to parse JSON", original_rating)
            
            # Validate structure
            validation_errors = self._validate_structure(data)
            if validation_errors:
                logger.warning(f"Structure validation failed for {symbol}: {validation_errors}")
                return self._create_fallback(symbol, f"Structure validation failed: {validation_errors[0]}", original_rating)
            
            # Create analysis object
            analysis = self._create_analysis_object(data)
            
            # Validate business rules
            business_errors = self._validate_business_rules(analysis)
            if business_errors:
                analysis.validation_errors.extend(business_errors)
                analysis.is_valid = False
                logger.warning(f"Business rule validation failed for {symbol}: {business_errors}")
            
            logger.info(f"Successfully parsed Claude response for {symbol} (rating: {analysis.rating})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing Claude response for {symbol}: {e}")
            return self._create_fallback(symbol, str(e), original_rating)
    
    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON content with multiple strategies"""
        if not content:
            return None
        
        content = content.strip()
        
        # Strategy 1: Try parsing entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON block in content
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Look for ```json code block
        try:
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                if end > start:
                    json_str = content[start:end].strip()
                    return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"Failed to parse JSON from response: {content[:200]}...")
        return None
    
    def _validate_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate JSON structure against schema"""
        errors = []
        
        # Check required top-level fields
        for field in self.required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if 'rating' in data and not isinstance(data['rating'], int):
            errors.append("Rating must be an integer")
        
        if 'symbol' in data and not isinstance(data['symbol'], str):
            errors.append("Symbol must be a string")
        
        if 'confidence' in data and data['confidence'] not in [level.value for level in ConfidenceLevel]:
            errors.append(f"Invalid confidence level: {data['confidence']}")
        
        # Validate component scores
        if 'component_scores' in data:
            component_errors = self._validate_component_scores(data['component_scores'])
            errors.extend(component_errors)
        
        # Validate option contract
        if 'option_contract' in data:
            option_errors = self._validate_option_contract(data['option_contract'])
            errors.extend(option_errors)
        
        # Validate lists
        for list_field in ['opportunities', 'risks', 'red_flags']:
            if list_field in data and not isinstance(data[list_field], list):
                errors.append(f"{list_field} must be a list")
        
        return errors
    
    def _validate_component_scores(self, component_scores: Dict[str, Any]) -> List[str]:
        """Validate component scores structure and values"""
        errors = []
        
        # Check all required component fields exist
        for field in self.component_score_fields:
            if field not in component_scores:
                errors.append(f"Missing component score: {field}")
        
        # Check all values are integers
        for field, value in component_scores.items():
            if not isinstance(value, int):
                errors.append(f"Component score {field} must be an integer")
        
        return errors
    
    def _validate_option_contract(self, option_contract: Dict[str, Any]) -> List[str]:
        """Validate option contract structure"""
        errors = []
        
        for field in self.option_contract_fields:
            if field not in option_contract:
                errors.append(f"Missing option contract field: {field}")
            elif not isinstance(option_contract[field], str):
                errors.append(f"Option contract {field} must be a string")
        
        return errors
    
    def _create_analysis_object(self, data: Dict[str, Any]) -> ClaudeAnalysis:
        """Create ClaudeAnalysis object from validated data"""
        # Create component scores
        component_scores = ComponentScores(
            trend_momentum=data['component_scores']['trend_momentum'],
            options_quality=data['component_scores']['options_quality'],
            iv_value=data['component_scores']['iv_value'],
            squeeze_volatility=data['component_scores']['squeeze_volatility'],
            fundamentals=data['component_scores']['fundamentals'],
            event_news=data['component_scores']['event_news']
        )
        
        # Create option recommendation
        option_contract = OptionRecommendation(
            recommendation=data['option_contract']['recommendation'],
            entry_timing=data['option_contract']['entry_timing'],
            risk_management=data['option_contract']['risk_management']
        )
        
        # Create main analysis object
        return ClaudeAnalysis(
            symbol=data['symbol'],
            rating=data['rating'],
            component_scores=component_scores,
            confidence=ConfidenceLevel(data['confidence']),
            thesis=data['thesis'],
            opportunities=data['opportunities'],
            risks=data['risks'],
            option_contract=option_contract,
            red_flags=data.get('red_flags', []),
            notes=data.get('notes', "")
        )
    
    def _validate_business_rules(self, analysis: ClaudeAnalysis) -> List[str]:
        """Validate business logic and scoring rules"""
        errors = []
        
        # Check rating range
        if not (0 <= analysis.rating <= 100):
            errors.append(f"Rating {analysis.rating} outside valid range 0-100")
        
        # Check component score ranges
        if not analysis.component_scores.validate_ranges():
            errors.append("One or more component scores outside valid ranges")
        
        # Check component scores sum to total (allow small rounding differences)
        component_total = analysis.component_scores.get_total()
        if abs(component_total - analysis.rating) > 2:  # Allow 2-point tolerance
            errors.append(f"Component scores sum to {component_total}, but rating is {analysis.rating}")
        
        # Validate string fields aren't empty
        if not analysis.thesis or len(analysis.thesis.strip()) < 10:
            errors.append("Thesis too short or empty")
        
        if len(analysis.opportunities) == 0:
            errors.append("No opportunities listed")
        
        if len(analysis.risks) == 0:
            errors.append("No risks listed")
        
        # Check for reasonable content length
        if len(analysis.thesis) > 500:
            errors.append("Thesis too long (>500 characters)")
        
        return errors
    
    def _create_fallback(
        self, 
        symbol: str, 
        error_reason: str,
        original_rating: Optional[float] = None
    ) -> ClaudeAnalysis:
        """Create fallback analysis when parsing fails"""
        fallback_rating = 50  # Default neutral
        
        # Use original quantitative rating if available
        if original_rating is not None:
            fallback_rating = max(0, min(100, int(original_rating)))
        
        fallback = FallbackAnalysis(
            symbol=symbol,
            rating=fallback_rating,
            error_reason=error_reason
        )
        
        return fallback.to_claude_analysis()
    
    def validate_and_correct_scores(self, analysis: ClaudeAnalysis) -> ClaudeAnalysis:
        """
        Post-process analysis to correct common score issues.
        
        Adjusts component scores to sum to total rating if needed.
        """
        component_total = analysis.component_scores.get_total()
        
        # If scores don't match, proportionally adjust components
        if abs(component_total - analysis.rating) > 2:
            logger.warning(f"Correcting score mismatch for {analysis.symbol}: "
                         f"{component_total} -> {analysis.rating}")
            
            if component_total > 0:
                # Proportionally scale all components
                scale_factor = analysis.rating / component_total
                
                analysis.component_scores.trend_momentum = int(
                    analysis.component_scores.trend_momentum * scale_factor
                )
                analysis.component_scores.options_quality = int(
                    analysis.component_scores.options_quality * scale_factor
                )
                analysis.component_scores.iv_value = int(
                    analysis.component_scores.iv_value * scale_factor
                )
                analysis.component_scores.squeeze_volatility = int(
                    analysis.component_scores.squeeze_volatility * scale_factor
                )
                analysis.component_scores.fundamentals = int(
                    analysis.component_scores.fundamentals * scale_factor
                )
                analysis.component_scores.event_news = int(
                    analysis.component_scores.event_news * scale_factor
                )
                
                # Handle any remaining difference due to rounding
                new_total = analysis.component_scores.get_total()
                diff = analysis.rating - new_total
                if diff != 0:
                    # Add/subtract difference to largest component
                    max_component = max(
                        ('trend_momentum', analysis.component_scores.trend_momentum),
                        ('options_quality', analysis.component_scores.options_quality),
                        ('iv_value', analysis.component_scores.iv_value),
                        ('squeeze_volatility', analysis.component_scores.squeeze_volatility),
                        ('fundamentals', analysis.component_scores.fundamentals),
                        ('event_news', analysis.component_scores.event_news),
                        key=lambda x: x[1]
                    )[0]
                    
                    current_value = getattr(analysis.component_scores, max_component)
                    setattr(analysis.component_scores, max_component, current_value + diff)
            
            analysis.validation_errors.append("Adjusted component scores to match total rating")
        
        return analysis


def parse_claude_response(
    response_content: str,
    symbol: str, 
    original_rating: Optional[float] = None
) -> ClaudeAnalysis:
    """
    Convenience function to parse Claude response.
    
    Args:
        response_content: Raw Claude response
        symbol: Stock symbol  
        original_rating: Original quantitative rating for fallback
        
    Returns:
        Validated ClaudeAnalysis object
    """
    parser = ResponseParser()
    analysis = parser.parse_response(response_content, symbol, original_rating)
    return parser.validate_and_correct_scores(analysis)


if __name__ == "__main__":
    # Test with sample response
    sample_response = '''
    {
        "symbol": "AAPL",
        "rating": 85,
        "component_scores": {
            "trend_momentum": 28,
            "options_quality": 18,
            "iv_value": 12,
            "squeeze_volatility": 8,
            "fundamentals": 9,
            "event_news": 10
        },
        "confidence": "high",
        "thesis": "Strong technical momentum with favorable IV conditions for options strategies",
        "opportunities": [
            "Strong uptrend with momentum acceleration",
            "IV in attractive range for premium collection",
            "Good liquidity in option chains"
        ],
        "risks": [
            "General market volatility risk",
            "Earnings announcement in 3 weeks",
            "Tech sector rotation concerns"
        ],
        "option_contract": {
            "recommendation": "Consider AAPL 180C expiring in 30-45 days",
            "entry_timing": "Current levels offer good entry point",
            "risk_management": "Position size appropriately for volatility"
        },
        "red_flags": [],
        "notes": "Solid opportunity with good risk/reward profile"
    }
    '''
    
    parser = ResponseParser()
    analysis = parser.parse_response(sample_response, "AAPL")
    
    print(f"Parsed successfully: {analysis.is_valid}")
    print(f"Rating: {analysis.rating}")
    print(f"Component total: {analysis.component_scores.get_total()}")
    print(f"Validation errors: {len(analysis.validation_errors)}")
    
    if analysis.validation_errors:
        for error in analysis.validation_errors:
            print(f"  - {error}")
    
    print(f"Analysis JSON length: {len(json.dumps(analysis.to_dict(), indent=2))}")