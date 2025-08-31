"""
Persistence Layer for Claude AI Analysis

Provides comprehensive data persistence for:
- Request/response pairs with full metadata
- Analysis results and audit trails
- Performance metrics and debugging data
- Data package storage for replay/analysis
- Error tracking and resolution history
"""

import json
import gzip
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import uuid

from .response_parser import ClaudeAnalysis
from .cost_manager import TokenCost

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Complete record of an AI analysis with all metadata"""
    # Identifiers
    record_id: str
    symbol: str
    timestamp: datetime
    
    # Input data
    data_package: Dict[str, Any]
    prompt: str
    data_package_hash: str
    
    # API interaction
    request_sent_at: Optional[datetime] = None
    response_received_at: Optional[datetime] = None
    response_time_seconds: Optional[float] = None
    
    # Response data
    raw_response: Optional[str] = None
    parsed_analysis: Optional[Dict[str, Any]] = None  # ClaudeAnalysis.to_dict()
    parsing_success: bool = False
    validation_errors: List[str] = None
    
    # Cost tracking
    token_cost: Optional[Dict[str, Any]] = None  # TokenCost.to_dict()
    
    # Performance metadata
    data_completeness_score: float = 0.0
    analysis_confidence: Optional[str] = None
    final_rating: Optional[int] = None
    
    # Error handling
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Processing metadata
    claude_model: Optional[str] = None
    rate_limit_wait_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'request_sent_at': self.request_sent_at.isoformat() if self.request_sent_at else None,
            'response_received_at': self.response_received_at.isoformat() if self.response_received_at else None
        }


@dataclass
class BatchAnalysisSession:
    """Records for batch analysis operations"""
    session_id: str
    started_at: datetime
    symbols: List[str]
    total_opportunities: int
    
    # Progress tracking
    completed_count: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    
    # Cost tracking
    total_cost: float = 0.0
    total_tokens: int = 0
    
    # Session metadata
    completed_at: Optional[datetime] = None
    session_duration_seconds: Optional[float] = None
    average_analysis_time: Optional[float] = None
    
    # Configuration
    daily_cost_limit: float = 50.0
    rate_limit_seconds: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class AnalysisPersistence:
    """
    Comprehensive persistence layer for Claude AI analysis.
    
    Handles:
    - Individual analysis records with full metadata
    - Batch session tracking
    - Data compression and efficient storage
    - Query and retrieval functionality
    - Export and reporting capabilities
    """
    
    def __init__(self, storage_path: Optional[Path] = None, compress_data: bool = True):
        """
        Initialize persistence layer.
        
        Args:
            storage_path: Base directory for data storage
            compress_data: Whether to compress stored data
        """
        self.storage_path = storage_path or Path("data/ai_analysis")
        self.compress_data = compress_data
        
        # Create directory structure
        self._create_directories()
        
        # Current session tracking
        self.current_session: Optional[BatchAnalysisSession] = None
        
        logger.info(f"Analysis persistence initialized at {self.storage_path}")
    
    def _create_directories(self):
        """Create required directory structure"""
        directories = [
            "daily_records",    # Daily analysis records
            "data_packages",    # Original data packages
            "sessions",         # Batch session records
            "errors",           # Error logs and debugging
            "exports"           # Report exports
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def save_analysis_record(
        self,
        symbol: str,
        data_package: Dict[str, Any],
        prompt: str,
        raw_response: Optional[str] = None,
        parsed_analysis: Optional[ClaudeAnalysis] = None,
        token_cost: Optional[TokenCost] = None,
        error_info: Optional[tuple] = None,
        performance_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete analysis record.
        
        Args:
            symbol: Stock symbol
            data_package: Original data package sent to Claude
            prompt: Generated prompt
            raw_response: Raw Claude response
            parsed_analysis: Parsed analysis result
            token_cost: Token usage and cost information
            error_info: (error_type, error_message) if error occurred
            performance_metadata: Additional performance data
            
        Returns:
            Unique record ID
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create data package hash for deduplication
        package_str = json.dumps(data_package, sort_keys=True, default=str)
        data_hash = hashlib.sha256(package_str.encode()).hexdigest()[:16]
        
        # Create analysis record
        record = AnalysisRecord(
            record_id=record_id,
            symbol=symbol,
            timestamp=timestamp,
            data_package=data_package,
            prompt=prompt,
            data_package_hash=data_hash,
            raw_response=raw_response,
            data_completeness_score=data_package.get('data_quality', {}).get('overall_score', 0.0)
        )
        
        # Add parsed analysis data
        if parsed_analysis:
            record.parsed_analysis = parsed_analysis.to_dict()
            record.parsing_success = True
            record.analysis_confidence = parsed_analysis.confidence.value
            record.final_rating = parsed_analysis.rating
            record.validation_errors = parsed_analysis.validation_errors
        
        # Add cost information
        if token_cost:
            record.token_cost = token_cost.to_dict()
        
        # Add error information
        if error_info:
            error_type, error_message = error_info
            record.error_occurred = True
            record.error_type = error_type
            record.error_message = error_message
        
        # Add performance metadata
        if performance_metadata:
            for key, value in performance_metadata.items():
                if hasattr(record, key):
                    setattr(record, key, value)
        
        # Update session tracking
        if self.current_session:
            self.current_session.completed_count += 1
            if parsed_analysis and parsed_analysis.is_valid:
                self.current_session.successful_analyses += 1
            else:
                self.current_session.failed_analyses += 1
            
            if token_cost:
                self.current_session.total_cost += token_cost.total_cost
                self.current_session.total_tokens += token_cost.total_tokens
        
        # Save to storage
        self._save_record_to_storage(record)
        
        logger.info(f"Saved analysis record {record_id} for {symbol}")
        return record_id
    
    def start_batch_session(
        self,
        symbols: List[str],
        session_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new batch analysis session.
        
        Args:
            symbols: List of symbols to be analyzed
            session_config: Configuration for the session
            
        Returns:
            Session ID
        """
        session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.current_session = BatchAnalysisSession(
            session_id=session_id,
            started_at=datetime.now(),
            symbols=symbols,
            total_opportunities=len(symbols)
        )
        
        # Apply configuration if provided
        if session_config:
            for key, value in session_config.items():
                if hasattr(self.current_session, key):
                    setattr(self.current_session, key, value)
        
        logger.info(f"Started batch session {session_id} with {len(symbols)} symbols")
        return session_id
    
    def finish_batch_session(self) -> Optional[Dict[str, Any]]:
        """
        Finish current batch session and return summary.
        
        Returns:
            Session summary dictionary
        """
        if not self.current_session:
            logger.warning("No active batch session to finish")
            return None
        
        # Finalize session
        self.current_session.completed_at = datetime.now()
        self.current_session.session_duration_seconds = (
            self.current_session.completed_at - self.current_session.started_at
        ).total_seconds()
        
        if self.current_session.completed_count > 0:
            self.current_session.average_analysis_time = (
                self.current_session.session_duration_seconds / self.current_session.completed_count
            )
        
        # Save session record
        session_file = self.storage_path / "sessions" / f"{self.current_session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2, default=str)
        
        session_summary = self.current_session.to_dict()
        
        logger.info(f"Finished batch session {self.current_session.session_id}: "
                   f"{self.current_session.successful_analyses}/{self.current_session.total_opportunities} successful")
        
        self.current_session = None
        return session_summary
    
    def get_analysis_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis record by ID"""
        # Search through daily record files
        for daily_file in (self.storage_path / "daily_records").glob("*.jsonl*"):
            try:
                records = self._read_jsonl_file(daily_file)
                for record in records:
                    if record.get('record_id') == record_id:
                        return record
            except Exception as e:
                logger.warning(f"Error reading {daily_file}: {e}")
        
        return None
    
    def get_analyses_for_symbol(
        self,
        symbol: str,
        days_back: int = 30,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get analysis records for a specific symbol.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to search back
            limit: Maximum number of records to return
            
        Returns:
            List of analysis records, newest first
        """
        records = []
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        current_date = start_date
        while current_date <= end_date:
            daily_file = self._get_daily_file_path(current_date)
            if daily_file.exists():
                try:
                    daily_records = self._read_jsonl_file(daily_file)
                    symbol_records = [r for r in daily_records if r.get('symbol') == symbol]
                    records.extend(symbol_records)
                except Exception as e:
                    logger.warning(f"Error reading {daily_file}: {e}")
            
            current_date += timedelta(days=1)
        
        # Sort by timestamp (newest first)
        records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        if limit:
            records = records[:limit]
        
        return records
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get summary of analysis activity for a specific date.
        
        Args:
            target_date: Date to summarize (defaults to today)
            
        Returns:
            Daily activity summary
        """
        check_date = target_date or date.today()
        daily_file = self._get_daily_file_path(check_date)
        
        summary = {
            'date': check_date.isoformat(),
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'unique_symbols': set(),
            'total_cost': 0.0,
            'total_tokens': 0,
            'error_types': {},
            'rating_distribution': {'0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0},
            'confidence_distribution': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        if not daily_file.exists():
            return summary
        
        try:
            records = self._read_jsonl_file(daily_file)
            
            for record in records:
                summary['total_analyses'] += 1
                summary['unique_symbols'].add(record.get('symbol', 'unknown'))
                
                if record.get('parsing_success'):
                    summary['successful_analyses'] += 1
                else:
                    summary['failed_analyses'] += 1
                
                # Cost tracking
                if record.get('token_cost'):
                    summary['total_cost'] += record['token_cost'].get('total_cost', 0)
                    summary['total_tokens'] += record['token_cost'].get('total_tokens', 0)
                
                # Error tracking
                if record.get('error_occurred'):
                    error_type = record.get('error_type', 'unknown')
                    summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
                
                # Rating distribution
                if record.get('final_rating') is not None:
                    rating = record['final_rating']
                    if rating <= 20:
                        summary['rating_distribution']['0-20'] += 1
                    elif rating <= 40:
                        summary['rating_distribution']['21-40'] += 1
                    elif rating <= 60:
                        summary['rating_distribution']['41-60'] += 1
                    elif rating <= 80:
                        summary['rating_distribution']['61-80'] += 1
                    else:
                        summary['rating_distribution']['81-100'] += 1
                
                # Confidence distribution
                confidence = record.get('analysis_confidence')
                if confidence in summary['confidence_distribution']:
                    summary['confidence_distribution'][confidence] += 1
            
            summary['unique_symbols'] = len(summary['unique_symbols'])
            
        except Exception as e:
            logger.error(f"Error generating daily summary for {check_date}: {e}")
        
        return summary
    
    def export_analysis_report(
        self,
        start_date: date,
        end_date: date,
        export_format: str = "json"
    ) -> Path:
        """
        Export comprehensive analysis report for date range.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            export_format: Export format ("json", "csv", "txt")
            
        Returns:
            Path to exported report file
        """
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_days': (end_date - start_date).days + 1
            },
            'daily_summaries': [],
            'overall_statistics': {
                'total_analyses': 0,
                'total_cost': 0.0,
                'total_tokens': 0,
                'success_rate': 0.0,
                'average_rating': 0.0
            }
        }
        
        # Collect daily summaries
        current_date = start_date
        ratings = []
        
        while current_date <= end_date:
            daily_summary = self.get_daily_summary(current_date)
            report_data['daily_summaries'].append(daily_summary)
            
            # Aggregate statistics
            report_data['overall_statistics']['total_analyses'] += daily_summary['total_analyses']
            report_data['overall_statistics']['total_cost'] += daily_summary['total_cost']
            report_data['overall_statistics']['total_tokens'] += daily_summary['total_tokens']
            
            # Collect ratings for average calculation
            daily_file = self._get_daily_file_path(current_date)
            if daily_file.exists():
                try:
                    records = self._read_jsonl_file(daily_file)
                    ratings.extend([r['final_rating'] for r in records if r.get('final_rating') is not None])
                except Exception:
                    pass
            
            current_date += timedelta(days=1)
        
        # Calculate derived statistics
        total_analyses = report_data['overall_statistics']['total_analyses']
        if total_analyses > 0:
            successful = sum(d['successful_analyses'] for d in report_data['daily_summaries'])
            report_data['overall_statistics']['success_rate'] = (successful / total_analyses) * 100
        
        if ratings:
            report_data['overall_statistics']['average_rating'] = sum(ratings) / len(ratings)
        
        # Export in requested format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"analysis_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{timestamp}"
        
        if export_format == "json":
            export_path = self.storage_path / "exports" / f"{filename_base}.json"
            with open(export_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        elif export_format == "txt":
            export_path = self.storage_path / "exports" / f"{filename_base}.txt"
            with open(export_path, 'w') as f:
                f.write(self._format_text_report(report_data))
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Exported analysis report to {export_path}")
        return export_path
    
    def _save_record_to_storage(self, record: AnalysisRecord):
        """Save analysis record to appropriate daily file"""
        daily_file = self._get_daily_file_path(record.timestamp.date())
        
        # Convert record to dictionary
        record_dict = record.to_dict()
        
        # Append to daily file
        if self.compress_data and daily_file.suffix != '.gz':
            daily_file = daily_file.with_suffix(daily_file.suffix + '.gz')
            
        try:
            if self.compress_data:
                with gzip.open(daily_file, 'at') as f:
                    f.write(json.dumps(record_dict, default=str) + '\n')
            else:
                with open(daily_file, 'a') as f:
                    f.write(json.dumps(record_dict, default=str) + '\n')
        except Exception as e:
            logger.error(f"Error saving record to {daily_file}: {e}")
            raise
    
    def _get_daily_file_path(self, target_date: date) -> Path:
        """Get file path for daily records"""
        filename = f"analyses_{target_date.isoformat()}.jsonl"
        if self.compress_data:
            filename += ".gz"
        return self.storage_path / "daily_records" / filename
    
    def _read_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSONL file (compressed or uncompressed)"""
        records = []
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {e}")
            raise
        
        return records
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as readable text"""
        lines = [
            "CLAUDE AI ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {report_data['report_metadata']['generated_at']}",
            f"Period: {report_data['report_metadata']['start_date']} to {report_data['report_metadata']['end_date']}",
            f"Total Days: {report_data['report_metadata']['total_days']}",
            "",
            "OVERALL STATISTICS:",
            f"• Total Analyses: {report_data['overall_statistics']['total_analyses']}",
            f"• Total Cost: ${report_data['overall_statistics']['total_cost']:.2f}",
            f"• Total Tokens: {report_data['overall_statistics']['total_tokens']:,}",
            f"• Success Rate: {report_data['overall_statistics']['success_rate']:.1f}%",
            f"• Average Rating: {report_data['overall_statistics']['average_rating']:.1f}",
            "",
            "DAILY BREAKDOWN:"
        ]
        
        for daily in report_data['daily_summaries']:
            if daily['total_analyses'] > 0:
                lines.append(
                    f"• {daily['date']}: {daily['total_analyses']} analyses, "
                    f"{daily['successful_analyses']} successful, "
                    f"${daily['total_cost']:.2f} cost"
                )
        
        return "\n".join(lines)


# Convenience functions

def create_persistence_manager(
    storage_path: Optional[Path] = None,
    compress_data: bool = True
) -> AnalysisPersistence:
    """
    Factory function to create persistence manager.
    
    Args:
        storage_path: Storage directory path
        compress_data: Whether to compress stored data
        
    Returns:
        Configured AnalysisPersistence instance
    """
    return AnalysisPersistence(storage_path, compress_data)


if __name__ == "__main__":
    # Example usage and testing
    from datetime import timedelta
    
    persistence = create_persistence_manager()
    
    # Test session tracking
    session_id = persistence.start_batch_session(["AAPL", "MSFT", "GOOGL"])
    print(f"Started session: {session_id}")
    
    # Test record saving
    sample_package = {
        "metadata": {"symbol": "AAPL"},
        "data_quality": {"overall_score": 85.0}
    }
    
    record_id = persistence.save_analysis_record(
        symbol="AAPL",
        data_package=sample_package,
        prompt="Test prompt...",
        raw_response='{"rating": 85, "symbol": "AAPL"}',
        performance_metadata={"response_time_seconds": 2.5}
    )
    print(f"Saved record: {record_id}")
    
    # Test session completion
    session_summary = persistence.finish_batch_session()
    if session_summary:
        print(f"Session completed: {session_summary['successful_analyses']}/{session_summary['total_opportunities']} successful")
    
    # Test daily summary
    today_summary = persistence.get_daily_summary()
    print(f"Today's summary: {today_summary['total_analyses']} analyses, ${today_summary['total_cost']:.2f} cost")
    
    print(f"Data stored in: {persistence.storage_path}")