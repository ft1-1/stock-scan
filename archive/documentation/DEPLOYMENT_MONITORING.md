# Production Deployment & Monitoring Strategy

## Deployment Architecture

### Infrastructure Overview

```
Production Environment Architecture

┌─────────────────────────────────────────────────────────────────┐
│                           Load Balancer                         │
│                    (nginx/HAProxy - Optional)                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                   Application Instances                         │
├─────────────────────────────────────────────────────────────────┤
│  Instance 1              │  Instance 2              │ Instance N │
│  ┌─────────────────┐    │  ┌─────────────────┐    │    ...      │
│  │ Options Screener│    │  │ Options Screener│    │             │
│  │ - Core Engine   │    │  │ - Core Engine   │    │             │
│  │ - API Providers │    │  │ - API Providers │    │             │
│  │ - AI Integration│    │  │ - AI Integration│    │             │
│  └─────────────────┘    │  └─────────────────┘    │             │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                     Shared Services                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │   Redis     │  │ PostgreSQL  │  │   Logging   │  │Monitoring││
│  │   Cache     │  │  Database   │  │  (Loki/ELK) │  │(Grafana)││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                     External Services                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    EODHD    │  │ MarketData  │  │   Claude    │             │
│  │     API     │  │     API     │  │     AI      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Strategies

#### 1. Single-Instance Deployment (Small Scale)
**Suitable for**: Daily screening of <1000 stocks

```bash
# Server Requirements
- CPU: 4 cores (8 recommended)
- RAM: 8GB (16GB recommended)
- Storage: 100GB SSD
- Network: Stable internet with >10Mbps
- OS: Ubuntu 22.04 LTS or CentOS 8

# Software Stack
- Python 3.11+
- systemd for service management
- nginx for reverse proxy (optional)
- SQLite or PostgreSQL for data storage
- Redis for caching (optional)
```

#### 2. Multi-Instance Deployment (Large Scale)
**Suitable for**: Daily screening of 5000+ stocks

```bash
# Load Balancer Requirements
- nginx or HAProxy
- SSL termination
- Health check configuration
- Request routing based on load

# Application Instance Requirements (per instance)
- CPU: 2-4 cores
- RAM: 4-8GB
- Storage: 50GB SSD
- Dedicated network interface

# Shared Services Requirements
- PostgreSQL: 8 cores, 16GB RAM, 500GB SSD
- Redis: 2 cores, 4GB RAM, 50GB SSD
- Monitoring: 2 cores, 4GB RAM, 100GB SSD
```

## Containerization Strategy

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim-bullseye

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 screener
WORKDIR /app
RUN chown screener:screener /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=screener:screener src/ ./src/
COPY --chown=screener:screener configs/ ./configs/
COPY --chown=screener:screener scripts/ ./scripts/

# Switch to non-root user
USER screener

# Create data directories
RUN mkdir -p data/logs data/cache data/results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python scripts/health_check.py

# Default command
CMD ["python", "-m", "src.main"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  options-screener:
    build: .
    container_name: options-screener
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://screener:${DB_PASSWORD}@postgres:5432/screening
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    networks:
      - screener-network

  postgres:
    image: postgres:15
    container_name: screener-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=screening
      - POSTGRES_USER=screener
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - screener-network

  redis:
    image: redis:7-alpine
    container_name: screener-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - screener-network

  nginx:
    image: nginx:alpine
    container_name: screener-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - options-screener
    networks:
      - screener-network

volumes:
  postgres_data:
  redis_data:

networks:
  screener-network:
    driver: bridge
```

## Scheduling and Job Management

### Systemd Service Configuration

```ini
# /etc/systemd/system/options-screener.service
[Unit]
Description=Options Screener Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=screener
Group=screener
WorkingDirectory=/opt/options-screener
Environment=PYTHONPATH=/opt/options-screener
ExecStart=/opt/options-screener/venv/bin/python -m src.main
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
MemoryLimit=2G
CPUQuota=200%

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/options-screener/data

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=options-screener

[Install]
WantedBy=multi-user.target
```

### Cron-based Scheduling

```bash
# /etc/cron.d/options-screener
# Daily screening at 6:30 AM EST (after market close)
30 6 * * 1-5 screener /opt/options-screener/scripts/run_daily_scan.sh >> /var/log/options-screener/cron.log 2>&1

# Weekly deep analysis on Sundays at 10:00 AM
0 10 * * 0 screener /opt/options-screener/scripts/run_weekly_analysis.sh >> /var/log/options-screener/weekly.log 2>&1

# Health check every 15 minutes
*/15 * * * * screener /opt/options-screener/scripts/health_check.sh >> /var/log/options-screener/health.log 2>&1

# Log rotation and cleanup daily at 2:00 AM
0 2 * * * screener /opt/options-screener/scripts/cleanup_logs.sh >> /var/log/options-screener/cleanup.log 2>&1
```

### Advanced Scheduling with APScheduler

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

class ScreeningScheduler:
    """Advanced scheduler for options screening jobs."""
    
    def __init__(self, workflow_manager):
        self.scheduler = AsyncIOScheduler()
        self.workflow_manager = workflow_manager
        
    def setup_jobs(self):
        """Configure all scheduled jobs."""
        
        # Daily screening job (6:30 AM EST, Monday-Friday)
        self.scheduler.add_job(
            func=self.run_daily_screening,
            trigger=CronTrigger(
                hour=6, minute=30, 
                day_of_week='mon-fri',
                timezone='America/New_York'
            ),
            id='daily_screening',
            name='Daily Options Screening',
            max_instances=1,
            misfire_grace_time=1800  # 30 minutes
        )
        
        # Pre-market quick scan (8:00 AM EST)
        self.scheduler.add_job(
            func=self.run_premarket_scan,
            trigger=CronTrigger(
                hour=8, minute=0,
                day_of_week='mon-fri', 
                timezone='America/New_York'
            ),
            id='premarket_scan',
            name='Pre-market Quick Scan'
        )
        
        # Health monitoring (every 5 minutes)
        self.scheduler.add_job(
            func=self.health_check,
            trigger=IntervalTrigger(minutes=5),
            id='health_check',
            name='System Health Check'
        )
        
        # Cost monitoring (hourly)
        self.scheduler.add_job(
            func=self.monitor_costs,
            trigger=IntervalTrigger(hours=1),
            id='cost_monitoring',
            name='API Cost Monitoring'
        )
        
        # Weekly cleanup (Sundays at 3:00 AM)
        self.scheduler.add_job(
            func=self.weekly_cleanup,
            trigger=CronTrigger(
                hour=3, minute=0,
                day_of_week='sun',
                timezone='America/New_York'
            ),
            id='weekly_cleanup',
            name='Weekly Data Cleanup'
        )
    
    async def run_daily_screening(self):
        """Execute daily screening workflow."""
        try:
            logger.info("Starting daily screening job")
            result = await self.workflow_manager.execute_daily_screening()
            await self.send_completion_notification(result)
            logger.info("Daily screening completed successfully")
            
        except Exception as e:
            logger.error(f"Daily screening failed: {e}")
            await self.send_error_notification(e)
    
    async def run_premarket_scan(self):
        """Execute quick pre-market scan."""
        # Implementation for quick scan
        pass
    
    def start(self):
        """Start the scheduler."""
        self.setup_jobs()
        self.scheduler.start()
        logger.info("Scheduler started successfully")
```

## Monitoring and Alerting System

### Application Performance Monitoring

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import psutil
import logging

@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    response_time_ms: float
    requests_per_second: float
    error_rate_percent: float

class MetricsCollector:
    """Collect and track application metrics."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.api_call_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
    async def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            active_connections=len(psutil.net_connections()),
            response_time_ms=self.get_avg_response_time(),
            requests_per_second=self.get_requests_per_second(),
            error_rate_percent=self.get_error_rate()
        )
    
    def track_api_call(self, provider: str, duration: float):
        """Track API call performance."""
        if provider not in self.api_call_times:
            self.api_call_times[provider] = []
        
        self.api_call_times[provider].append(duration)
        
        # Keep only last 1000 calls
        if len(self.api_call_times[provider]) > 1000:
            self.api_call_times[provider] = self.api_call_times[provider][-1000:]
    
    def track_error(self, error_type: str):
        """Track error occurrences."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, webhook_urls: List[str]):
        self.webhook_urls = webhook_urls
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate_percent': 5.0,
            'response_time_ms': 5000.0
        }
        self.last_alerts: Dict[str, float] = {}
        self.alert_cooldown = 300  # 5 minutes
    
    async def check_and_alert(self, metrics: PerformanceMetrics):
        """Check metrics and send alerts if needed."""
        current_time = time.time()
        
        alerts_to_send = []
        
        # Check each threshold
        for metric, threshold in self.alert_thresholds.items():
            value = getattr(metrics, metric)
            
            if value > threshold:
                alert_key = f"{metric}_high"
                last_alert_time = self.last_alerts.get(alert_key, 0)
                
                if current_time - last_alert_time > self.alert_cooldown:
                    alerts_to_send.append({
                        'type': 'threshold_exceeded',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'timestamp': current_time
                    })
                    self.last_alerts[alert_key] = current_time
        
        # Send alerts
        for alert in alerts_to_send:
            await self.send_alert(alert)
    
    async def send_alert(self, alert: Dict):
        """Send alert to configured channels."""
        message = self.format_alert_message(alert)
        
        for webhook_url in self.webhook_urls:
            try:
                await self.send_webhook_alert(webhook_url, message)
            except Exception as e:
                logger.error(f"Failed to send alert to {webhook_url}: {e}")
    
    def format_alert_message(self, alert: Dict) -> str:
        """Format alert message."""
        if alert['type'] == 'threshold_exceeded':
            return (
                f"🚨 ALERT: {alert['metric']} exceeded threshold\n"
                f"Current value: {alert['value']:.2f}\n"
                f"Threshold: {alert['threshold']:.2f}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))}"
            )
        return str(alert)
```

### Business Metrics Monitoring

```python
class BusinessMetricsTracker:
    """Track business-specific metrics."""
    
    def __init__(self):
        self.daily_stats = {}
        self.cost_tracking = {}
        
    def track_screening_completion(self, execution: WorkflowExecution):
        """Track daily screening metrics."""
        date_key = execution.started_at.date()
        
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'total_stocks_screened': 0,
                'opportunities_found': 0,
                'ai_analyses_performed': 0,
                'total_cost_usd': 0.0,
                'execution_time_seconds': 0.0,
                'error_count': 0
            }
        
        stats = self.daily_stats[date_key]
        stats['total_stocks_screened'] += execution.total_symbols_screened
        stats['opportunities_found'] += len(execution.final_results)
        stats['total_cost_usd'] += execution.total_cost_usd
        stats['execution_time_seconds'] += execution.duration_seconds or 0
        stats['error_count'] += len(execution.errors)
        
        # Count AI analyses
        ai_count = sum(1 for result in execution.final_results 
                      if result.ai_analysis is not None)
        stats['ai_analyses_performed'] += ai_count
    
    def get_daily_summary(self, date: date = None) -> Dict:
        """Get daily performance summary."""
        if date is None:
            date = datetime.now().date()
        
        return self.daily_stats.get(date, {})
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        total_costs = {}
        
        for date_stats in self.daily_stats.values():
            total_costs['total'] = total_costs.get('total', 0) + date_stats.get('total_cost_usd', 0)
        
        return total_costs
```

### Log Management and Analysis

```python
import structlog
from loguru import logger
import json

# Structured logging configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Enhanced logger with business context
class BusinessLogger:
    """Enhanced logger with business context."""
    
    def __init__(self):
        self.log = structlog.get_logger()
        
    def log_screening_start(self, execution_id: str, criteria: ScreeningCriteria):
        """Log screening workflow start."""
        self.log.info(
            "screening_started",
            execution_id=execution_id,
            criteria_name=criteria.name,
            max_stocks=criteria.get('max_stocks', 'unlimited'),
            ai_enabled=criteria.get('ai_enabled', False)
        )
    
    def log_screening_complete(self, execution: WorkflowExecution):
        """Log screening workflow completion."""
        self.log.info(
            "screening_completed",
            execution_id=execution.execution_id,
            duration_seconds=execution.duration_seconds,
            stocks_screened=execution.total_symbols_screened,
            opportunities_found=len(execution.final_results),
            total_cost_usd=execution.total_cost_usd,
            error_count=len(execution.errors)
        )
    
    def log_api_call(self, provider: str, endpoint: str, duration: float, status: str):
        """Log API call details."""
        self.log.info(
            "api_call",
            provider=provider,
            endpoint=endpoint,
            duration_ms=duration * 1000,
            status=status
        )
    
    def log_error(self, error: Exception, context: Dict = None):
        """Log error with context."""
        self.log.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
```

## Health Checks and Diagnostics

### Comprehensive Health Check System

```python
from enum import Enum
from typing import Dict, List
import asyncio
import aiohttp

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, settings):
        self.settings = settings
        self.providers = {}
        
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform complete system health check."""
        checks = {
            'system': await self.check_system_resources(),
            'database': await self.check_database(),
            'cache': await self.check_cache(),
            'external_apis': await self.check_external_apis(),
            'disk_space': await self.check_disk_space(),
            'configuration': await self.check_configuration()
        }
        
        overall_status = self.determine_overall_health(checks)
        
        return {
            'status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        if cpu_percent > 80 or memory.percent > 85:
            status = HealthStatus.DEGRADED
        if cpu_percent > 95 or memory.percent > 95:
            status = HealthStatus.UNHEALTHY
            
        return {
            'status': status.value,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'load_average': psutil.getloadavg()
        }
    
    async def check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        api_checks = {}
        
        # EODHD API check
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.settings.eodhd.base_url}/eod/AAPL.US"
                params = {'api_token': self.settings.eodhd.api_key, 'fmt': 'json'}
                
                start_time = time.time()
                async with session.get(url, params=params, timeout=10) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        api_checks['eodhd'] = {
                            'status': HealthStatus.HEALTHY.value,
                            'response_time_ms': duration * 1000
                        }
                    else:
                        api_checks['eodhd'] = {
                            'status': HealthStatus.UNHEALTHY.value,
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            api_checks['eodhd'] = {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e)
            }
        
        # MarketData API check
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.settings.marketdata.base_url}/stocks/quotes/AAPL/"
                headers = {'Authorization': f'Bearer {self.settings.marketdata.api_key}'}
                
                start_time = time.time()
                async with session.get(url, headers=headers, timeout=10) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        api_checks['marketdata'] = {
                            'status': HealthStatus.HEALTHY.value,
                            'response_time_ms': duration * 1000
                        }
                    else:
                        api_checks['marketdata'] = {
                            'status': HealthStatus.UNHEALTHY.value,
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            api_checks['marketdata'] = {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e)
            }
        
        return api_checks

# Health check script
def create_health_check_script():
    """Create standalone health check script."""
    script_content = '''#!/usr/bin/env python3
import asyncio
import sys
import json
from src.monitoring.health_checker import HealthChecker
from src.config.settings import get_settings

async def main():
    """Run health check and exit with appropriate code."""
    settings = get_settings()
    checker = HealthChecker(settings)
    
    try:
        health_result = await checker.check_system_health()
        print(json.dumps(health_result, indent=2))
        
        if health_result['status'] == 'healthy':
            sys.exit(0)
        elif health_result['status'] == 'degraded':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())
'''
    return script_content
```

## Backup and Recovery Procedures

### Data Backup Strategy

```bash
#!/bin/bash
# backup_data.sh - Comprehensive backup script

set -euo pipefail

BACKUP_DIR="/opt/backups/options-screener"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup application data
echo "Backing up application data..."
tar -czf "$BACKUP_DIR/app_data_$DATE.tar.gz" -C /opt/options-screener data/

# Backup configuration
echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" -C /opt/options-screener configs/ .env

# Backup database (if using PostgreSQL)
if [ "$DATABASE_TYPE" = "postgresql" ]; then
    echo "Backing up PostgreSQL database..."
    pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/database_$DATE.sql.gz"
fi

# Backup Redis data (if using Redis)
if [ "$REDIS_ENABLED" = "true" ]; then
    echo "Backing up Redis data..."
    redis-cli --rdb "$BACKUP_DIR/redis_$DATE.rdb"
fi

# Clean up old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully at $DATE"

# Optional: Upload to cloud storage
if [ "$CLOUD_BACKUP_ENABLED" = "true" ]; then
    echo "Uploading to cloud storage..."
    aws s3 sync "$BACKUP_DIR" "s3://$BACKUP_BUCKET/options-screener/"
fi
```

### Disaster Recovery Plan

```yaml
# disaster_recovery.yaml
recovery_procedures:
  complete_system_failure:
    steps:
      1. "Provision new infrastructure"
      2. "Restore application code from repository"
      3. "Restore configuration from backup"
      4. "Restore database from latest backup"
      5. "Restore application data"
      6. "Verify API connectivity"
      7. "Run health checks"
      8. "Resume scheduled operations"
    estimated_recovery_time: "2-4 hours"
    
  database_corruption:
    steps:
      1. "Stop application services"
      2. "Restore database from latest good backup"
      3. "Verify data integrity"
      4. "Restart application services"
      5. "Monitor for issues"
    estimated_recovery_time: "30-60 minutes"
    
  api_provider_outage:
    steps:
      1. "Switch to backup provider"
      2. "Update configuration"
      3. "Test connectivity"
      4. "Resume operations with degraded service"
    estimated_recovery_time: "5-15 minutes"

monitoring_requirements:
  backup_verification:
    frequency: "daily"
    method: "automated restore test"
  
  recovery_testing:
    frequency: "monthly"
    scope: "complete disaster recovery simulation"
```

This comprehensive deployment and monitoring strategy provides:

1. **Scalable Infrastructure**: Support for both small and large-scale deployments
2. **Robust Monitoring**: Complete application and business metrics tracking
3. **Proactive Alerting**: Intelligent alerting with threshold-based notifications
4. **Health Diagnostics**: Comprehensive health checking for all components
5. **Disaster Recovery**: Complete backup and recovery procedures
6. **Container Support**: Docker and Kubernetes deployment options
7. **Security**: Proper user permissions and security hardening
8. **Cost Control**: Monitoring and alerting for API usage costs

The system is designed to run reliably as a daily scheduled job with minimal manual intervention while providing comprehensive observability and recovery capabilities.