"""Mailgun email notification client for sending opportunity alerts."""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import json
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MailgunClient:
    """Client for sending email notifications via Mailgun API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        from_email: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Mailgun client.
        
        Args:
            api_key: Mailgun API key
            domain: Mailgun domain (e.g., 'mg.yourdomain.com')
            from_email: Sender email address
            base_url: Mailgun API base URL (defaults to US endpoint)
        """
        self.api_key = api_key or os.getenv('MAILGUN_API_KEY')
        self.domain = domain or os.getenv('MAILGUN_DOMAIN')
        self.from_email = from_email or os.getenv('MAILGUN_FROM_EMAIL', 'Stock Scanner <noreply@stockscanner.ai>')
        self.base_url = base_url or os.getenv('MAILGUN_BASE_URL', 'https://api.mailgun.net/v3')
        
        if not self.api_key:
            raise ValueError("Mailgun API key not provided")
        if not self.domain:
            raise ValueError("Mailgun domain not provided")
            
        logger.info(f"Mailgun client initialized for domain: {self.domain}")
    
    async def send_opportunity_alert(
        self,
        recipients: List[str],
        opportunities: List[Dict[str, Any]],
        execution_summary: Dict[str, Any],
        subject: Optional[str] = None
    ) -> bool:
        """
        Send opportunity alert email with top ranked stocks.
        
        Args:
            recipients: List of email addresses to send to
            opportunities: List of top opportunity dictionaries
            execution_summary: Workflow execution summary data
            subject: Optional email subject override
            
        Returns:
            True if email sent successfully
        """
        try:
            # Generate email content
            html_content = self._generate_html_content(opportunities, execution_summary)
            text_content = self._generate_text_content(opportunities, execution_summary)
            
            # Default subject with timestamp
            if not subject:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                subject = f"🎯 Top {len(opportunities)} Stock Opportunities - {timestamp}"
            
            # Send email
            success = await self._send_email(
                to=recipients,
                subject=subject,
                text=text_content,
                html=html_content
            )
            
            if success:
                logger.info(f"Successfully sent opportunity alert to {len(recipients)} recipients")
            else:
                logger.error("Failed to send opportunity alert")
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending opportunity alert: {e}")
            return False
    
    async def _send_email(
        self,
        to: List[str],
        subject: str,
        text: str,
        html: Optional[str] = None,
        attachments: Optional[List[Dict]] = None
    ) -> bool:
        """
        Send email via Mailgun API.
        
        Args:
            to: List of recipient email addresses
            subject: Email subject
            text: Plain text content
            html: Optional HTML content
            attachments: Optional list of attachment dictionaries
            
        Returns:
            True if email sent successfully
        """
        try:
            url = f"{self.base_url}/{self.domain}/messages"
            
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('from', self.from_email)
            data.add_field('to', ','.join(to))
            data.add_field('subject', subject)
            data.add_field('text', text)
            
            if html:
                data.add_field('html', html)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    data.add_field(
                        'attachment',
                        attachment['content'],
                        filename=attachment['filename'],
                        content_type=attachment.get('content_type', 'application/octet-stream')
                    )
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    auth=aiohttp.BasicAuth('api', self.api_key),
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Email sent successfully. Message ID: {result.get('id')}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Mailgun API error ({response.status}): {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending email via Mailgun: {e}")
            return False
    
    def _generate_html_content(
        self,
        opportunities: List[Dict[str, Any]],
        execution_summary: Dict[str, Any]
    ) -> str:
        """Generate HTML email content for opportunities."""
        
        # Calculate average scores
        avg_ai_rating = sum((opp.get('ai_rating') or 0) for opp in opportunities) / len(opportunities) if opportunities else 0
        avg_local_score = sum((opp.get('local_score') or 0) for opp in opportunities) / len(opportunities) if opportunities else 0
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.7;
                    color: #2d3748;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 20px 0;
                }}
                .container {{ 
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px 30px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                .header:before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    right: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                    animation: pulse 15s ease-in-out infinite;
                }}
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); opacity: 0.5; }}
                    50% {{ transform: scale(1.1); opacity: 0.3; }}
                }}
                .header h1 {{ 
                    font-size: 32px;
                    font-weight: 700;
                    margin-bottom: 8px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                    position: relative;
                    z-index: 1;
                }}
                .header p {{ 
                    font-size: 16px;
                    opacity: 0.95;
                    position: relative;
                    z-index: 1;
                }}
                .content-wrapper {{ padding: 30px; }}
                .summary {{ 
                    background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
                    padding: 25px;
                    border-radius: 12px;
                    margin-bottom: 35px;
                    border: 1px solid #e1e4e8;
                }}
                .summary h2 {{
                    color: #4a5568;
                    font-size: 20px;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .summary h2:before {{
                    content: '📊';
                    font-size: 24px;
                }}
                .summary-grid {{ 
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                }}
                .stat {{ 
                    text-align: center;
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .stat:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                }}
                .stat-value {{ 
                    font-size: 32px;
                    font-weight: 700;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                .stat-label {{ 
                    font-size: 11px;
                    color: #718096;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    font-weight: 600;
                    margin-top: 5px;
                }}
                .opportunities-title {{
                    font-size: 24px;
                    color: #2d3748;
                    margin: 35px 0 25px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .opportunities-title:before {{
                    content: '🎯';
                    font-size: 28px;
                }}
                .opportunity-card {{ 
                    background: white;
                    border-radius: 12px;
                    padding: 0;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border: 1px solid #e1e4e8;
                    overflow: hidden;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .opportunity-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
                }}
                .opportunity-header {{ 
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 20px 25px;
                    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                    border-bottom: 2px solid #e2e8f0;
                }}
                .symbol-info {{ 
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }}
                .rank-badge {{ 
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    font-weight: 700;
                    font-size: 16px;
                    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
                }}
                .symbol-name {{ 
                    font-size: 26px;
                    font-weight: 700;
                    color: #1a202c;
                    letter-spacing: -0.5px;
                }}
                .combined-score {{
                    font-size: 14px;
                    color: #718096;
                    font-weight: 500;
                }}
                .rating-box {{ 
                    text-align: center;
                    padding: 10px 20px;
                    border-radius: 10px;
                    min-width: 80px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .high-rating {{ 
                    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                    color: white;
                }}
                .medium-rating {{ 
                    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
                    color: white;
                }}
                .low-rating {{ 
                    background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
                    color: white;
                }}
                .rating-value {{ 
                    font-size: 32px;
                    font-weight: 700;
                    line-height: 1;
                }}
                .rating-label {{ 
                    font-size: 10px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-top: 4px;
                    opacity: 0.9;
                }}
                .opportunity-body {{ padding: 25px; }}
                .thesis-section {{ 
                    background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
                    padding: 18px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border-left: 4px solid #667eea;
                }}
                .thesis-label {{
                    font-weight: 600;
                    color: #4a5568;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .confidence-badge {{
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 11px;
                    font-weight: 500;
                    background: rgba(102, 126, 234, 0.1);
                    color: #667eea;
                }}
                .thesis-text {{ 
                    color: #2d3748;
                    font-size: 15px;
                    line-height: 1.6;
                }}
                .details-grid {{ 
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }}
                .detail-box {{ 
                    padding: 18px;
                    background: #f7fafc;
                    border-radius: 10px;
                    border: 1px solid #e2e8f0;
                }}
                .detail-title {{ 
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 12px;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .list-item {{ 
                    margin: 8px 0;
                    padding-left: 20px;
                    position: relative;
                    color: #4a5568;
                    font-size: 14px;
                    line-height: 1.5;
                }}
                .list-item:before {{ 
                    content: '→';
                    position: absolute;
                    left: 0;
                    color: #667eea;
                    font-weight: bold;
                }}
                .option-recommendation {{ 
                    background: linear-gradient(135deg, #ebf4ff 0%, #e0ecff 100%);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    border: 1px solid #c3dafe;
                }}
                .option-title {{
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 15px;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .option-section {{ 
                    margin: 10px 0;
                    padding: 10px;
                    background: white;
                    border-radius: 6px;
                    font-size: 14px;
                }}
                .option-label {{ 
                    font-weight: 600;
                    color: #4a5568;
                    display: inline-block;
                    min-width: 140px;
                }}
                .option-value {{
                    color: #2d3748;
                }}
                .red-flag {{ 
                    background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
                    color: #742a2a;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                    border-left: 4px solid #fc8181;
                }}
                .red-flag-title {{
                    font-weight: 600;
                    margin-bottom: 8px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .red-flag-item {{
                    margin: 5px 0;
                    font-size: 14px;
                }}
                .notes-section {{ 
                    background: #f7fafc;
                    padding: 15px;
                    border-radius: 10px;
                    margin-top: 15px;
                    font-size: 14px;
                    color: #4a5568;
                    border: 1px solid #e2e8f0;
                }}
                .notes-label {{
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 5px;
                }}
                .footer {{ 
                    margin-top: 40px;
                    padding: 25px;
                    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                    border-top: 1px solid #e2e8f0;
                    text-align: center;
                }}
                .footer-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .footer-stat {{
                    font-size: 13px;
                    color: #4a5568;
                }}
                .footer-stat-label {{
                    font-weight: 600;
                    color: #2d3748;
                }}
                .footer-note {{
                    font-size: 12px;
                    color: #718096;
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #e2e8f0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Top Stock Opportunities</h1>
                    <p>AI-Powered Options Screening Results</p>
                </div>
                
                <div class="content-wrapper">
                    <div class="summary">
                        <h2>Execution Summary</h2>
                        <div class="summary-grid">
                            <div class="stat">
                                <div class="stat-value">{execution_summary.get('symbols_screened', 0)}</div>
                                <div class="stat-label">Symbols Screened</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">{len(opportunities)}</div>
                                <div class="stat-label">Top Opportunities</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">{avg_ai_rating:.1f}</div>
                                <div class="stat-label">Avg AI Rating</div>
                            </div>
                        </div>
                    </div>
                    
                    <h2 class="opportunities-title">Top {min(len(opportunities), 10)} Ranked Opportunities</h2>
        """
        
        # Limit to top 10 opportunities
        for i, opp in enumerate(opportunities[:10], 1):
            ai_rating = opp.get('ai_rating', 0) or 0
            local_score = opp.get('local_score', 0) or 0
            combined = (local_score * 0.1) + (ai_rating * 0.9)
            
            # Determine rating class
            rating_class = 'high-rating' if ai_rating >= 80 else 'medium-rating' if ai_rating >= 60 else 'low-rating'
            
            # Get AI analysis details
            ai_analysis = opp.get('ai_analysis', {})
            thesis = ai_analysis.get('thesis', 'Analysis pending')
            opportunities_list = ai_analysis.get('opportunities', [])
            risks_list = ai_analysis.get('risks', [])
            option_contract = ai_analysis.get('option_contract', {})
            red_flags = ai_analysis.get('red_flags', [])
            notes = ai_analysis.get('notes', '')
            confidence = ai_analysis.get('confidence', 'medium')
            
            # Format risk and position sizing details
            risk_metrics = opp.get('enhanced_data', {}).get('risk_metrics', {})
            position_sizing = opp.get('enhanced_data', {}).get('position_sizing', {})
            risk_str = "N/A"
            if risk_metrics:
                current_price = risk_metrics.get('current_price', 0)
                atr_pct = risk_metrics.get('atr_percent', 0)
                stop_loss = risk_metrics.get('suggested_stop_loss', 0)
                risk_str = f"Price: ${current_price:.2f} | ATR: {atr_pct:.2%} | Stop: ${stop_loss:.2f}"
                if position_sizing and 'position_size_shares' in position_sizing:
                    shares = position_sizing.get('position_size_shares', 0)
                    risk_str += f" | Size: {shares} shares"
            
            html += f"""
                    <div class="opportunity-card">
                        <div class="opportunity-header">
                            <div class="symbol-info">
                                <span class="rank-badge">{i}</span>
                                <span class="symbol-name">{opp.get('symbol', 'N/A')}</span>
                                <span class="combined-score">Combined Score: {combined:.1f}</span>
                            </div>
                            <div class="rating-box {rating_class}">
                                <div class="rating-value">{ai_rating:.0f}</div>
                                <div class="rating-label">AI Rating</div>
                            </div>
                        </div>
                        
                        <div class="opportunity-body">
                            <div class="thesis-section">
                                <div class="thesis-label">
                                    Investment Thesis
                                    <span class="confidence-badge">{confidence.upper()} CONFIDENCE</span>
                                </div>
                                <div class="thesis-text">{thesis}</div>
                            </div>
                    
                            <div class="details-grid">
                                <div class="detail-box">
                                    <div class="detail-title">Opportunities</div>
            """
            
            for opportunity in opportunities_list[:3]:  # Show top 3
                html += f'<div class="list-item">{opportunity}</div>'
            
            html += f"""
                                </div>
                                <div class="detail-box">
                                    <div class="detail-title">Risks</div>
            """
            
            for risk in risks_list[:3]:  # Show top 3
                html += f'<div class="list-item">{risk}</div>'
            
            html += f"""
                                </div>
                            </div>
                            
                            <div class="option-recommendation">
                                <div class="option-title">Position Sizing & Risk Management</div>
                                <div class="option-section">
                                    <span class="option-label">Risk Profile:</span>
                                    <span class="option-value">{risk_str}</span>
                                </div>
                                <div class="option-section">
                                    <span class="option-label">Entry Strategy:</span>
                                    <span class="option-value">{ai_analysis.get('entry_strategy', 'Momentum breakout entry')}</span>
                                </div>
                                <div class="option-section">
                                    <span class="option-label">Risk Level:</span>
                                    <span class="option-value">{"High" if risk_metrics.get('atr_percent', 0) > 0.04 else "Moderate" if risk_metrics.get('atr_percent', 0) > 0.02 else "Low"} Volatility</span>
                                </div>
                                <div class="option-section">
                                    <span class="option-label">Stop Loss:</span>
                                    <span class="option-value">${risk_metrics.get('suggested_stop_loss', 0):.2f} ({risk_metrics.get('stop_loss_risk_percent', 0):.1%} risk)</span>
                                </div>
                            </div>
            """
            
            # Add red flags if any
            if red_flags:
                html += '<div class="red-flag"><div class="red-flag-title">⚠️ Red Flags</div>'
                for flag in red_flags:
                    html += f'<div class="red-flag-item">{flag}</div>'
                html += '</div>'
            
            # Add notes if any
            if notes:
                html += f'<div class="notes-section"><div class="notes-label">Notes</div>{notes}</div>'
            
            html += """
                        </div>
                    </div>
            """
        
        html += f"""
                </div>
                
                <div class="footer">
                    <div class="footer-grid">
                        <div class="footer-stat">
                            <div class="footer-stat-label">Report Generated</div>
                            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                        <div class="footer-stat">
                            <div class="footer-stat-label">Execution Time</div>
                            {execution_summary.get('total_execution_time_seconds', 0):.1f} seconds
                        </div>
                        <div class="footer-stat">
                            <div class="footer-stat-label">Total Cost</div>
                            ${execution_summary.get('total_cost_usd', 0):.4f}
                        </div>
                    </div>
                    <div class="footer-note">
                        This is an automated report from your Stock Options Screening System.<br>
                        AI ratings are weighted at 90% and technical scores at 10% for final rankings.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_content(
        self,
        opportunities: List[Dict[str, Any]],
        execution_summary: Dict[str, Any]
    ) -> str:
        """Generate plain text email content for opportunities."""
        
        text = f"""
TOP STOCK OPPORTUNITIES - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

EXECUTION SUMMARY:
- Symbols Screened: {execution_summary.get('symbols_screened', 0)}
- Opportunities Found: {len(opportunities)}
- Execution Time: {execution_summary.get('total_execution_time_seconds', 0):.1f}s
- Total Cost: ${execution_summary.get('total_cost_usd', 0):.4f}

TOP {min(len(opportunities), 10)} RANKED OPPORTUNITIES:
{'='*60}

"""
        
        # Limit to top 10 opportunities
        for i, opp in enumerate(opportunities[:10], 1):
            ai_rating = opp.get('ai_rating', 0) or 0
            local_score = opp.get('local_score', 0) or 0
            combined = (local_score * 0.1) + (ai_rating * 0.9)
            
            # Get AI analysis details
            ai_analysis = opp.get('ai_analysis', {})
            thesis = ai_analysis.get('thesis', 'Analysis pending')
            opportunities_list = ai_analysis.get('opportunities', [])[:3]
            risks_list = ai_analysis.get('risks', [])[:3]
            option_contract = ai_analysis.get('option_contract', {})
            red_flags = ai_analysis.get('red_flags', [])
            confidence = ai_analysis.get('confidence', 'medium')
            
            # Get risk metrics for text display
            risk_metrics_text = opp.get('enhanced_data', {}).get('risk_metrics', {})
            risk_display = "N/A"
            if risk_metrics_text:
                price = risk_metrics_text.get('current_price', 0)
                atr = risk_metrics_text.get('atr_percent', 0)
                stop = risk_metrics_text.get('suggested_stop_loss', 0)
                risk_display = f"Price: ${price:.2f} | ATR: {atr:.2%} | Stop: ${stop:.2f}"
            
            text += f"""
{'='*60}
#{i}. {opp.get('symbol', 'N/A')} - AI RATING: {ai_rating:.0f}/100 ({confidence} confidence)
{'='*60}
SCORES: AI: {ai_rating:.1f} | Local: {local_score:.1f} | Combined: {combined:.1f}

THESIS:
{thesis}

OPPORTUNITIES:
"""
            for opp_item in opportunities_list:
                text += f"  • {opp_item}\n"
            
            text += "\nRISKS:\n"
            for risk in risks_list:
                text += f"  • {risk}\n"
            
            text += f"""
POSITION SIZING & RISK:
  Risk Profile: {risk_display}
  Entry Strategy: {ai_analysis.get('entry_strategy', 'Momentum breakout entry')}
  Risk Level: {"High" if risk_metrics_text.get('atr_percent', 0) > 0.04 else "Moderate" if risk_metrics_text.get('atr_percent', 0) > 0.02 else "Low"} Volatility
  Stop Loss: ${risk_metrics_text.get('suggested_stop_loss', 0):.2f} ({risk_metrics_text.get('stop_loss_risk_percent', 0):.1%} risk)
"""
            
            if red_flags:
                text += "\nRED FLAGS:\n"
                for flag in red_flags:
                    text += f"  🚩 {flag}\n"
            
            text += "\n"
        
        text += f"""

Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
This is an automated report from your Stock Options Screening System.
"""
        
        return text


def create_mailgun_client() -> Optional[MailgunClient]:
    """
    Create Mailgun client from environment configuration.
    
    Returns:
        MailgunClient instance or None if not configured
    """
    try:
        # Check if Mailgun is configured
        api_key = os.getenv('MAILGUN_API_KEY')
        domain = os.getenv('MAILGUN_DOMAIN')
        
        if not api_key or not domain:
            logger.info("Mailgun not configured - skipping email notifications")
            return None
        
        return MailgunClient(
            api_key=api_key,
            domain=domain
        )
        
    except Exception as e:
        logger.error(f"Failed to create Mailgun client: {e}")
        return None