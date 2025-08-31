#!/usr/bin/env python
"""Test script to verify email format with AI details."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.notifications.mailgun_client import MailgunClient

# Sample data with AI analysis details
opportunities = [
    {
        'symbol': 'MSFT',
        'local_score': 42.1,
        'ai_rating': 72,
        'best_call': {
            'strike': 500,
            'bid': 21.30,
            'ask': 21.45,
            'delta': 0.611,
            'expiration': '2025-10-17'
        },
        'ai_analysis': {
            'thesis': 'MSFT shows strong long-term momentum with 24.26% 126-day return and solid technical indicators above both 50 and 200 SMAs, supported by robust fundamentals with 36.1% profit margin.',
            'opportunities': [
                'Strong technical foundation with price above key moving averages',
                'Excellent options liquidity with tight 0.70% bid-ask spread',
                'Strong fundamental metrics with 36.1% profit margin and 33.3% ROE'
            ],
            'risks': [
                'Negative short-term momentum (-0.71% 21-day ROC)',
                'Momentum acceleration showing weakness (-12.23)',
                'Upcoming earnings in September could increase volatility'
            ],
            'option_contract': {
                'recommendation': 'Consider LEAPS Call spread strategy given strong fundamentals but near-term weakness',
                'entry_timing': 'Wait for short-term momentum stabilization before entry',
                'risk_management': 'Use October 2025 expiration to avoid near-term volatility, consider spread to reduce cost'
            },
            'red_flags': [
                'Deteriorating short-term momentum metrics',
                'Earnings event within 40 days'
            ],
            'notes': 'Options metrics show good liquidity and reasonable implied volatility at 20.47%, but lacking historical volatility comparison data for complete IV analysis',
            'confidence': 'medium'
        }
    }
]

execution_summary = {
    'symbols_screened': 100,
    'total_execution_time_seconds': 120.5,
    'total_cost_usd': 0.0254
}

# Create client instance (won't actually send)
client = MailgunClient(
    api_key='test',
    domain='test',
    from_email='test@test.com'
)

# Generate HTML content
html_content = client._generate_html_content(opportunities, execution_summary)

# Save to file for inspection
output_file = Path('test_email_output.html')
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"✅ Test email HTML saved to: {output_file}")
print("\nTo view the email format, open test_email_output.html in a browser")

# Also generate text version
text_content = client._generate_text_content(opportunities, execution_summary)
print("\n" + "="*60)
print("TEXT VERSION PREVIEW:")
print("="*60)
print(text_content[:1500])  # Show first 1500 chars
print("...")