# Email Notifications Setup

## Overview
The stock scanner can send email notifications with the top 10 ranked opportunities after each workflow run. Emails are sent via Mailgun API.

## Configuration

### 1. Mailgun Account Setup
1. Sign up for a Mailgun account at https://mailgun.com
2. Verify your domain or use the sandbox domain for testing
3. Get your API key from the Mailgun dashboard

### 2. Environment Variables
Add these to your `.env` file:

```bash
# Enable email notifications
ENABLE_EMAIL_NOTIFICATIONS=true

# Mailgun Configuration
MAILGUN_API_KEY=your-mailgun-api-key
MAILGUN_DOMAIN=mg.yourdomain.com  # or sandbox domain
MAILGUN_FROM_EMAIL=Stock Scanner <alerts@yourdomain.com>

# Recipients (comma-separated for multiple)
EMAIL_RECIPIENTS=user1@example.com,user2@example.com
```

### 3. Sandbox Testing
For testing without a verified domain:
1. Use your Mailgun sandbox domain (e.g., `sandboxXXXX.mailgun.org`)
2. Add authorized recipients in Mailgun dashboard (sandbox only sends to verified emails)

## Email Content

Each email includes:
- **Top 10 Opportunities** ranked by combined score (90% AI rating, 10% local score)
- **Execution Summary** with total symbols screened, execution time, and costs
- **Detailed Table** with:
  - Symbol and rank
  - AI rating (0-100)
  - Local technical score
  - Combined final score
  - Best option contract details
  - Key AI insights

## Email Format

Emails are sent in both HTML and plain text formats:
- **HTML**: Rich formatting with tables, colors, and styling
- **Plain Text**: Simple format for email clients that don't support HTML

## Testing

To test email notifications:

1. Configure environment variables
2. Enable notifications:
   ```bash
   export ENABLE_EMAIL_NOTIFICATIONS=true
   export EMAIL_RECIPIENTS=your-email@example.com
   ```
3. Run a test with a single symbol:
   ```bash
   SCREENER_SPECIFIC_SYMBOLS=AAPL python3 run_production.py
   ```

## Troubleshooting

### Email Not Sending
- Check `ENABLE_EMAIL_NOTIFICATIONS=true` is set
- Verify Mailgun API key and domain are correct
- Check logs for Mailgun API errors
- Ensure recipients are verified (for sandbox domains)

### No Opportunities in Email
- Verify opportunities passed local ranking threshold (score >= 70)
- Check if AI analysis completed successfully
- Review logs for workflow errors

### Rate Limits
Mailgun free tier limits:
- 1,000 emails/month (sandbox)
- 5,000 emails/month (verified domain)

## Security Notes
- Never commit API keys to version control
- Use environment variables or secrets management
- Rotate API keys periodically
- Monitor Mailgun logs for unauthorized usage

## Example Email

Subject: 🎯 Top 10 Stock Opportunities - 2024-01-15 14:30

```
TOP RANKED OPPORTUNITIES
========================
1. AAPL  - AI: 92.5, Local: 85.2, Combined: 91.8
2. MSFT  - AI: 88.3, Local: 78.5, Combined: 87.3
3. NVDA  - AI: 85.7, Local: 82.1, Combined: 85.3
...
```