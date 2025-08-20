# GitHub Secrets Setup Guide

To fix the CI/CD workflow email notifications, you need to configure these GitHub repository secrets.

## Required API Keys for Testing

### Navigate to Repository Settings
1. Go to your GitHub repository
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions"
4. Click "New repository secret"

### Add These Secrets:

#### **Trading API Keys**
```
ALPACA_API_KEY_TEST=your_alpaca_paper_trading_api_key
ALPACA_SECRET_KEY_TEST=your_alpaca_paper_trading_secret_key
```

#### **News API Keys** (Get free tiers)
```
ALPHA_VANTAGE_API_KEY_TEST=get_from_alphavantage.co
FINNHUB_API_KEY_TEST=get_from_finnhub.io
NEWS_API_KEY_TEST=get_from_newsapi.org
POLYGON_API_KEY_TEST=get_from_polygon.io
IEX_CLOUD_API_KEY_TEST=get_from_iexcloud.io
```

#### **Social Media API Keys** (Optional)
```
TWITTER_BEARER_TOKEN_TEST=get_from_developer.twitter.com
REDDIT_CLIENT_SECRET_TEST=get_from_reddit.com/prefs/apps
```

#### **Crypto API Keys** (Optional)
```
COINMARKETCAP_API_KEY_TEST=get_from_pro.coinmarketcap.com
MESSARI_API_KEY_TEST=get_from_messari.io
```

## Free API Key Sources

### Alpha Vantage (Financial Data)
- Website: https://www.alphavantage.co/support/#api-key
- Free tier: 5 calls/minute, 500 calls/day
- Use for: Stock news and sentiment analysis

### Finnhub (Financial Data)
- Website: https://finnhub.io/register
- Free tier: 60 calls/minute
- Use for: Company news and earnings data

### NewsAPI (General News)
- Website: https://newsapi.org/register
- Free tier: 1000 requests/day
- Use for: Business news headlines

### Polygon.io (Market Data)
- Website: https://polygon.io/
- Free tier: 5 calls/minute
- Use for: Market news and data

### IEX Cloud (Financial Data)
- Website: https://iexcloud.io/
- Free tier: 500,000 calls/month
- Use for: Stock market data

## Test the Workflow

After adding secrets:

1. Make a small commit to trigger the workflow:
   ```bash
   git add .
   git commit -m "Update CI configuration with proper secrets"
   git push
   ```

2. Check GitHub Actions tab for workflow results

3. Email notifications should stop failing

## Troubleshooting

### If Tests Still Fail:

1. **Check Secret Names**: Ensure they match exactly (case-sensitive)
2. **Verify API Keys**: Test keys manually with curl
3. **Check Rate Limits**: Some APIs have strict rate limits
4. **Review Logs**: Check GitHub Actions logs for specific errors

### Common Issues:

- **401 Unauthorized**: Wrong API key or expired
- **403 Forbidden**: Rate limit exceeded or wrong permissions
- **404 Not Found**: Wrong API endpoint URL
- **Timeout**: API is slow or down

## Production vs Test Keys

- Use separate API keys for production and testing
- Test keys should have lower rate limits
- Never commit real API keys to code
- Use environment variables in production