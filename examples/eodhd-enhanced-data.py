"""Enhanced PMCC Data Collection Script"""
import config as cfg
from eodhd import APIClient
from datetime import datetime, timedelta
import pandas as pd
import json

def get_last_trading_day(api, today=None):
    """Get the most recent trading day, accounting for market holidays"""
    if today is None:
        today = datetime.now()
    
    # Look back 10 days to catch any holidays
    ten_days_ago = (today - timedelta(days=10)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')
    
    try:
        # Get market holidays in the recent period
        holidays_data = api.get_details_trading_hours_stock_market_holidays(
            code="US", 
            from_date=ten_days_ago, 
            to_date=today_str
        )
        
        # Extract holiday dates
        holidays_df = pd.DataFrame(holidays_data)
        holiday_dates = set()
        if not holidays_df.empty and 'date' in holidays_df.columns:
            holiday_dates = set(holidays_df['date'].tolist())
    except Exception as e:
        print(f"Warning: Could not fetch holiday data: {e}")
        holiday_dates = set()
    
    # Find last trading day
    current_date = today
    while True:
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip weekends (Saturday=5, Sunday=6) and holidays
        if current_date.weekday() < 5 and current_date_str not in holiday_dates:
            return current_date_str
            
        current_date = current_date - timedelta(days=1)
        
        # Safety check - don't go back more than 10 days
        if (today - current_date).days > 10:
            break
    
    # Fallback to simple business day logic
    return (today - pd.BDay(1)).strftime('%Y-%m-%d')

def get_trading_dates(api):
    """Get all the dynamic dates needed for API calls"""
    today = datetime.now()
    last_trading_day = get_last_trading_day(api, today)
    last_trading_date = datetime.strptime(last_trading_day, '%Y-%m-%d')
    
    return {
        'today': last_trading_day,
        'thirty_days_ago': (last_trading_date - timedelta(days=30)).strftime('%Y-%m-%d'),
        'sixty_days_ago': (last_trading_date - timedelta(days=60)).strftime('%Y-%m-%d'),
        'six_months_ago': (last_trading_date - timedelta(days=180)).strftime('%Y-%m-%d'),
        'one_year_ago': (last_trading_date - timedelta(days=365)).strftime('%Y-%m-%d'),
        'hundred_days_ago': (last_trading_date - timedelta(days=100)).strftime('%Y-%m-%d')
    }

def collect_enhanced_data(api, ticker):
    """Collect all enhanced data for a given ticker"""
    print(f"\n=== Collecting Enhanced Data for {ticker} ===")
    
    # Get dynamic dates
    dates = get_trading_dates(api)
    print(f"Using last trading day: {dates['today']}")
    
    enhanced_data = {}
    
    try:
        # 1. Economic Events Data (US macro context)
        print("Fetching economic events...")
        enhanced_data['economic_events'] = api.get_economic_events_data(
            date_from=dates['six_months_ago'], 
            date_to=dates['today'], 
            country='US', 
            comparison='mom', 
            offset=0, 
            limit=30
        )
        
        # 2. Recent News
        print("Fetching recent news...")
        enhanced_data['news'] = api.financial_news(
            s=f'{ticker}.US', 
            t=None, 
            from_date=dates['thirty_days_ago'], 
            to_date=dates['today'], 
            limit=5, 
            offset=0
        )
        
        # 3. Fundamental Data (will be filtered locally)
        print("Fetching fundamental data...")
        raw_fundamentals = api.get_fundamentals_data(ticker=ticker)
        
        # Filter fundamental data immediately and only keep filtered version
        if raw_fundamentals:
            print("Filtering fundamental data...")
            enhanced_data['fundamentals'] = filter_fundamental_data(raw_fundamentals)
        else:
            enhanced_data['fundamentals'] = None
        
        # 4. Live Price
        print("Fetching live price...")
        enhanced_data['live_price'] = api.get_live_stock_prices(
            date_from=dates['today'], 
            date_to=dates['today'], 
            ticker=ticker
        )
        
        # 5. Earnings Data
        print("Fetching earnings data...")
        enhanced_data['earnings'] = api.get_upcoming_earnings_data(
            from_date=dates['one_year_ago'], 
            to_date=dates['today'], 
            symbols=ticker
        )
        
        # 6. Historical Price Data (30 days)
        print("Fetching historical prices...")
        enhanced_data['historical_prices'] = api.get_eod_historical_stock_market_data(
            symbol=ticker, 
            period='d', 
            from_date=dates['thirty_days_ago'], 
            to_date=dates['today'], 
            order='d'
        )
        
        # 7. Sentiment Analysis
        print("Fetching sentiment data...")
        enhanced_data['sentiment'] = api.get_sentiment(
            s=ticker, 
            from_date=dates['thirty_days_ago'], 
            to_date=dates['today']
        )
        
        # 8. Technical Indicators
        print("Fetching technical indicators...")
        enhanced_data['technical_indicators'] = {}
        
        # List of indicators to try
        indicators = [
            {'name': 'rsi', 'function': 'rsi', 'period': 14, 'days_back': 60},
            {'name': 'volatility', 'function': 'volatility', 'period': 30, 'days_back': 60},
            {'name': 'atr', 'function': 'atr', 'period': 14, 'days_back': 60}
        ]
        
        for indicator in indicators:
            try:
                print(f"  Fetching {indicator['name']}...")
                days_back = (datetime.now() - timedelta(days=indicator['days_back'])).strftime('%Y-%m-%d')
                
                result = api.get_technical_indicator_data(
                    ticker=f'{ticker}.US',
                    function=indicator['function'],
                    period=indicator['period'],
                    date_from=days_back,
                    date_to=dates['today'],
                    order='d',
                    splitadjusted_only='0'
                )
                enhanced_data['technical_indicators'][indicator['name']] = result
                print(f"  ✓ {indicator['name']} collected successfully")
                
            except Exception as e:
                print(f"  ✗ Failed to fetch {indicator['name']}: {e}")
                enhanced_data['technical_indicators'][indicator['name']] = None
        
        print(f"✓ Successfully collected all enhanced data for {ticker}")
        return enhanced_data
        
    except Exception as e:
        print(f"✗ Error collecting data for {ticker}: {e}")
        return None

def filter_fundamental_data(fundamentals):
    """Filter fundamental data to essential PMCC-relevant metrics"""
    if not fundamentals:
        return None
    
    try:
        # Extract key fundamental metrics for PMCC analysis
        filtered = {}
        
        if 'General' in fundamentals:
            general = fundamentals['General']
            filtered['company_info'] = {
                'name': general.get('Name'),
                'sector': general.get('Sector'),
                'industry': general.get('Industry'),
                'market_cap_mln': general.get('MarketCapitalization', 0) / 1000000 if general.get('MarketCapitalization') else None,
                'employees': general.get('FullTimeEmployees'),
                'description': general.get('Description', '')[:300]  # Brief description for context
            }
        
        if 'Highlights' in fundamentals:
            highlights = fundamentals['Highlights']
            filtered['financial_health'] = {
                # Profitability metrics
                'eps_ttm': highlights.get('EarningsShare'),
                'profit_margin': highlights.get('ProfitMargin'),
                'operating_margin': highlights.get('OperatingMarginTTM'),
                'roe': highlights.get('ReturnOnEquityTTM'),
                'roa': highlights.get('ReturnOnAssetsTTM'),
                
                # Growth metrics
                'revenue_growth_yoy': highlights.get('QuarterlyRevenueGrowthYOY'),
                'earnings_growth_yoy': highlights.get('QuarterlyEarningsGrowthYOY'),
                'eps_estimate_current_year': highlights.get('EPSEstimateCurrentYear'),
                'eps_estimate_next_year': highlights.get('EPSEstimateNextYear'),
                
                # Dividend information (critical for PMCC)
                'dividend_yield': highlights.get('DividendYield'),
                'dividend_per_share': highlights.get('DividendShare'),
                
                # Revenue and earnings
                'revenue_ttm': highlights.get('RevenueTTM'),
                'revenue_per_share': highlights.get('RevenuePerShareTTM'),
                'most_recent_quarter': highlights.get('MostRecentQuarter')
            }
        
        if 'Valuation' in fundamentals:
            valuation = fundamentals['Valuation']
            filtered['valuation_metrics'] = {
                'pe_ratio': valuation.get('TrailingPE'),
                'forward_pe': valuation.get('ForwardPE'),
                'price_to_sales': valuation.get('PriceSalesTTM'),
                'price_to_book': valuation.get('PriceBookMRQ'),
                'enterprise_value': valuation.get('EnterpriseValue'),
                'ev_to_revenue': valuation.get('EnterpriseValueRevenue'),
                'ev_to_ebitda': valuation.get('EnterpriseValueEbitda')
            }
        
        if 'Technicals' in fundamentals:
            technicals = fundamentals['Technicals']
            filtered['stock_technicals'] = {
                'beta': technicals.get('Beta'),
                '52_week_high': technicals.get('52WeekHigh'),
                '52_week_low': technicals.get('52WeekLow'),
                '50_day_ma': technicals.get('50DayMA'),
                '200_day_ma': technicals.get('200DayMA'),
                'short_interest': technicals.get('ShortPercent'),
                'short_ratio': technicals.get('ShortRatio')
            }
        
        if 'SplitsDividends' in fundamentals:
            dividends = fundamentals['SplitsDividends']
            filtered['dividend_info'] = {
                'forward_dividend_rate': dividends.get('ForwardAnnualDividendRate'),
                'forward_dividend_yield': dividends.get('ForwardAnnualDividendYield'),
                'payout_ratio': dividends.get('PayoutRatio'),
                'dividend_date': dividends.get('DividendDate'),
                'ex_dividend_date': dividends.get('ExDividendDate'),
                'last_split_date': dividends.get('LastSplitDate'),
                'last_split_factor': dividends.get('LastSplitFactor')
            }
        
        if 'AnalystRatings' in fundamentals:
            ratings = fundamentals['AnalystRatings']
            filtered['analyst_sentiment'] = {
                'avg_rating': ratings.get('Rating'),  # 1=Strong Buy, 5=Strong Sell
                'target_price': ratings.get('TargetPrice'),
                'strong_buy': ratings.get('StrongBuy'),
                'buy': ratings.get('Buy'),
                'hold': ratings.get('Hold'),
                'sell': ratings.get('Sell'),
                'strong_sell': ratings.get('StrongSell')
            }
        
        # Add institutional ownership summary (indicates confidence)
        if 'SharesStats' in fundamentals:
            shares = fundamentals['SharesStats']
            filtered['ownership_structure'] = {
                'shares_outstanding': shares.get('SharesOutstanding'),
                'percent_institutions': shares.get('PercentInstitutions'),
                'percent_insiders': shares.get('PercentInsiders'),
                'shares_float': shares.get('SharesFloat')
            }
        
        # Extract key financial statement data (most recent quarter)
        if 'Financials' in fundamentals:
            financials = fundamentals['Financials']
            
            # Balance Sheet - Financial Strength Indicators
            if 'Balance_Sheet' in financials and 'quarterly' in financials['Balance_Sheet']:
                bs_data = financials['Balance_Sheet']['quarterly']
                # Get most recent quarter
                latest_quarter = max(bs_data.keys()) if bs_data else None
                if latest_quarter:
                    bs = bs_data[latest_quarter]
                    filtered['balance_sheet'] = {
                        'total_assets': float(bs.get('totalAssets', 0)) / 1000000 if bs.get('totalAssets') else None,  # Convert to millions
                        'total_debt': float(bs.get('shortLongTermDebtTotal', 0)) / 1000000 if bs.get('shortLongTermDebtTotal') else None,
                        'cash_and_equivalents': float(bs.get('cashAndEquivalents', 0)) / 1000000 if bs.get('cashAndEquivalents') else None,
                        'net_debt': float(bs.get('netDebt', 0)) / 1000000 if bs.get('netDebt') else None,
                        'working_capital': float(bs.get('netWorkingCapital', 0)) / 1000000 if bs.get('netWorkingCapital') else None,
                        'shareholders_equity': float(bs.get('totalStockholderEquity', 0)) / 1000000 if bs.get('totalStockholderEquity') else None,
                        'debt_to_equity': None,  # Will calculate if both values exist
                        'quarter_date': latest_quarter
                    }
                    # Calculate debt-to-equity ratio
                    if filtered['balance_sheet']['total_debt'] and filtered['balance_sheet']['shareholders_equity']:
                        filtered['balance_sheet']['debt_to_equity'] = round(
                            filtered['balance_sheet']['total_debt'] / filtered['balance_sheet']['shareholders_equity'], 2
                        )
            
            # Income Statement - Profitability and Revenue Trends
            if 'Income_Statement' in financials and 'quarterly' in financials['Income_Statement']:
                is_data = financials['Income_Statement']['quarterly']
                # Get most recent quarter
                latest_quarter = max(is_data.keys()) if is_data else None
                if latest_quarter:
                    is_ = is_data[latest_quarter]
                    filtered['income_statement'] = {
                        'total_revenue': float(is_.get('totalRevenue', 0)) / 1000000 if is_.get('totalRevenue') else None,
                        'gross_profit': float(is_.get('grossProfit', 0)) / 1000000 if is_.get('grossProfit') else None,
                        'operating_income': float(is_.get('operatingIncome', 0)) / 1000000 if is_.get('operatingIncome') else None,
                        'net_income': float(is_.get('netIncome', 0)) / 1000000 if is_.get('netIncome') else None,
                        'ebitda': float(is_.get('ebitda', 0)) / 1000000 if is_.get('ebitda') else None,
                        'gross_margin': None,  # Will calculate
                        'operating_margin': None,  # Will calculate
                        'net_margin': None,  # Will calculate
                        'quarter_date': latest_quarter
                    }
                    # Calculate margins
                    if filtered['income_statement']['total_revenue'] and filtered['income_statement']['total_revenue'] > 0:
                        revenue = filtered['income_statement']['total_revenue']
                        if filtered['income_statement']['gross_profit']:
                            filtered['income_statement']['gross_margin'] = round(
                                (filtered['income_statement']['gross_profit'] / revenue) * 100, 2
                            )
                        if filtered['income_statement']['operating_income']:
                            filtered['income_statement']['operating_margin'] = round(
                                (filtered['income_statement']['operating_income'] / revenue) * 100, 2
                            )
                        if filtered['income_statement']['net_income']:
                            filtered['income_statement']['net_margin'] = round(
                                (filtered['income_statement']['net_income'] / revenue) * 100, 2
                            )
            
            # Cash Flow - Critical for PMCC (company sustainability)
            if 'Cash_Flow' in financials and 'quarterly' in financials['Cash_Flow']:
                cf_data = financials['Cash_Flow']['quarterly']
                # Get most recent quarter
                latest_quarter = max(cf_data.keys()) if cf_data else None
                if latest_quarter:
                    cf = cf_data[latest_quarter]
                    filtered['cash_flow'] = {
                        'operating_cash_flow': float(cf.get('totalCashFromOperatingActivities', 0)) / 1000000 if cf.get('totalCashFromOperatingActivities') else None,
                        'free_cash_flow': float(cf.get('freeCashFlow', 0)) / 1000000 if cf.get('freeCashFlow') else None,  # Key metric for email report
                        'capex': float(cf.get('capitalExpenditures', 0)) / 1000000 if cf.get('capitalExpenditures') else None,
                        'net_income': float(cf.get('netIncome', 0)) / 1000000 if cf.get('netIncome') else None,
                        'cash_change': float(cf.get('changeInCash', 0)) / 1000000 if cf.get('changeInCash') else None,
                        'dividends_paid': float(cf.get('dividendsPaid', 0)) / 1000000 if cf.get('dividendsPaid') else None,
                        'quarter_date': latest_quarter
                    }
        
        return filtered
        
    except Exception as e:
        print(f"Error filtering fundamental data: {e}")
        return fundamentals  # Return original if filtering fails

def save_data_to_file(data, ticker, filename=None):
    """Save collected data to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_enhanced_data_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Data saved to {filename}")
    except Exception as e:
        print(f"✗ Error saving data to file: {e}")

def main() -> None:
    """Main function to demonstrate enhanced data collection"""
    
    # Initialize API client
    api = APIClient(cfg.API_KEY)
    
    # Test ticker
    ticker = 'KSS'
    
    print("PMCC Enhanced Data Collection Script")
    print("=" * 50)
    
    # Collect all enhanced data
    enhanced_data = collect_enhanced_data(api, ticker)
    
    if enhanced_data:
        # Fundamental data is already filtered above
        
        # Display summary of collected data
        print(f"\n=== Data Collection Summary for {ticker} ===")
        for key, value in enhanced_data.items():
            if value:
                if isinstance(value, list):
                    print(f"{key}: {len(value)} items")
                elif isinstance(value, dict):
                    print(f"{key}: {len(value)} fields")
                else:
                    print(f"{key}: Available")
            else:
                print(f"{key}: No data")
        
        # Save complete data to file (with filtered fundamentals)
        save_data_to_file(enhanced_data, ticker)
        
        # Display sample of latest technical indicators
        print(f"\n=== Latest Technical Indicators for {ticker} ===")
        for indicator in ['volatility', 'rsi', 'atr', 'beta']:
            if indicator in enhanced_data and enhanced_data[indicator]:
                data = enhanced_data[indicator]
                if isinstance(data, list) and len(data) > 0:
                    latest = data[0]  # First item (most recent due to order='d')
                    if isinstance(latest, dict):
                        value = latest.get(indicator, 'N/A')
                        date = latest.get('date', 'N/A')
                        print(f"{indicator.upper()}: {value} (as of {date})")
        
        print(f"\n✓ Enhanced data collection completed for {ticker}")
        
    else:
        print(f"✗ Failed to collect enhanced data for {ticker}")

if __name__ == "__main__":
    main()
