#!/usr/bin/env python3
"""Debug the format error."""

def test_format():
    symbol = "AAPL"
    quote = {'close': 224.90, 'volume': 30411723, 'change_p': 1.5}
    technicals = {'rsi': 75.3, 'sma_20': 220.5, 'sma_50': 215.2, 'price_vs_sma20': 2.0}
    fundamentals = {'market_cap': 3500000000000, 'pe_ratio': 29.5, 'sector': 'Technology'}
    options = {'expirations': ['2024-01-19', '2024-02-16']}
    
    # Test each section separately
    try:
        test1 = f"""You are an expert stock analyst specializing in options trading opportunities. Analyze {symbol} and provide a comprehensive 0-100 rating."""
        print("✅ Basic format OK")
    except Exception as e:
        print(f"❌ Basic format error: {e}")
        return
    
    try:
        test2 = f"""## CURRENT MARKET DATA
**Symbol**: {symbol}
**Price**: ${quote.get('close', 0):.2f}
**Volume**: {quote.get('volume', 0):,}
**Change**: {quote.get('change_p', 0):.2f}%"""
        print("✅ Market data format OK")
    except Exception as e:
        print(f"❌ Market data format error: {e}")
        return
    
    try:
        rsi_format = f"{technicals.get('rsi', 'N/A'):.1f if isinstance(technicals.get('rsi'), (int, float)) else 'N/A'}"
        print(f"✅ RSI format OK: {rsi_format}")
    except Exception as e:
        print(f"❌ RSI format error: {e}")
        return
    
    try:
        market_cap_val = fundamentals.get('market_cap', 0)
        if market_cap_val:
            market_cap_str = f"${market_cap_val:,}"
        else:
            market_cap_str = 'N/A'
        print(f"✅ Market cap format OK: {market_cap_str}")
    except Exception as e:
        print(f"❌ Market cap format error: {e}")
        return
    
    print("All individual formats work!")

if __name__ == "__main__":
    test_format()