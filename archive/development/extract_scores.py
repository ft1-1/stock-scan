#!/usr/bin/env python3
"""Extract and display all stock scores from the latest log file."""

import re
from pathlib import Path
import json
from datetime import datetime

def extract_scores_from_logs():
    """Extract scoring information from debug log."""
    log_file = Path("/home/deployuser/stock-scan/stock-scanner/logs/debug.log")
    
    if not log_file.exists():
        print("Debug log not found")
        return
    
    scores = {}
    
    # Pattern to match rejection lines
    reject_pattern = r"Rejecting (\w+): Score ([\d.]+) < ([\d.]+) threshold"
    
    # Pattern to match successful scores
    success_pattern = r"Symbol (\w+) .*?composite_score=([\d.]+)"
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for rejection pattern
            match = re.search(reject_pattern, line)
            if match:
                symbol = match.group(1)
                score = float(match.group(2))
                scores[symbol] = score
            
            # Check for success pattern
            match = re.search(success_pattern, line)
            if match:
                symbol = match.group(1)
                score = float(match.group(2))
                scores[symbol] = score
    
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Save to file
    output_file = f"scoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w') as f:
        f.write("STOCK SCORING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total stocks analyzed: {len(scores)}\n")
        f.write(f"Stocks above 60: {sum(1 for _, s in scores.items() if s >= 60)}\n")
        f.write(f"Stocks above 50: {sum(1 for _, s in scores.items() if s >= 50)}\n")
        f.write(f"Stocks above 40: {sum(1 for _, s in scores.items() if s >= 40)}\n")
        f.write(f"Stocks with 0 score: {sum(1 for _, s in scores.items() if s == 0)}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("TOP 50 SCORES:\n")
        f.write("-" * 50 + "\n")
        
        for i, (symbol, score) in enumerate(sorted_scores[:50], 1):
            f.write(f"{i:3}. {symbol:6} - Score: {score:6.2f}\n")
        
        if len(sorted_scores) > 50:
            f.write("\n" + "-" * 50 + "\n")
            f.write("BOTTOM 20 SCORES:\n")
            f.write("-" * 50 + "\n")
            for i, (symbol, score) in enumerate(sorted_scores[-20:], 1):
                f.write(f"{i:3}. {symbol:6} - Score: {score:6.2f}\n")
    
    print(f"Report saved to: {output_file}")
    
    # Print summary to console
    print("\nSCORING SUMMARY")
    print("=" * 50)
    print(f"Total stocks analyzed: {len(scores)}")
    print(f"Stocks above 60: {sum(1 for _, s in scores.items() if s >= 60)}")
    print(f"Stocks above 50: {sum(1 for _, s in scores.items() if s >= 50)}")
    print(f"Stocks above 40: {sum(1 for _, s in scores.items() if s >= 40)}")
    print(f"Stocks with 0 score: {sum(1 for _, s in scores.items() if s == 0)}")
    
    if sorted_scores:
        print("\nTOP 10 SCORES:")
        for i, (symbol, score) in enumerate(sorted_scores[:10], 1):
            print(f"{i:2}. {symbol:6} - Score: {score:6.2f}")

if __name__ == "__main__":
    extract_scores_from_logs()