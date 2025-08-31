#!/bin/bash
# Test script for the options screening application

echo "=============================================="
echo "OPTIONS SCREENING APP - TEST SUITE"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run test
run_test() {
    local test_name=$1
    local command=$2
    echo -e "${YELLOW}Testing: $test_name${NC}"
    echo "Command: $command"
    echo "---"
    
    if eval $command; then
        echo -e "${GREEN}✅ $test_name: PASSED${NC}\n"
        return 0
    else
        echo -e "${RED}❌ $test_name: FAILED${NC}\n"
        return 1
    fi
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Test 1: Basic Python and package test
echo "================================"
echo "Test 1: Python Environment"
echo "================================"
python --version
pip list | grep -E "pydantic|rich|click|dotenv" || echo "Some packages may be missing"
echo ""

# Test 2: Minimal test script
echo "================================"
echo "Test 2: Minimal Functionality"
echo "================================"
run_test "Minimal imports" "python test_minimal.py"

# Test 3: Simple app health check
echo "================================"
echo "Test 3: Simple App Health Check"
echo "================================"
run_test "Health check" "python main_simple.py health"

# Test 4: Simple app screening test
echo "================================"
echo "Test 4: Simple App Screening"
echo "================================"
run_test "Mock screening" "python main_simple.py screen --symbols AAPL,MSFT"

# Test 5: Environment variables test
echo "================================"
echo "Test 5: Environment Variables"
echo "================================"
echo "Checking .env file..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    # Check for API keys (without showing them)
    if grep -q "SCREENER_EODHD_API_KEY=" .env && grep -q "SCREENER_MARKETDATA_API_KEY=" .env; then
        echo -e "${GREEN}✅ API keys are configured in .env${NC}"
    else
        echo -e "${YELLOW}⚠️ Some API keys may be missing in .env${NC}"
    fi
else
    echo -e "${RED}❌ .env file not found${NC}"
fi
echo ""

# Summary
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo "Tests complete. Check results above."
echo ""
echo "Next steps if all tests passed:"
echo "1. Run: python main_simple.py test"
echo "2. Set SCREENER_SPECIFIC_SYMBOLS in .env"
echo "3. Run: python main_simple.py screen"
echo ""