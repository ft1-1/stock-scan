# Options Screening System - Test Suite

This directory contains comprehensive tests for the options screening system, providing quality assurance, performance validation, and regression testing.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_providers_extended.py      # Provider layer tests
│   ├── test_analytics_extended.py      # Analytics components tests
│   ├── test_ai_analysis_extended.py    # AI analysis tests
│   └── test_core.py                    # Core models and workflow tests
├── integration/                # Integration tests
│   ├── test_workflows.py              # End-to-end workflow tests
│   └── test_data_pipeline.py          # Data pipeline integration tests
├── performance/                # Performance and load tests
│   ├── test_load_performance.py       # Load testing suite
│   ├── test_benchmark_analytics.py    # Analytics benchmarks
│   └── test_memory_performance.py     # Memory usage tests
├── utils/                      # Test utilities
│   ├── mock_generators.py             # Mock data generators
│   ├── validators.py                  # Test validation helpers
│   └── performance_helpers.py         # Performance testing utilities
└── fixtures/                   # Test data fixtures
```

## Test Categories

### Unit Tests
- **Provider Tests**: EODHD, MarketData, cache, validators
- **Analytics Tests**: Technical indicators, momentum, options selection, scoring
- **AI Analysis Tests**: Claude client, cost management, data packaging
- **Core Tests**: Models, configuration, workflow engine

### Integration Tests
- **Workflow Tests**: Complete screening workflows
- **Data Pipeline Tests**: Provider coordination, data flow validation
- **Error Handling**: Resilience and recovery testing

### Performance Tests
- **Load Tests**: Concurrent processing, throughput measurement
- **Benchmark Tests**: Component performance baselines
- **Memory Tests**: Memory usage, leak detection
- **Stress Tests**: Resource limit testing

## Running Tests

### Quick Test Run
```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_analytics_extended.py

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Full Test Suite
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                 # Unit tests only
pytest -m integration          # Integration tests only
pytest -m performance          # Performance tests only
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/performance/ -m performance

# Run load tests
pytest tests/performance/ -m "performance and slow"

# Run memory tests
pytest tests/performance/test_memory_performance.py
```

### Test Configuration
```bash
# Run with specific markers
pytest -m "unit and not slow"
pytest -m "integration and not api"

# Parallel execution
pytest -n auto

# Generate detailed reports
pytest --html=report.html --self-contained-html
```

## Test Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests across components
- `performance`: Performance and benchmark tests
- `slow`: Tests taking >10 seconds
- `api`: Tests requiring external API access
- `ai`: Tests requiring AI/Claude API
- `memory`: Memory usage and leak tests
- `load`: Load testing scenarios

## Test Data Generation

The test suite includes sophisticated mock data generators:

```python
# Using test data generator
def test_example(test_data_generator):
    # Generate stock quotes
    quote = test_data_generator.generate_stock_quote("AAPL", base_price=150.0)
    
    # Generate options chain
    options = test_data_generator.generate_options_chain("AAPL", 150.0, days_to_exp=30)
    
    # Generate historical data
    history = test_data_generator.generate_historical_data("AAPL", days=252)
```

## Performance Benchmarking

Performance tests establish baselines and detect regressions:

```python
# Example performance test
@pytest.mark.performance
def test_technical_indicators_performance(test_data_generator):
    data = test_data_generator.generate_historical_data("BENCH", days=252)
    
    # Benchmark calculation time
    start_time = time.time()
    indicators = TechnicalIndicators.calculate_all_indicators(data)
    execution_time = time.time() - start_time
    
    # Assert performance baseline
    assert execution_time < 1.0, f"Performance regression: {execution_time:.3f}s"
```

## Coverage Requirements

- **Overall Coverage**: >80%
- **Critical Components**: >95%
- **Error Handling**: 100%

Coverage reports are generated automatically:
- Terminal output with `--cov-report=term`
- HTML report in `htmlcov/`
- XML report for CI/CD in `coverage.xml`

## Continuous Integration

Tests run automatically on:
- Every push to main/develop branches
- Pull requests
- Daily scheduled runs
- Performance tests on schedule

See `.github/workflows/test.yml` for CI configuration.

## Test Utilities

### Mock Data Generators
- `MockDataFactory`: Creates realistic financial data
- `OptionsChainGenerator`: Generates options chains with proper Greeks
- `MarketDataGenerator`: Market conditions and scenarios
- `APIResponseMocker`: Mock API responses

### Validation Helpers
- `TestValidators`: Financial data validation
- `DataQualityChecker`: Data quality metrics
- `TestAssertions`: Custom financial assertions

### Performance Helpers
- `PerformanceMonitor`: Resource usage monitoring
- `LoadTestHelper`: Concurrent load testing
- `BenchmarkRunner`: Performance benchmarking

## Writing Tests

### Test Naming Convention
```python
class TestComponentName:
    def test_specific_functionality(self):
        """Test description."""
        pass
    
    def test_error_condition_handling(self):
        """Test error handling."""
        pass
    
    @pytest.mark.slow
    def test_performance_characteristic(self):
        """Performance test."""
        pass
```

### Test Structure
```python
def test_example_functionality(fixture1, fixture2):
    """Test description explaining what is being tested."""
    # Arrange - Set up test data
    test_data = create_test_data()
    
    # Act - Execute the functionality
    result = function_under_test(test_data)
    
    # Assert - Verify results
    assert result.is_valid()
    assert result.value > 0
```

### Error Testing
```python
def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError, match="Invalid parameter"):
        function_under_test(invalid_input)
```

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async operations."""
    result = await async_function()
    assert result is not None
```

## Debugging Tests

### Running Single Test
```bash
pytest tests/unit/test_analytics_extended.py::TestTechnicalIndicators::test_rsi_calculation -v
```

### Debug Mode
```bash
pytest --pdb  # Drop into debugger on failure
pytest --lf   # Run last failed tests
pytest --ff   # Run failed tests first
```

### Verbose Output
```bash
pytest -v --tb=long  # Detailed tracebacks
pytest -s            # Show print statements
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on others
2. **Mock External Dependencies**: Use mocks for APIs, databases, external services
3. **Clear Assertions**: Use descriptive assertion messages
4. **Performance Baselines**: Set realistic performance expectations
5. **Error Coverage**: Test both success and failure paths
6. **Data Validation**: Validate all test data meets requirements
7. **Clean Up**: Ensure tests clean up resources properly

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate test markers
3. Include performance tests for new algorithms
4. Update this README if adding new test categories
5. Ensure tests pass in CI/CD pipeline
6. Maintain test coverage above thresholds

## Troubleshooting

### Common Issues

**Import Errors**: Ensure `src/` is in Python path
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**API Rate Limits**: Use mock responses for external APIs
```python
with aioresponses() as m:
    m.get("https://api.example.com", payload={"data": "mock"})
```

**Memory Issues**: Run tests with memory monitoring
```bash
pytest tests/performance/test_memory_performance.py -v
```

**Slow Tests**: Use performance markers to skip slow tests
```bash
pytest -m "not slow"
```