#!/usr/bin/env python3
"""
Regression Test Suite

Comprehensive regression testing to ensure system stability across changes.
Tests core functionality, performance benchmarks, and integration points.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

# Add project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from algotrading_agent.config.settings import get_config
from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain
from algotrading_agent.components.trading_cost_calculator import TradingCostCalculator
from algotrading_agent.components.decision_engine import DecisionEngine


class RegressionTestSuite:
    """Comprehensive regression test suite"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_benchmarks': {},
            'regression_failures': [],
            'test_results': {}
        }
        self.logger = logging.getLogger("regression.suite")
        
        # Load baseline results if they exist
        self.baseline_file = "tests/regression/baseline_results.json"
        self.baseline = self._load_baseline()
        
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline regression test results"""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    baseline = json.load(f)
                    self.logger.info(f"Loaded baseline from {baseline['timestamp']}")
                    return baseline
            except Exception as e:
                self.logger.warning(f"Could not load baseline: {e}")
        
        return {}
    
    def _save_baseline(self):
        """Save current results as new baseline"""
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"Saved new baseline to {self.baseline_file}")
        except Exception as e:
            self.logger.error(f"Could not save baseline: {e}")
    
    async def run_full_regression_suite(self) -> Dict[str, Any]:
        """Run complete regression test suite"""
        print("🔄 Algotrading Agent - Regression Test Suite")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test categories
        test_categories = [
            ("Core Components", self._test_core_components),
            ("AI Integration", self._test_ai_integration),
            ("Trading Costs", self._test_trading_costs),
            ("News Processing", self._test_news_processing),
            ("Performance Benchmarks", self._test_performance_benchmarks),
            ("Configuration Validation", self._test_configuration),
            ("Correlation Analysis", self._test_correlation_analysis)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)
            
            try:
                category_results = await test_function()
                self.results['test_results'][category_name] = category_results
                
                # Update counters
                self.results['tests_run'] += category_results.get('tests_run', 0)
                self.results['tests_passed'] += category_results.get('tests_passed', 0)
                self.results['tests_failed'] += category_results.get('tests_failed', 0)
                
                # Check for regressions
                self._check_for_regressions(category_name, category_results)
                
            except Exception as e:
                self.logger.error(f"Category {category_name} failed: {e}")
                self.results['tests_failed'] += 1
                self.results['regression_failures'].append({
                    'category': category_name,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Generate final report
        self._generate_regression_report()
        
        return self.results
    
    async def _test_core_components(self) -> Dict[str, Any]:
        """Test core system components"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Configuration loading
        results['tests_run'] += 1
        try:
            config = get_config()
            assert config is not None
            assert 'ai_analyzer' in config.get_dict()
            results['tests_passed'] += 1
            results['details'].append("✅ Configuration loading")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ Configuration loading: {e}")
        
        # Test 2: Component initialization
        results['tests_run'] += 1
        try:
            config_dict = get_config().get_dict()
            cost_calculator = TradingCostCalculator(config_dict)
            assert cost_calculator is not None
            results['tests_passed'] += 1
            results['details'].append("✅ Component initialization")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ Component initialization: {e}")
        
        return results
    
    async def _test_ai_integration(self) -> Dict[str, Any]:
        """Test AI integration functionality"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: AI component creation
        results['tests_run'] += 1
        try:
            config = {
                'ai_analyzer': {
                    'enabled': True,
                    'provider': 'openai',
                    'fallback_enabled': True,
                    'ai_weight': 0.7,
                    'traditional_weight': 0.3,
                    'providers': {
                        'openai': {
                            'enabled': True,
                            'model': 'gpt-3.5-turbo',
                            'api_key_env': 'OPENAI_API_KEY'
                        }
                    }
                }
            }
            
            news_brain = NewsAnalysisBrain(config)
            assert news_brain is not None
            results['tests_passed'] += 1
            results['details'].append("✅ AI component creation")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ AI component creation: {e}")
        
        # Test 2: Fallback mechanism
        results['tests_run'] += 1
        try:
            # Test with invalid AI config (should fallback)
            config['ai_analyzer']['provider'] = 'invalid_provider'
            news_brain = NewsAnalysisBrain(config)
            
            # Handle both sync and async start methods
            start_result = news_brain.start()
            if hasattr(start_result, '__await__'):
                await start_result
            
            # Process test news (should work via fallback)
            test_news = [{
                'title': 'Test news title',
                'content': 'Test news content',
                'symbol': 'TEST'
            }]
            
            process_result = news_brain.process(test_news)
            if hasattr(process_result, '__await__'):
                processed = await process_result
            else:
                processed = process_result
            
            assert len(processed) > 0
            assert 'sentiment' in processed[0]
            
            # Clean up
            stop_result = news_brain.stop()
            if hasattr(stop_result, '__await__'):
                await stop_result
            
            results['tests_passed'] += 1
            results['details'].append("✅ AI fallback mechanism")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ AI fallback mechanism: {e}")
        
        return results
    
    async def _test_trading_costs(self) -> Dict[str, Any]:
        """Test trading cost calculations"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test different commission models
        test_configs = [
            ('zero', {'commission_model': 'zero'}),
            ('per_trade', {'commission_model': 'per_trade', 'per_trade_commission': 9.95}),
            ('per_share', {'commission_model': 'per_share', 'per_share_commission': 0.005, 'per_share_minimum': 1.00}),
            ('percentage', {'commission_model': 'percentage', 'percentage_commission': 0.001, 'percentage_minimum': 1.00})
        ]
        
        for model_name, config in test_configs:
            results['tests_run'] += 1
            try:
                full_config = {
                    'trading_costs': {
                        'paper_trading_costs': True,
                        **config,
                        'regulatory_fees': {
                            'sec_fee_rate': 0.0000278,
                            'taf_fee_rate': 0.000166
                        }
                    }
                }
                
                calculator = TradingCostCalculator(full_config)
                
                # Test round-trip calculation
                costs = calculator.calculate_round_trip_costs('AAPL', 100, 150.0, 155.0, True)
                
                assert 'total_costs' in costs
                assert 'net_pnl' in costs
                assert 'gross_pnl' in costs
                assert costs['gross_pnl'] == 500.0  # (155-150)*100
                
                # Store benchmark data
                self.results['performance_benchmarks'][f'trading_costs_{model_name}'] = {
                    'total_costs': costs['total_costs'],
                    'net_pnl': costs['net_pnl']
                }
                
                results['tests_passed'] += 1
                results['details'].append(f"✅ Trading costs - {model_name}")
            except Exception as e:
                results['tests_failed'] += 1
                results['details'].append(f"❌ Trading costs - {model_name}: {e}")
        
        return results
    
    async def _test_news_processing(self) -> Dict[str, Any]:
        """Test news processing pipeline"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test sentiment analysis consistency
        results['tests_run'] += 1
        try:
            config = {'ai_analyzer': {'enabled': False}}
            news_brain = NewsAnalysisBrain(config)
            
            # Handle start method
            start_result = news_brain.start()
            if hasattr(start_result, '__await__'):
                await start_result
            
            # Test consistent sentiment analysis
            test_news = [
                {
                    'title': 'Company beats earnings expectations with record revenue',
                    'content': 'The company reported exceptional growth with strong performance across all metrics.',
                    'symbol': 'TEST'
                }
            ]
            
            # Run analysis multiple times to check consistency
            results_list = []
            for _ in range(3):
                process_result = news_brain.process(test_news.copy())
                if hasattr(process_result, '__await__'):
                    processed = await process_result
                else:
                    processed = process_result
                
                if processed and len(processed) > 0:
                    sentiment = processed[0]['sentiment']['polarity']
                    results_list.append(sentiment)
            
            # Check consistency (should be identical for same input)
            assert len(set(results_list)) == 1, f"Inconsistent results: {results_list}"
            assert results_list[0] > 0, "Positive news should have positive sentiment"
            
            # Store benchmark
            self.results['performance_benchmarks']['sentiment_consistency'] = {
                'sentiment_value': results_list[0],
                'consistency_check': 'passed'
            }
            
            # Clean up
            stop_result = news_brain.stop()
            if hasattr(stop_result, '__await__'):
                await stop_result
            
            results['tests_passed'] += 1
            results['details'].append("✅ News sentiment consistency")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ News sentiment consistency: {e}")
        
        return results
    
    async def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: News processing speed
        results['tests_run'] += 1
        try:
            config = {'ai_analyzer': {'enabled': False}}
            news_brain = NewsAnalysisBrain(config)
            
            start_result = news_brain.start()
            if hasattr(start_result, '__await__'):
                await start_result
            
            # Generate test news
            test_news = []
            for i in range(50):  # Process 50 news items
                test_news.append({
                    'title': f'Test news item {i}',
                    'content': f'This is test content for news item {i} with some financial keywords like revenue and growth.',
                    'symbol': 'TEST'
                })
            
            # Time the processing
            start_time = time.time()
            
            process_result = news_brain.process(test_news)
            if hasattr(process_result, '__await__'):
                processed = await process_result
            else:
                processed = process_result
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Benchmark: should process 50 items in under 5 seconds
            assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, expected < 5.0s"
            assert len(processed) == 50, f"Expected 50 processed items, got {len(processed)}"
            
            # Store benchmark
            self.results['performance_benchmarks']['news_processing_speed'] = {
                'items_processed': len(processed),
                'processing_time_seconds': processing_time,
                'items_per_second': len(processed) / processing_time
            }
            
            stop_result = news_brain.stop()
            if hasattr(stop_result, '__await__'):
                await stop_result
            
            results['tests_passed'] += 1
            results['details'].append(f"✅ News processing speed: {processing_time:.2f}s for 50 items")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ News processing speed: {e}")
        
        return results
    
    async def _test_configuration(self) -> Dict[str, Any]:
        """Test configuration validation"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test configuration completeness
        results['tests_run'] += 1
        try:
            config = get_config()
            
            # Check required sections
            required_sections = [
                'news_scraper', 'news_filter', 'news_analysis_brain',
                'decision_engine', 'risk_manager', 'trading_costs', 'ai_analyzer'
            ]
            
            for section in required_sections:
                assert section in config.get_dict(), f"Missing config section: {section}"
            
            # Check AI configuration
            ai_config = config.get('ai_analyzer', {})
            assert 'provider' in ai_config, "Missing AI provider configuration"
            assert 'fallback_chain' in ai_config, "Missing AI fallback chain"
            
            results['tests_passed'] += 1
            results['details'].append("✅ Configuration completeness")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ Configuration completeness: {e}")
        
        return results
    
    async def _test_correlation_analysis(self) -> Dict[str, Any]:
        """Test correlation analysis functionality"""
        results = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Sentiment direction prediction
        results['tests_run'] += 1
        try:
            config = {'ai_analyzer': {'enabled': False}}
            news_brain = NewsAnalysisBrain(config)
            
            start_result = news_brain.start()
            if hasattr(start_result, '__await__'):
                await start_result
            
            # Test positive news
            positive_news = [{
                'title': 'Company beats earnings with record profits and strong growth',
                'content': 'Exceptional performance with revenue surge and breakthrough achievements.',
                'symbol': 'TEST'
            }]
            
            process_result = news_brain.process(positive_news)
            if hasattr(process_result, '__await__'):
                processed = await process_result
            else:
                processed = process_result
            
            positive_sentiment = processed[0]['sentiment']['polarity']
            
            # Test negative news  
            negative_news = [{
                'title': 'Company misses earnings with declining revenue and major concerns',
                'content': 'Poor performance with significant drop in profits and troubling outlook.',
                'symbol': 'TEST'
            }]
            
            process_result = news_brain.process(negative_news)
            if hasattr(process_result, '__await__'):
                processed = await process_result
            else:
                processed = process_result
            
            negative_sentiment = processed[0]['sentiment']['polarity']
            
            # Validate sentiment direction
            assert positive_sentiment > 0, f"Positive news should have positive sentiment, got {positive_sentiment}"
            assert negative_sentiment < 0, f"Negative news should have negative sentiment, got {negative_sentiment}"
            assert positive_sentiment > negative_sentiment, "Positive sentiment should be higher than negative"
            
            # Store benchmark
            self.results['performance_benchmarks']['sentiment_direction'] = {
                'positive_sentiment': positive_sentiment,
                'negative_sentiment': negative_sentiment,
                'sentiment_spread': positive_sentiment - negative_sentiment
            }
            
            stop_result = news_brain.stop()
            if hasattr(stop_result, '__await__'):
                await stop_result
            
            results['tests_passed'] += 1
            results['details'].append("✅ Sentiment direction prediction")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"❌ Sentiment direction prediction: {e}")
        
        return results
    
    def _check_for_regressions(self, category: str, current_results: Dict[str, Any]):
        """Check for performance regressions against baseline"""
        if not self.baseline or 'test_results' not in self.baseline:
            return
        
        baseline_category = self.baseline['test_results'].get(category, {})
        
        # Check test pass rate regression
        if 'tests_passed' in baseline_category and 'tests_run' in baseline_category:
            baseline_pass_rate = baseline_category['tests_passed'] / baseline_category['tests_run']
            current_pass_rate = current_results['tests_passed'] / current_results['tests_run']
            
            if current_pass_rate < baseline_pass_rate * 0.9:  # 10% tolerance
                self.results['regression_failures'].append({
                    'category': category,
                    'type': 'pass_rate_regression',
                    'baseline_pass_rate': baseline_pass_rate,
                    'current_pass_rate': current_pass_rate,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Check performance benchmarks
        if category in self.baseline.get('performance_benchmarks', {}):
            baseline_perf = self.baseline['performance_benchmarks'][category]
            current_perf = self.results['performance_benchmarks'].get(category, {})
            
            # Check for performance degradation (case-by-case basis)
            if 'processing_time_seconds' in baseline_perf and 'processing_time_seconds' in current_perf:
                if current_perf['processing_time_seconds'] > baseline_perf['processing_time_seconds'] * 1.5:
                    self.results['regression_failures'].append({
                        'category': category,
                        'type': 'performance_regression',
                        'metric': 'processing_time_seconds',
                        'baseline_value': baseline_perf['processing_time_seconds'],
                        'current_value': current_perf['processing_time_seconds'],
                        'timestamp': datetime.utcnow().isoformat()
                    })
    
    def _generate_regression_report(self):
        """Generate final regression test report"""
        print(f"\n" + "="*60)
        print("🎯 REGRESSION TEST REPORT")
        print("="*60)
        
        total_tests = self.results['tests_run']
        passed_tests = self.results['tests_passed']
        failed_tests = self.results['tests_failed']
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        
        if self.results['regression_failures']:
            print(f"\n⚠️  Regression Failures ({len(self.results['regression_failures'])}):")
            for failure in self.results['regression_failures']:
                print(f"   • {failure['category']}: {failure['type']}")
        else:
            print(f"\n✅ No regressions detected!")
        
        print(f"\n🚀 Performance Benchmarks:")
        for benchmark, data in self.results['performance_benchmarks'].items():
            print(f"   • {benchmark}: {data}")
        
        # Overall status
        if failed_tests == 0 and len(self.results['regression_failures']) == 0:
            print(f"\n🎉 REGRESSION SUITE PASSED - System is stable!")
            return True
        else:
            print(f"\n❌ REGRESSION SUITE FAILED - Review failures above")
            return False


async def main():
    """Run regression test suite"""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    suite = RegressionTestSuite()
    
    try:
        results = await suite.run_full_regression_suite()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"tests/regression/regression_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Ask if user wants to save as new baseline
        if results['tests_failed'] == 0 and len(results['regression_failures']) == 0:
            print(f"\n🤔 Save as new baseline? This will become the reference for future regression tests.")
            # For automated testing, we could add logic here to save baseline automatically
            # suite._save_baseline()
        
        return results['tests_failed'] == 0 and len(results['regression_failures']) == 0
        
    except Exception as e:
        print(f"❌ Regression suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)