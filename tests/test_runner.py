#!/usr/bin/env python3
"""
Algotrading Agent Test Runner

Centralized test runner for all test categories with comprehensive reporting.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import unittest
from datetime import datetime
from typing import Dict, List, Any, Optional


class TestRunner:
    """Comprehensive test runner for all test categories"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'test_categories': {},
            'summary': {
                'total_tests': 0,
                'total_passed': 0,
                'total_failed': 0,
                'total_errors': 0,
                'execution_time': 0.0
            }
        }
        
        self.test_directories = {
            'unit': 'tests/unit',
            'integration': 'tests/integration', 
            'regression': 'tests/regression',
            'performance': 'tests/performance'
        }
        
    def run_all_tests(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all test categories or specified categories"""
        
        print("🧪 Algotrading Agent - Test Suite Runner")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Determine which categories to run
        if categories is None:
            categories = list(self.test_directories.keys())
        
        print(f"Running categories: {', '.join(categories)}")
        
        # Run each test category
        for category in categories:
            if category in self.test_directories:
                print(f"\n📂 {category.upper()} TESTS")
                print("-" * 40)
                
                try:
                    category_results = self._run_test_category(category)
                    self.results['test_categories'][category] = category_results
                    
                    # Update summary
                    self.results['summary']['total_tests'] += category_results.get('tests_run', 0)
                    self.results['summary']['total_passed'] += category_results.get('tests_passed', 0)
                    self.results['summary']['total_failed'] += category_results.get('tests_failed', 0)
                    self.results['summary']['total_errors'] += category_results.get('errors', 0)
                    
                except Exception as e:
                    print(f"❌ Category {category} failed: {e}")
                    self.results['test_categories'][category] = {
                        'status': 'error',
                        'error': str(e),
                        'tests_run': 0,
                        'tests_passed': 0,
                        'tests_failed': 0
                    }
                    self.results['summary']['total_errors'] += 1
            else:
                print(f"⚠️  Unknown test category: {category}")
        
        end_time = time.time()
        self.results['summary']['execution_time'] = end_time - start_time
        
        # Generate final report
        self._generate_final_report()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _run_test_category(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category"""
        
        test_dir = self.test_directories[category]
        
        if not os.path.exists(test_dir):
            print(f"⚠️  Test directory not found: {test_dir}")
            return {'status': 'skipped', 'reason': 'directory_not_found'}
        
        # Find all test files in the category
        test_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            print(f"⚠️  No test files found in {test_dir}")
            return {'status': 'skipped', 'reason': 'no_test_files'}
        
        print(f"Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"   • {os.path.basename(test_file)}")
        
        category_results = {
            'status': 'completed',
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': 0,
            'test_files': [],
            'execution_time': 0.0
        }
        
        category_start_time = time.time()
        
        # Run each test file
        for test_file in test_files:
            print(f"\n🔍 Running {os.path.basename(test_file)}...")
            
            try:
                file_results = self._run_test_file(test_file, category)
                category_results['test_files'].append(file_results)
                
                # Aggregate results
                category_results['tests_run'] += file_results.get('tests_run', 0)
                category_results['tests_passed'] += file_results.get('tests_passed', 0)
                category_results['tests_failed'] += file_results.get('tests_failed', 0)
                category_results['errors'] += file_results.get('errors', 0)
                
                # Print file summary
                status = "✅" if file_results.get('tests_failed', 0) == 0 and file_results.get('errors', 0) == 0 else "❌"
                print(f"   {status} {file_results.get('tests_passed', 0)}/{file_results.get('tests_run', 0)} passed")
                
            except Exception as e:
                print(f"   ❌ Error running {test_file}: {e}")
                category_results['errors'] += 1
                category_results['test_files'].append({
                    'file': test_file,
                    'status': 'error',
                    'error': str(e)
                })
        
        category_end_time = time.time()
        category_results['execution_time'] = category_end_time - category_start_time
        
        # Print summary for category
        print(f"\n📊 {category.upper()} SUMMARY:")
        print(f"   Tests Run: {category_results['tests_run']}")
        print(f"   Passed: {category_results['tests_passed']}")
        print(f"   Failed: {category_results['tests_failed']}")
        print(f"   Errors: {category_results['errors']}")
        print(f"   Time: {category_results['execution_time']:.2f}s")
        
        return category_results
    
    def _run_test_file(self, test_file: str, category: str) -> Dict[str, Any]:
        """Run a specific test file"""
        
        file_results = {
            'file': test_file,
            'status': 'unknown',
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': 0,
            'output': '',
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            if category == 'unit':
                # Run unit tests with unittest
                result = self._run_unittest_file(test_file)
            elif category in ['integration', 'regression', 'performance']:
                # Run integration/regression tests directly
                result = self._run_direct_test_file(test_file)
            else:
                raise ValueError(f"Unknown test category: {category}")
            
            file_results.update(result)
            
        except Exception as e:
            file_results['status'] = 'error'
            file_results['errors'] = 1
            file_results['output'] = str(e)
        
        file_results['execution_time'] = time.time() - start_time
        
        return file_results
    
    def _run_unittest_file(self, test_file: str) -> Dict[str, Any]:
        """Run unittest-based test file"""
        
        try:
            # Run unittest with subprocess to capture output
            cmd = [sys.executable, '-m', 'unittest', test_file.replace('/', '.').replace('.py', '')]
            
            # Change to project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            output = result.stdout + result.stderr
            
            # Parse unittest output
            tests_run = 0
            tests_failed = 0
            errors = 0
            
            for line in output.split('\n'):
                if 'Ran ' in line and ' test' in line:
                    # Extract number of tests run
                    import re
                    match = re.search(r'Ran (\d+) test', line)
                    if match:
                        tests_run = int(match.group(1))
                elif 'FAILED' in line:
                    # Extract failures and errors
                    if 'failures=' in line:
                        match = re.search(r'failures=(\d+)', line)
                        if match:
                            tests_failed += int(match.group(1))
                    if 'errors=' in line:
                        match = re.search(r'errors=(\d+)', line)
                        if match:
                            errors += int(match.group(1))
            
            tests_passed = tests_run - tests_failed - errors
            
            return {
                'status': 'completed',
                'tests_run': tests_run,
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'errors': errors,
                'output': output,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'errors': 1,
                'output': 'Test execution timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'errors': 1,
                'output': str(e)
            }
    
    def _run_direct_test_file(self, test_file: str) -> Dict[str, Any]:
        """Run test file directly (for integration/regression tests)"""
        
        try:
            # Run the test file directly
            cmd = [sys.executable, test_file]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for integration tests
            )
            
            output = result.stdout + result.stderr
            
            # For direct execution, we consider success/failure based on return code
            if result.returncode == 0:
                return {
                    'status': 'passed',
                    'tests_run': 1,
                    'tests_passed': 1,
                    'tests_failed': 0,
                    'errors': 0,
                    'output': output,
                    'return_code': result.returncode
                }
            else:
                return {
                    'status': 'failed',
                    'tests_run': 1,
                    'tests_passed': 0,
                    'tests_failed': 1,
                    'errors': 0,
                    'output': output,
                    'return_code': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'tests_run': 1,
                'tests_passed': 0,
                'tests_failed': 0,
                'errors': 1,
                'output': 'Test execution timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'tests_run': 1,
                'tests_passed': 0,
                'tests_failed': 0,
                'errors': 1,
                'output': str(e)
            }
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        
        print(f"\n" + "="*60)
        print("🎯 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        summary = self.results['summary']
        total_tests = summary['total_tests']
        passed_tests = summary['total_passed']
        failed_tests = summary['total_failed']
        errors = summary['total_errors']
        execution_time = summary['execution_time']
        
        print(f"\n📊 Overall Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Errors: {errors}")
        print(f"   Execution Time: {execution_time:.2f}s")
        
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            print(f"   Pass Rate: {pass_rate:.1f}%")
        
        # Category breakdown
        print(f"\n📂 Category Breakdown:")
        for category, results in self.results['test_categories'].items():
            status_icon = "✅" if results.get('tests_failed', 1) == 0 and results.get('errors', 1) == 0 else "❌"
            print(f"   {status_icon} {category.upper()}: {results.get('tests_passed', 0)}/{results.get('tests_run', 0)} passed")
        
        # Overall status
        if failed_tests == 0 and errors == 0:
            print(f"\n🎉 ALL TESTS PASSED - System is healthy!")
            return True
        else:
            print(f"\n⚠️  SOME TESTS FAILED - Review failures above")
            return False
    
    def _save_results(self):
        """Save test results to file"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"tests/test_results_{timestamp}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")


def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Algotrading Agent Test Runner')
    parser.add_argument(
        '--categories', 
        nargs='+', 
        choices=['unit', 'integration', 'regression', 'performance'],
        help='Test categories to run (default: all)'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    runner = TestRunner()
    results = runner.run_all_tests(args.categories)
    
    # Exit with appropriate code
    summary = results['summary']
    success = summary['total_failed'] == 0 and summary['total_errors'] == 0
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()