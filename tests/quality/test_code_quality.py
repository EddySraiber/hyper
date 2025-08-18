#!/usr/bin/env python3
"""
Code Quality Gates Validation

Validates code quality metrics, test coverage,
and maintains coding standards for production readiness.
"""

import os
import subprocess
import pytest
from pathlib import Path


class TestCodeQuality:
    """Code quality validation tests"""
    
    def test_test_coverage_threshold(self):
        """Ensure test coverage meets minimum threshold"""
        # Run coverage analysis
        result = subprocess.run([
            'pytest', '--cov=algotrading_agent', '--cov-report=term-missing',
            '--cov-fail-under=75', 'tests/'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Test coverage below 75% threshold:\n{result.stdout}"
    
    def test_critical_component_coverage(self):
        """Ensure critical safety components have high test coverage"""
        critical_components = [
            'algotrading_agent/trading/guardian_service.py',
            'algotrading_agent/trading/position_protector.py',
            'algotrading_agent/trading/bracket_order_manager.py',
            'algotrading_agent/components/risk_manager.py'
        ]
        
        for component in critical_components:
            result = subprocess.run([
                'pytest', f'--cov={component}', '--cov-report=term-missing',
                '--cov-fail-under=90', 'tests/'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Critical component {component} below 90% coverage"
    
    def test_no_security_vulnerabilities(self):
        """Ensure no known security vulnerabilities"""
        result = subprocess.run([
            'bandit', '-r', 'algotrading_agent/', '-f', 'json'
        ], capture_output=True, text=True)
        
        # Bandit returns 0 for no issues, 1 for issues found
        assert result.returncode == 0, f"Security vulnerabilities found:\n{result.stdout}"
    
    def test_code_style_compliance(self):
        """Ensure code follows style guidelines"""
        result = subprocess.run([
            'flake8', 'algotrading_agent/', '--max-line-length=127', '--extend-ignore=E203,W503'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Code style violations found:\n{result.stdout}"
    
    def test_no_todo_fixme_in_production(self):
        """Ensure no TODO/FIXME comments in production code"""
        production_files = Path('algotrading_agent').rglob('*.py')
        
        todo_files = []
        for file_path in production_files:
            with open(file_path, 'r') as f:
                content = f.read().upper()
                if 'TODO' in content or 'FIXME' in content:
                    todo_files.append(str(file_path))
        
        assert len(todo_files) == 0, f"TODO/FIXME found in production files: {todo_files}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
