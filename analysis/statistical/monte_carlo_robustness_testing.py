#!/usr/bin/env python3
"""
Monte Carlo Robustness Testing Framework for Enhanced Trading System

This module performs rigorous Monte Carlo simulations to test system robustness
across different market conditions, parameter variations, and stress scenarios.

Key Features:
- Bootstrap sampling for statistical confidence
- Parameter sensitivity analysis
- Regime-dependent performance testing
- Stress testing under extreme market conditions
- Risk model validation

Dr. Sarah Chen - Quantitative Finance Expert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.optimize import minimize
import json
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MonteCarloParameters:
    """Parameters for Monte Carlo simulation"""
    n_simulations: int = 1000
    confidence_levels: List[float] = None
    bootstrap_sample_size: int = 100
    parameter_ranges: Dict[str, Tuple[float, float]] = None
    stress_scenarios: List[str] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.parameter_ranges is None:
            self.parameter_ranges = {
                'confidence_threshold': (0.05, 0.30),
                'sentiment_weight': (0.2, 0.6),
                'regime_weight': (0.1, 0.4),
                'options_weight': (0.1, 0.4),
                'volatility_adjustment': (0.5, 2.0)
            }
        if self.stress_scenarios is None:
            self.stress_scenarios = ['black_swan', 'high_volatility', 'bear_market', 'correlation_breakdown']


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""
    simulation_id: str
    n_simulations: int
    
    # Performance distributions
    return_distribution: List[float]
    sharpe_distribution: List[float]
    drawdown_distribution: List[float]
    win_rate_distribution: List[float]
    
    # Statistical measures
    expected_return: float
    return_std: float
    return_confidence_intervals: Dict[float, Tuple[float, float]]
    
    expected_sharpe: float
    sharpe_std: float
    sharpe_confidence_intervals: Dict[float, Tuple[float, float]]
    
    expected_drawdown: float
    drawdown_std: float
    worst_case_drawdown: float
    
    # Risk metrics
    var_estimates: Dict[float, float]
    expected_shortfall: Dict[float, float]
    tail_risk_metrics: Dict[str, float]
    
    # Robustness measures
    parameter_sensitivity: Dict[str, float]
    stress_test_results: Dict[str, Dict[str, float]]
    regime_performance: Dict[str, Dict[str, float]]
    
    # Model validation
    model_validity_score: float
    statistical_significance: bool
    recommendations: List[str]
    warnings: List[str]


class MarketScenarioGenerator:
    """Generates realistic market scenarios for Monte Carlo testing"""
    
    def __init__(self):
        self.logger = logging.getLogger("market_scenario_generator")
        
    def generate_price_paths(self, 
                           initial_price: float,
                           n_days: int,
                           n_paths: int,
                           volatility: float = 0.2,
                           drift: float = 0.08,
                           regime: str = 'normal') -> np.ndarray:
        """Generate realistic price paths using geometric Brownian motion with regime adjustments"""
        
        dt = 1/252  # Daily time step
        
        # Regime-specific adjustments
        if regime == 'bull_market':
            drift *= 1.5
            volatility *= 0.8
        elif regime == 'bear_market':
            drift *= -0.5
            volatility *= 1.3
        elif regime == 'high_volatility':
            volatility *= 2.0
        elif regime == 'black_swan':
            volatility *= 3.0
            # Add jump component
            jump_probability = 0.05
            jump_magnitude = -0.15
        
        # Generate random walks
        random_shocks = np.random.normal(0, 1, (n_paths, n_days))
        
        # Add jumps for black swan scenario
        if regime == 'black_swan':
            jump_times = np.random.binomial(1, jump_probability, (n_paths, n_days))
            jump_sizes = np.random.normal(jump_magnitude, 0.05, (n_paths, n_days))
            random_shocks += jump_times * jump_sizes
        
        # Calculate price paths
        price_changes = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks
        log_prices = np.cumsum(price_changes, axis=1)
        log_prices = np.column_stack([np.zeros(n_paths), log_prices])
        
        price_paths = initial_price * np.exp(log_prices)
        
        return price_paths
    
    def generate_correlation_matrix(self, 
                                  n_assets: int, 
                                  base_correlation: float = 0.3,
                                  scenario: str = 'normal') -> np.ndarray:
        """Generate correlation matrix for different market scenarios"""
        
        if scenario == 'correlation_breakdown':
            # Low correlations
            correlations = np.random.uniform(0.0, 0.2, (n_assets, n_assets))
        elif scenario == 'crisis':
            # High correlations during crisis
            correlations = np.random.uniform(0.7, 0.9, (n_assets, n_assets))
        else:
            # Normal correlations
            correlations = np.random.uniform(base_correlation - 0.1, base_correlation + 0.1, (n_assets, n_assets))
        
        # Make symmetric and set diagonal to 1
        correlations = (correlations + correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlations)
        eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return correlations
    
    def generate_news_sentiment_scenarios(self, 
                                        n_days: int,
                                        base_volatility: float = 0.3,
                                        regime: str = 'normal') -> np.ndarray:
        """Generate news sentiment time series for different regimes"""
        
        if regime == 'bear_market':
            # More negative sentiment, higher volatility
            mean_sentiment = -0.2
            sentiment_vol = base_volatility * 1.5
        elif regime == 'bull_market':
            # More positive sentiment, lower volatility
            mean_sentiment = 0.2
            sentiment_vol = base_volatility * 0.8
        elif regime == 'high_volatility':
            # Neutral mean but high volatility
            mean_sentiment = 0.0
            sentiment_vol = base_volatility * 2.0
        else:
            mean_sentiment = 0.0
            sentiment_vol = base_volatility
        
        # Generate AR(1) process for sentiment persistence
        phi = 0.3  # Autocorrelation coefficient
        innovations = np.random.normal(0, sentiment_vol, n_days)
        
        sentiment = np.zeros(n_days)
        sentiment[0] = mean_sentiment + innovations[0]
        
        for t in range(1, n_days):
            sentiment[t] = phi * sentiment[t-1] + (1-phi) * mean_sentiment + innovations[t]
        
        # Clip to [-1, 1] range
        sentiment = np.clip(sentiment, -1.0, 1.0)
        
        return sentiment


class ParameterSensitivityAnalyzer:
    """Analyzes system sensitivity to parameter changes"""
    
    def __init__(self):
        self.logger = logging.getLogger("parameter_sensitivity_analyzer")
        
    def analyze_sensitivity(self, 
                          base_performance: float,
                          parameter_ranges: Dict[str, Tuple[float, float]],
                          performance_function: Callable,
                          n_samples: int = 100) -> Dict[str, float]:
        """Analyze sensitivity to parameter changes using Sobol indices"""
        
        sensitivities = {}
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            # Generate parameter samples
            param_values = np.linspace(min_val, max_val, n_samples)
            performance_values = []
            
            for param_val in param_values:
                # Create parameter dict with this value
                params = {param_name: param_val}
                performance = performance_function(params)
                performance_values.append(performance)
            
            # Calculate sensitivity as normalized standard deviation
            performance_std = np.std(performance_values)
            sensitivity = performance_std / abs(base_performance) if base_performance != 0 else 0
            sensitivities[param_name] = sensitivity
            
        return sensitivities
    
    def find_optimal_parameters(self,
                              parameter_ranges: Dict[str, Tuple[float, float]],
                              objective_function: Callable,
                              maximize: bool = True) -> Dict[str, float]:
        """Find optimal parameters using optimization"""
        
        # Define bounds
        bounds = [(min_val, max_val) for min_val, max_val in parameter_ranges.values()]
        param_names = list(parameter_ranges.keys())
        
        # Initial guess (middle of ranges)
        x0 = [(min_val + max_val) / 2 for min_val, max_val in parameter_ranges.values()]
        
        # Objective function wrapper
        def objective(x):
            params = dict(zip(param_names, x))
            result = objective_function(params)
            return -result if maximize else result
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # Return optimal parameters
        optimal_params = dict(zip(param_names, result.x))
        return optimal_params


class MonteCarloEngine:
    """Main Monte Carlo simulation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("monte_carlo_engine")
        
        # Initialize components
        self.scenario_generator = MarketScenarioGenerator()
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer()
        
        # Simulation state
        self.results_cache = {}
        
    async def run_monte_carlo_simulation(self,
                                       system_performance_func: Callable,
                                       mc_params: MonteCarloParameters,
                                       market_context: Dict[str, Any]) -> MonteCarloResults:
        """Run comprehensive Monte Carlo simulation"""
        
        self.logger.info(f"🎲 Starting Monte Carlo simulation with {mc_params.n_simulations} iterations")
        
        simulation_id = f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize result collectors
        return_distribution = []
        sharpe_distribution = []
        drawdown_distribution = []
        win_rate_distribution = []
        
        # Run simulations
        for i in range(mc_params.n_simulations):
            if i % 100 == 0:
                self.logger.info(f"   Progress: {i}/{mc_params.n_simulations} ({i/mc_params.n_simulations*100:.1f}%)")
            
            # Generate market scenario
            scenario = self._generate_single_scenario(market_context, mc_params)
            
            # Run system simulation
            performance = await system_performance_func(scenario)
            
            # Collect results
            return_distribution.append(performance.get('total_return_pct', 0.0))
            sharpe_distribution.append(performance.get('sharpe_ratio', 0.0))
            drawdown_distribution.append(performance.get('max_drawdown_pct', 0.0))
            win_rate_distribution.append(performance.get('win_rate_pct', 0.0))
        
        # Calculate statistical measures
        results = self._calculate_monte_carlo_statistics(
            simulation_id, mc_params, return_distribution, sharpe_distribution,
            drawdown_distribution, win_rate_distribution
        )
        
        # Run additional analyses
        results.parameter_sensitivity = await self._analyze_parameter_sensitivity(
            system_performance_func, mc_params, market_context
        )
        
        results.stress_test_results = await self._run_stress_tests(
            system_performance_func, mc_params, market_context
        )
        
        results.regime_performance = await self._analyze_regime_performance(
            system_performance_func, market_context
        )
        
        # Validate model and generate recommendations
        results.model_validity_score = self._calculate_model_validity(results)
        results.statistical_significance = self._test_statistical_significance(results)
        results.recommendations, results.warnings = self._generate_recommendations(results)
        
        self.logger.info("✅ Monte Carlo simulation completed")
        return results
    
    def _generate_single_scenario(self, 
                                market_context: Dict[str, Any],
                                mc_params: MonteCarloParameters) -> Dict[str, Any]:
        """Generate a single market scenario for simulation"""
        
        # Randomly select market regime
        regimes = ['normal', 'bull_market', 'bear_market', 'high_volatility', 'black_swan']
        regime_weights = [0.6, 0.15, 0.15, 0.08, 0.02]  # Normal is most common
        regime = np.random.choice(regimes, p=regime_weights)
        
        # Generate price paths
        n_assets = len(market_context.get('symbols', ['SPY', 'QQQ', 'AAPL']))
        n_days = market_context.get('simulation_days', 120)
        
        price_paths = {}
        for i, symbol in enumerate(market_context.get('symbols', ['SPY', 'QQQ', 'AAPL'])):
            initial_price = 100 + i * 50  # Different starting prices
            volatility = np.random.uniform(0.15, 0.35) if regime == 'normal' else None
            paths = self.scenario_generator.generate_price_paths(
                initial_price, n_days, 1, volatility=volatility, regime=regime
            )
            price_paths[symbol] = paths[0]  # Take first path
        
        # Generate sentiment scenarios
        sentiment_data = self.scenario_generator.generate_news_sentiment_scenarios(
            n_days, regime=regime
        )
        
        # Generate correlation matrix
        correlation_matrix = self.scenario_generator.generate_correlation_matrix(
            n_assets, scenario=regime if regime != 'normal' else 'normal'
        )
        
        # Randomize parameters within ranges
        randomized_params = {}
        for param, (min_val, max_val) in mc_params.parameter_ranges.items():
            randomized_params[param] = np.random.uniform(min_val, max_val)
        
        scenario = {
            'regime': regime,
            'price_paths': price_paths,
            'sentiment_data': sentiment_data,
            'correlation_matrix': correlation_matrix,
            'randomized_params': randomized_params,
            'market_context': market_context
        }
        
        return scenario
    
    def _calculate_monte_carlo_statistics(self,
                                        simulation_id: str,
                                        mc_params: MonteCarloParameters,
                                        returns: List[float],
                                        sharpes: List[float],
                                        drawdowns: List[float],
                                        win_rates: List[float]) -> MonteCarloResults:
        """Calculate comprehensive Monte Carlo statistics"""
        
        # Convert to numpy arrays
        returns_array = np.array(returns)
        sharpes_array = np.array(sharpes)
        drawdowns_array = np.array(drawdowns)
        win_rates_array = np.array(win_rates)
        
        # Calculate confidence intervals
        return_cis = {}
        sharpe_cis = {}
        
        for conf_level in mc_params.confidence_levels:
            alpha = 1 - conf_level
            
            # Returns confidence intervals
            return_lower = np.percentile(returns_array, alpha/2 * 100)
            return_upper = np.percentile(returns_array, (1 - alpha/2) * 100)
            return_cis[conf_level] = (return_lower, return_upper)
            
            # Sharpe confidence intervals
            sharpe_lower = np.percentile(sharpes_array, alpha/2 * 100)
            sharpe_upper = np.percentile(sharpes_array, (1 - alpha/2) * 100)
            sharpe_cis[conf_level] = (sharpe_lower, sharpe_upper)
        
        # Calculate VaR and Expected Shortfall
        var_estimates = {}
        expected_shortfall = {}
        
        for conf_level in mc_params.confidence_levels:
            alpha = 1 - conf_level
            var_threshold = np.percentile(returns_array, alpha * 100)
            var_estimates[conf_level] = var_threshold
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns_array[returns_array <= var_threshold]
            expected_shortfall[conf_level] = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        
        # Calculate tail risk metrics
        tail_risk_metrics = {
            'skewness': stats.skew(returns_array),
            'kurtosis': stats.kurtosis(returns_array),
            'jarque_bera_stat': stats.jarque_bera(returns_array)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns_array)[1]
        }
        
        results = MonteCarloResults(
            simulation_id=simulation_id,
            n_simulations=mc_params.n_simulations,
            
            # Distributions
            return_distribution=returns,
            sharpe_distribution=sharpes,
            drawdown_distribution=drawdowns,
            win_rate_distribution=win_rates,
            
            # Statistical measures
            expected_return=np.mean(returns_array),
            return_std=np.std(returns_array),
            return_confidence_intervals=return_cis,
            
            expected_sharpe=np.mean(sharpes_array),
            sharpe_std=np.std(sharpes_array),
            sharpe_confidence_intervals=sharpe_cis,
            
            expected_drawdown=np.mean(drawdowns_array),
            drawdown_std=np.std(drawdowns_array),
            worst_case_drawdown=np.max(drawdowns_array),
            
            # Risk metrics
            var_estimates=var_estimates,
            expected_shortfall=expected_shortfall,
            tail_risk_metrics=tail_risk_metrics,
            
            # Initialize other fields
            parameter_sensitivity={},
            stress_test_results={},
            regime_performance={},
            model_validity_score=0.0,
            statistical_significance=False,
            recommendations=[],
            warnings=[]
        )
        
        return results
    
    async def _analyze_parameter_sensitivity(self,
                                           system_func: Callable,
                                           mc_params: MonteCarloParameters,
                                           market_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze parameter sensitivity using variance-based methods"""
        
        sensitivities = {}
        base_scenario = self._generate_single_scenario(market_context, mc_params)
        base_performance = await system_func(base_scenario)
        base_return = base_performance.get('total_return_pct', 0.0)
        
        for param_name, (min_val, max_val) in mc_params.parameter_ranges.items():
            param_performances = []
            
            # Sample parameter space
            param_values = np.linspace(min_val, max_val, 20)
            
            for param_val in param_values:
                # Create modified scenario
                scenario = base_scenario.copy()
                scenario['randomized_params'][param_name] = param_val
                
                # Run system
                performance = await system_func(scenario)
                param_performances.append(performance.get('total_return_pct', 0.0))
            
            # Calculate sensitivity as coefficient of variation
            param_std = np.std(param_performances)
            sensitivity = param_std / abs(base_return) if base_return != 0 else 0
            sensitivities[param_name] = sensitivity
        
        return sensitivities
    
    async def _run_stress_tests(self,
                              system_func: Callable,
                              mc_params: MonteCarloParameters,
                              market_context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Run stress tests under extreme scenarios"""
        
        stress_results = {}
        
        for scenario_name in mc_params.stress_scenarios:
            self.logger.info(f"   Running stress test: {scenario_name}")
            
            # Generate stress scenario
            stress_scenario = self._generate_stress_scenario(scenario_name, market_context)
            
            # Run multiple simulations for each stress scenario
            scenario_returns = []
            scenario_sharpes = []
            scenario_drawdowns = []
            
            for _ in range(50):  # Reduced iterations for stress tests
                performance = await system_func(stress_scenario)
                scenario_returns.append(performance.get('total_return_pct', 0.0))
                scenario_sharpes.append(performance.get('sharpe_ratio', 0.0))
                scenario_drawdowns.append(performance.get('max_drawdown_pct', 0.0))
            
            stress_results[scenario_name] = {
                'mean_return': np.mean(scenario_returns),
                'std_return': np.std(scenario_returns),
                'worst_return': np.min(scenario_returns),
                'mean_sharpe': np.mean(scenario_sharpes),
                'worst_drawdown': np.max(scenario_drawdowns)
            }
        
        return stress_results
    
    def _generate_stress_scenario(self, scenario_name: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific stress test scenario"""
        
        n_days = market_context.get('simulation_days', 120)
        symbols = market_context.get('symbols', ['SPY', 'QQQ', 'AAPL'])
        
        if scenario_name == 'black_swan':
            # Extreme negative event
            price_paths = {}
            for symbol in symbols:
                paths = self.scenario_generator.generate_price_paths(
                    100, n_days, 1, volatility=0.5, drift=-0.3, regime='black_swan'
                )
                price_paths[symbol] = paths[0]
            sentiment_data = np.full(n_days, -0.8)  # Very negative sentiment
            
        elif scenario_name == 'high_volatility':
            # High volatility but neutral return
            price_paths = {}
            for symbol in symbols:
                paths = self.scenario_generator.generate_price_paths(
                    100, n_days, 1, volatility=0.6, drift=0.0, regime='high_volatility'
                )
                price_paths[symbol] = paths[0]
            sentiment_data = np.random.normal(0, 0.8, n_days)  # High sentiment volatility
            
        elif scenario_name == 'bear_market':
            # Extended bear market
            price_paths = {}
            for symbol in symbols:
                paths = self.scenario_generator.generate_price_paths(
                    100, n_days, 1, volatility=0.3, drift=-0.2, regime='bear_market'
                )
                price_paths[symbol] = paths[0]
            sentiment_data = np.random.normal(-0.3, 0.4, n_days)  # Persistently negative
            
        else:  # correlation_breakdown
            # Correlations break down
            price_paths = {}
            for symbol in symbols:
                # Independent random walks
                drift = np.random.uniform(-0.1, 0.1)
                vol = np.random.uniform(0.2, 0.4)
                paths = self.scenario_generator.generate_price_paths(
                    100, n_days, 1, volatility=vol, drift=drift
                )
                price_paths[symbol] = paths[0]
            sentiment_data = np.random.normal(0, 0.5, n_days)
        
        return {
            'regime': scenario_name,
            'price_paths': price_paths,
            'sentiment_data': sentiment_data,
            'correlation_matrix': np.eye(len(symbols)),  # No correlation for stress
            'randomized_params': {},
            'market_context': market_context
        }
    
    async def _analyze_regime_performance(self,
                                        system_func: Callable,
                                        market_context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different market regimes"""
        
        regimes = ['normal', 'bull_market', 'bear_market', 'high_volatility']
        regime_results = {}
        
        for regime in regimes:
            regime_returns = []
            regime_sharpes = []
            
            # Run multiple simulations for each regime
            for _ in range(100):
                scenario = {
                    'regime': regime,
                    'price_paths': {},
                    'sentiment_data': self.scenario_generator.generate_news_sentiment_scenarios(
                        120, regime=regime
                    ),
                    'market_context': market_context
                }
                
                # Generate price paths for this regime
                for symbol in market_context.get('symbols', ['SPY']):
                    paths = self.scenario_generator.generate_price_paths(
                        100, 120, 1, regime=regime
                    )
                    scenario['price_paths'][symbol] = paths[0]
                
                performance = await system_func(scenario)
                regime_returns.append(performance.get('total_return_pct', 0.0))
                regime_sharpes.append(performance.get('sharpe_ratio', 0.0))
            
            regime_results[regime] = {
                'mean_return': np.mean(regime_returns),
                'std_return': np.std(regime_returns),
                'mean_sharpe': np.mean(regime_sharpes),
                'success_rate': sum(1 for r in regime_returns if r > 0) / len(regime_returns) * 100
            }
        
        return regime_results
    
    def _calculate_model_validity(self, results: MonteCarloResults) -> float:
        """Calculate overall model validity score"""
        
        validity_score = 0.0
        max_score = 100.0
        
        # Expected return check (30 points)
        if results.expected_return > 5.0:  # 5% annual return threshold
            validity_score += 30.0
        elif results.expected_return > 0:
            validity_score += 15.0
        
        # Sharpe ratio check (25 points)
        if results.expected_sharpe > 1.0:
            validity_score += 25.0
        elif results.expected_sharpe > 0.5:
            validity_score += 15.0
        elif results.expected_sharpe > 0:
            validity_score += 5.0
        
        # Stability check (25 points) - low volatility of returns
        return_cv = results.return_std / abs(results.expected_return) if results.expected_return != 0 else float('inf')
        if return_cv < 0.5:
            validity_score += 25.0
        elif return_cv < 1.0:
            validity_score += 15.0
        elif return_cv < 2.0:
            validity_score += 5.0
        
        # Tail risk check (20 points) - reasonable tail behavior
        if results.tail_risk_metrics['jarque_bera_pvalue'] > 0.05:  # Normal distribution
            validity_score += 20.0
        elif abs(results.tail_risk_metrics['skewness']) < 1.0:  # Reasonable skewness
            validity_score += 10.0
        
        return validity_score
    
    def _test_statistical_significance(self, results: MonteCarloResults) -> bool:
        """Test if results are statistically significant"""
        
        # Test if expected return is significantly different from zero
        returns_array = np.array(results.return_distribution)
        t_stat, p_value = stats.ttest_1samp(returns_array, 0)
        
        return p_value < 0.05 and results.expected_return > 0
    
    def _generate_recommendations(self, results: MonteCarloResults) -> Tuple[List[str], List[str]]:
        """Generate actionable recommendations and warnings"""
        
        recommendations = []
        warnings = []
        
        # Performance-based recommendations
        if results.expected_return > 15.0 and results.expected_sharpe > 1.5:
            recommendations.append("Excellent performance metrics - consider increasing position sizes")
        elif results.expected_return > 5.0 and results.expected_sharpe > 1.0:
            recommendations.append("Good performance - suitable for live trading with standard risk management")
        elif results.expected_return > 0:
            recommendations.append("Marginal performance - consider additional optimization before deployment")
        else:
            warnings.append("Negative expected returns - system requires significant improvements")
        
        # Risk-based warnings
        if results.worst_case_drawdown > 30.0:
            warnings.append(f"Extreme drawdown risk: {results.worst_case_drawdown:.1f}% in worst case")
            recommendations.append("Implement stronger position sizing controls and stop-losses")
        
        if results.expected_drawdown > 15.0:
            warnings.append(f"High average drawdown: {results.expected_drawdown:.1f}%")
        
        # Stability warnings
        return_cv = results.return_std / abs(results.expected_return) if results.expected_return != 0 else float('inf')
        if return_cv > 1.0:
            warnings.append("High return volatility - performance may be unstable")
            recommendations.append("Consider parameter optimization to reduce volatility")
        
        # Parameter sensitivity warnings
        if results.parameter_sensitivity:
            max_sensitivity = max(results.parameter_sensitivity.values())
            if max_sensitivity > 0.5:
                most_sensitive_param = max(results.parameter_sensitivity, key=results.parameter_sensitivity.get)
                warnings.append(f"High parameter sensitivity to {most_sensitive_param}")
                recommendations.append(f"Carefully tune {most_sensitive_param} parameter")
        
        # Stress test warnings
        if results.stress_test_results:
            for scenario, metrics in results.stress_test_results.items():
                if metrics['worst_return'] < -20.0:
                    warnings.append(f"Poor stress test performance in {scenario}: {metrics['worst_return']:.1f}%")
        
        # Model validity
        if results.model_validity_score < 50.0:
            warnings.append("Low model validity score - consider system redesign")
        elif results.model_validity_score > 80.0:
            recommendations.append("High model validity - system appears robust")
        
        return recommendations, warnings


class MonteCarloReportGenerator:
    """Generates comprehensive Monte Carlo analysis reports"""
    
    def __init__(self):
        self.logger = logging.getLogger("monte_carlo_report_generator")
    
    def generate_comprehensive_report(self, results: MonteCarloResults) -> str:
        """Generate comprehensive Monte Carlo analysis report"""
        
        report = []
        report.append("🎲 MONTE CARLO ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Simulation ID: {results.simulation_id}")
        report.append(f"Number of Simulations: {results.n_simulations:,}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # PERFORMANCE DISTRIBUTION ANALYSIS
        report.append("📊 PERFORMANCE DISTRIBUTION ANALYSIS")
        report.append("-" * 50)
        report.append(f"Expected Return: {results.expected_return:.1f}% ± {results.return_std:.1f}%")
        report.append(f"Expected Sharpe Ratio: {results.expected_sharpe:.2f} ± {results.sharpe_std:.2f}")
        report.append(f"Expected Drawdown: {results.expected_drawdown:.1f}% ± {results.drawdown_std:.1f}%")
        report.append(f"Worst Case Drawdown: {results.worst_case_drawdown:.1f}%")
        report.append("")
        
        # CONFIDENCE INTERVALS
        report.append("📈 CONFIDENCE INTERVALS")
        report.append("-" * 50)
        for conf_level, (lower, upper) in results.return_confidence_intervals.items():
            report.append(f"Return {conf_level:.0%} CI: [{lower:.1f}%, {upper:.1f}%]")
        report.append("")
        for conf_level, (lower, upper) in results.sharpe_confidence_intervals.items():
            report.append(f"Sharpe {conf_level:.0%} CI: [{lower:.2f}, {upper:.2f}]")
        report.append("")
        
        # RISK METRICS
        report.append("⚠️ RISK ANALYSIS")
        report.append("-" * 50)
        report.append("Value at Risk (VaR):")
        for conf_level, var_val in results.var_estimates.items():
            report.append(f"  VaR {conf_level:.0%}: {var_val:.1f}%")
        report.append("")
        report.append("Expected Shortfall (ES):")
        for conf_level, es_val in results.expected_shortfall.items():
            report.append(f"  ES {conf_level:.0%}: {es_val:.1f}%")
        report.append("")
        
        # TAIL RISK ANALYSIS
        report.append("📉 TAIL RISK ANALYSIS")
        report.append("-" * 50)
        report.append(f"Skewness: {results.tail_risk_metrics['skewness']:.3f}")
        report.append(f"Excess Kurtosis: {results.tail_risk_metrics['kurtosis']:.3f}")
        report.append(f"Jarque-Bera Test: {results.tail_risk_metrics['jarque_bera_stat']:.2f} (p={results.tail_risk_metrics['jarque_bera_pvalue']:.4f})")
        
        normality_assessment = "Normal" if results.tail_risk_metrics['jarque_bera_pvalue'] > 0.05 else "Non-normal"
        report.append(f"Distribution Assessment: {normality_assessment}")
        report.append("")
        
        # PARAMETER SENSITIVITY
        if results.parameter_sensitivity:
            report.append("🎛️ PARAMETER SENSITIVITY ANALYSIS")
            report.append("-" * 50)
            sorted_sensitivity = sorted(results.parameter_sensitivity.items(), key=lambda x: x[1], reverse=True)
            for param, sensitivity in sorted_sensitivity:
                sensitivity_level = "HIGH" if sensitivity > 0.3 else "MEDIUM" if sensitivity > 0.1 else "LOW"
                report.append(f"{param:<25}: {sensitivity:.3f} ({sensitivity_level})")
            report.append("")
        
        # STRESS TEST RESULTS
        if results.stress_test_results:
            report.append("💥 STRESS TEST RESULTS")
            report.append("-" * 50)
            for scenario, metrics in results.stress_test_results.items():
                report.append(f"{scenario.upper()}:")
                report.append(f"  Mean Return: {metrics['mean_return']:+.1f}%")
                report.append(f"  Worst Return: {metrics['worst_return']:+.1f}%")
                report.append(f"  Mean Sharpe: {metrics['mean_sharpe']:.2f}")
                report.append(f"  Worst Drawdown: {metrics['worst_drawdown']:.1f}%")
                report.append("")
        
        # REGIME PERFORMANCE
        if results.regime_performance:
            report.append("🌍 REGIME PERFORMANCE ANALYSIS")
            report.append("-" * 50)
            report.append(f"{'Regime':<15} {'Avg Return':<12} {'Success Rate':<12} {'Avg Sharpe':<12}")
            report.append("-" * 55)
            for regime, metrics in results.regime_performance.items():
                report.append(f"{regime:<15} {metrics['mean_return']:>9.1f}% {metrics['success_rate']:>9.1f}% {metrics['mean_sharpe']:>10.2f}")
            report.append("")
        
        # MODEL VALIDATION
        report.append("✅ MODEL VALIDATION")
        report.append("-" * 50)
        report.append(f"Model Validity Score: {results.model_validity_score:.1f}/100")
        validity_assessment = "EXCELLENT" if results.model_validity_score > 80 else \
                            "GOOD" if results.model_validity_score > 60 else \
                            "FAIR" if results.model_validity_score > 40 else "POOR"
        report.append(f"Assessment: {validity_assessment}")
        report.append(f"Statistical Significance: {'✅ YES' if results.statistical_significance else '❌ NO'}")
        report.append("")
        
        # RECOMMENDATIONS
        if results.recommendations:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 50)
            for i, rec in enumerate(results.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # WARNINGS
        if results.warnings:
            report.append("⚠️ WARNINGS")
            report.append("-" * 50)
            for i, warning in enumerate(results.warnings, 1):
                report.append(f"{i}. {warning}")
            report.append("")
        
        # FINAL ASSESSMENT
        report.append("🎯 FINAL ASSESSMENT")
        report.append("-" * 50)
        
        if results.statistical_significance and results.expected_return > 10.0 and results.model_validity_score > 70:
            report.append("✅ SYSTEM APPROVED FOR DEPLOYMENT")
            report.append("The enhanced trading system demonstrates robust performance across")
            report.append("multiple scenarios and passes all statistical validation tests.")
        elif results.expected_return > 5.0 and results.model_validity_score > 50:
            report.append("⚠️ CONDITIONAL APPROVAL")
            report.append("System shows promise but requires additional optimization")
            report.append("and monitoring before full deployment.")
        else:
            report.append("❌ SYSTEM NOT READY FOR DEPLOYMENT")
            report.append("Significant improvements needed before considering live trading.")
        
        return "\n".join(report)
    
    def save_detailed_results(self, results: MonteCarloResults, output_dir: str = "/home/eddy/Hyper/analysis/statistical/") -> str:
        """Save detailed results to JSON file"""
        
        # Convert results to dictionary
        results_dict = {
            'simulation_id': results.simulation_id,
            'n_simulations': results.n_simulations,
            'timestamp': datetime.now().isoformat(),
            
            'performance_statistics': {
                'expected_return': results.expected_return,
                'return_std': results.return_std,
                'expected_sharpe': results.expected_sharpe,
                'sharpe_std': results.sharpe_std,
                'expected_drawdown': results.expected_drawdown,
                'worst_case_drawdown': results.worst_case_drawdown
            },
            
            'confidence_intervals': {
                'return_intervals': {str(k): v for k, v in results.return_confidence_intervals.items()},
                'sharpe_intervals': {str(k): v for k, v in results.sharpe_confidence_intervals.items()}
            },
            
            'risk_metrics': {
                'var_estimates': {str(k): v for k, v in results.var_estimates.items()},
                'expected_shortfall': {str(k): v for k, v in results.expected_shortfall.items()},
                'tail_risk_metrics': results.tail_risk_metrics
            },
            
            'sensitivity_analysis': results.parameter_sensitivity,
            'stress_test_results': results.stress_test_results,
            'regime_performance': results.regime_performance,
            
            'validation': {
                'model_validity_score': results.model_validity_score,
                'statistical_significance': results.statistical_significance,
                'recommendations': results.recommendations,
                'warnings': results.warnings
            }
        }
        
        # Save to file
        filename = f"{output_dir}monte_carlo_results_{results.simulation_id}.json"
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to: {filename}")
        return filename


# Example usage and demonstration
async def run_monte_carlo_validation():
    """Run comprehensive Monte Carlo validation"""
    
    print("🎲 Monte Carlo Robustness Testing Framework")
    print("=" * 80)
    
    # Mock system performance function for demonstration
    async def mock_system_performance(scenario: Dict[str, Any]) -> Dict[str, float]:
        """Mock system performance function"""
        
        # Simulate realistic performance based on scenario
        regime = scenario.get('regime', 'normal')
        params = scenario.get('randomized_params', {})
        
        # Base performance
        if regime == 'bull_market':
            base_return = np.random.normal(15.0, 8.0)
            base_sharpe = np.random.normal(1.2, 0.3)
        elif regime == 'bear_market':
            base_return = np.random.normal(-5.0, 12.0)
            base_sharpe = np.random.normal(0.2, 0.4)
        elif regime == 'high_volatility':
            base_return = np.random.normal(2.0, 20.0)
            base_sharpe = np.random.normal(0.4, 0.6)
        else:  # normal
            base_return = np.random.normal(8.0, 6.0)
            base_sharpe = np.random.normal(0.8, 0.2)
        
        # Parameter adjustments
        confidence_adj = params.get('confidence_threshold', 0.15)
        if confidence_adj > 0.2:
            base_return *= 0.8  # Higher threshold reduces returns
        
        return {
            'total_return_pct': base_return,
            'sharpe_ratio': max(0, base_sharpe),
            'max_drawdown_pct': abs(np.random.normal(8.0, 4.0)),
            'win_rate_pct': np.random.uniform(45, 70)
        }
    
    # Configure Monte Carlo parameters
    mc_params = MonteCarloParameters(
        n_simulations=500,  # Reduced for demo
        confidence_levels=[0.90, 0.95, 0.99],
        parameter_ranges={
            'confidence_threshold': (0.05, 0.30),
            'sentiment_weight': (0.2, 0.6),
            'regime_weight': (0.1, 0.4),
            'options_weight': (0.1, 0.4)
        },
        stress_scenarios=['black_swan', 'high_volatility', 'bear_market', 'correlation_breakdown']
    )
    
    # Market context
    market_context = {
        'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT'],
        'simulation_days': 120
    }
    
    # Initialize Monte Carlo engine
    config = {
        'statistical_validator': {
            'confidence_level': 0.95,
            'significance_level': 0.05
        }
    }
    
    mc_engine = MonteCarloEngine(config)
    report_generator = MonteCarloReportGenerator()
    
    try:
        # Run Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        results = await mc_engine.run_monte_carlo_simulation(
            mock_system_performance, mc_params, market_context
        )
        
        # Generate and display report
        report = report_generator.generate_comprehensive_report(results)
        print("\n" + report)
        
        # Save detailed results
        filename = report_generator.save_detailed_results(results)
        print(f"\nDetailed results saved to: {filename}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during Monte Carlo simulation: {e}")
        logging.exception("Monte Carlo simulation failed")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run Monte Carlo validation
    asyncio.run(run_monte_carlo_validation())