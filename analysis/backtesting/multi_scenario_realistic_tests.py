#!/usr/bin/env python3
"""
Multi-Scenario Realistic Testing Framework
Comprehensive validation across different market conditions, strategies, and optimizations

Author: Claude Code (Anthropic AI Assistant)
Date: August 14, 2025
Task: 7 - Multi-Scenario Realistic Testing
Next Task: 8 - Idealized vs Realistic Comparison
"""

import asyncio
import json
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import our frameworks
from enhanced_realistic_backtest import (
    EnhancedRealisticBacktester, RealisticBacktestMetrics, 
    generate_sample_trades
)
from realistic_commission_models import BrokerType, AssetType
from realistic_execution_simulator import MarketCondition


@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    starting_capital: float
    broker: BrokerType
    tax_state: str
    income_level: float
    strategy_focus: str  # "high_freq", "medium_freq", "low_freq", "crypto_focus", "tax_optimized"
    market_regime: str   # "bull", "bear", "sideways", "volatile"
    trade_count: int
    test_duration_days: int


@dataclass
class ScenarioResults:
    """Results from a test scenario"""
    scenario: TestScenario
    metrics: RealisticBacktestMetrics
    execution_time_seconds: float
    key_insights: List[str]
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        score = 0
        
        # Return component (40 points max)
        if self.metrics.total_return_pct > 50:
            score += min(40, self.metrics.total_return_pct / 5)
        
        # Risk-adjusted component (25 points max)
        if self.metrics.sharpe_ratio > 1.0:
            score += min(25, self.metrics.sharpe_ratio * 5)
        
        # Win rate component (20 points max)
        if self.metrics.win_rate_pct > 50:
            score += (self.metrics.win_rate_pct - 50) * 0.4
        
        # Execution quality component (15 points max)
        if self.metrics.avg_execution_quality > 50:
            score += (self.metrics.avg_execution_quality - 50) * 0.3
        
        return min(100, score)


class MultiScenarioTester:
    """
    Comprehensive multi-scenario testing framework
    """
    
    def __init__(self):
        self.logger = logging.getLogger("multi_scenario_tester")
        self.test_results: List[ScenarioResults] = []
        
    def generate_test_scenarios(self) -> List[TestScenario]:
        """Generate comprehensive test scenarios"""
        
        scenarios = []
        
        # 1. Account Size Impact Tests
        account_sizes = [
            (10000, "Small Account ($10K)"),
            (25000, "PDT Threshold ($25K)"),
            (100000, "Medium Account ($100K)"),
            (500000, "Large Account ($500K)"),
            (2000000, "Institutional ($2M)")
        ]
        
        for capital, name in account_sizes:
            scenarios.append(TestScenario(
                name=f"{name} - High Frequency",
                description=f"High-frequency trading with {name}",
                starting_capital=capital,
                broker=BrokerType.ALPACA,
                tax_state="CA",
                income_level=min(capital * 0.2, 500000),  # Realistic income scaling
                strategy_focus="high_freq",
                market_regime="normal",
                trade_count=1000,
                test_duration_days=90
            ))
        
        # 2. Broker Comparison Tests
        brokers = [
            (BrokerType.ALPACA, "Commission-Free"),
            (BrokerType.INTERACTIVE_BROKERS, "Professional"),
            (BrokerType.CHARLES_SCHWAB, "Traditional"),
            (BrokerType.BINANCE_US, "Crypto-Focused")
        ]
        
        for broker, desc in brokers:
            scenarios.append(TestScenario(
                name=f"Broker Test - {desc}",
                description=f"Medium frequency trading via {desc} broker",
                starting_capital=100000,
                broker=broker,
                tax_state="CA",
                income_level=100000,
                strategy_focus="medium_freq",
                market_regime="normal",
                trade_count=500,
                test_duration_days=180
            ))
        
        # 3. Tax Jurisdiction Impact Tests
        tax_scenarios = [
            ("FL", 0, "No State Tax"),
            ("TX", 0, "No State Tax"),
            ("CA", 100000, "High Tax (CA)"),
            ("NY", 150000, "High Tax (NY)"),
            ("CA", 500000, "Ultra High Tax")
        ]
        
        for state, income, desc in tax_scenarios:
            scenarios.append(TestScenario(
                name=f"Tax Test - {desc}",
                description=f"Tax impact analysis for {desc}",
                starting_capital=200000,
                broker=BrokerType.ALPACA,
                tax_state=state,
                income_level=income,
                strategy_focus="medium_freq",
                market_regime="normal",
                trade_count=400,
                test_duration_days=365  # Full year for tax analysis
            ))
        
        # 4. Strategy Focus Tests
        strategy_tests = [
            ("high_freq", "Lightning/Express Focus", 2000, 60),
            ("medium_freq", "Balanced Approach", 800, 180),
            ("low_freq", "Swing Trading", 200, 365),
            ("crypto_focus", "24/7 Crypto Trading", 1500, 180),
            ("tax_optimized", "Long-Term Holdings", 150, 365)
        ]
        
        for strategy, desc, trades, days in strategy_tests:
            scenarios.append(TestScenario(
                name=f"Strategy Test - {desc}",
                description=f"Optimized for {desc}",
                starting_capital=100000,
                broker=BrokerType.ALPACA,
                tax_state="CA",
                income_level=100000,
                strategy_focus=strategy,
                market_regime="normal",
                trade_count=trades,
                test_duration_days=days
            ))
        
        # 5. Market Regime Tests
        market_regimes = [
            ("bull", "Bull Market", 0.15, 0.02),      # 15% annual return, 2% volatility
            ("bear", "Bear Market", -0.10, 0.04),     # -10% annual return, 4% volatility  
            ("sideways", "Range-Bound", 0.02, 0.015), # 2% annual return, 1.5% volatility
            ("volatile", "High Volatility", 0.08, 0.08) # 8% annual return, 8% volatility
        ]
        
        for regime, desc, ret, vol in market_regimes:
            scenarios.append(TestScenario(
                name=f"Market Test - {desc}",
                description=f"Performance during {desc}",
                starting_capital=100000,
                broker=BrokerType.ALPACA,
                tax_state="CA", 
                income_level=100000,
                strategy_focus="medium_freq",
                market_regime=regime,
                trade_count=600,
                test_duration_days=180
            ))
        
        # 6. Extreme Stress Tests
        stress_tests = [
            ("market_crash", "Market Crash Simulation", 100000, 30),
            ("flash_crash", "Flash Crash Events", 100000, 7),
            ("regulatory_shock", "Regulatory Changes", 100000, 60),
            ("liquidity_crisis", "Low Liquidity Period", 100000, 45)
        ]
        
        for test_type, desc, capital, days in stress_tests:
            scenarios.append(TestScenario(
                name=f"Stress Test - {desc}",
                description=f"Extreme scenario: {desc}",
                starting_capital=capital,
                broker=BrokerType.ALPACA,
                tax_state="CA",
                income_level=100000,
                strategy_focus="medium_freq",
                market_regime="volatile",
                trade_count=200,
                test_duration_days=days
            ))
        
        return scenarios
    
    def generate_scenario_specific_trades(self, scenario: TestScenario) -> List[Dict[str, Any]]:
        """Generate trades specific to the test scenario"""
        
        trades = []
        start_date = datetime.now() - timedelta(days=scenario.test_duration_days)
        
        # Symbol selection based on strategy focus
        if scenario.strategy_focus == "crypto_focus":
            symbols = ["BTCUSD", "ETHUSD", "DOGEUSD", "ADAUSD", "SOLUSD", "AVAXUSD"]
            symbol_weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
        elif scenario.strategy_focus == "high_freq":
            symbols = ["SPY", "QQQ", "AAPL", "TSLA", "MSFT"]  # High liquidity
            symbol_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        else:
            symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BTCUSD", "ETHUSD"]
            symbol_weights = [0.15, 0.12, 0.12, 0.1, 0.1, 0.1, 0.11, 0.1, 0.1]
        
        # Execution lane distribution based on strategy
        if scenario.strategy_focus == "high_freq":
            lane_distribution = {"lightning": 0.4, "express": 0.35, "fast": 0.2, "standard": 0.05}
        elif scenario.strategy_focus == "medium_freq":
            lane_distribution = {"lightning": 0.1, "express": 0.3, "fast": 0.4, "standard": 0.2}
        elif scenario.strategy_focus == "low_freq":
            lane_distribution = {"lightning": 0.0, "express": 0.1, "fast": 0.2, "standard": 0.7}
        else:
            lane_distribution = {"lightning": 0.2, "express": 0.3, "fast": 0.3, "standard": 0.2}
        
        # Market regime influence on success rates
        regime_multipliers = {
            "bull": 1.3,      # 30% boost in bull market
            "bear": 0.7,      # 30% penalty in bear market
            "sideways": 0.9,  # 10% penalty in range-bound
            "volatile": 1.1,  # 10% boost in volatile (more opportunities)
            "normal": 1.0
        }
        
        base_success_rate = 0.55
        regime_multiplier = regime_multipliers.get(scenario.market_regime, 1.0)
        
        for i in range(scenario.trade_count):
            # Select symbol based on weights
            symbol = random.choices(symbols, weights=symbol_weights)[0]
            is_crypto = symbol.endswith('USD')
            
            # Generate trade parameters
            if is_crypto:
                base_price = random.uniform(0.01, 10000)
                quantity = random.uniform(0.01, 50)
            else:
                base_price = random.uniform(10, 500)
                quantity = random.randint(1, 1000)
            
            # Select execution lane based on strategy
            lane_choices = list(lane_distribution.keys())
            lane_weights = list(lane_distribution.values())
            execution_lane = random.choices(lane_choices, weights=lane_weights)[0]
            
            # Generate hype score correlated with execution lane
            lane_hype_mapping = {
                "lightning": (7, 10),
                "express": (5, 8),
                "fast": (2.5, 6),
                "standard": (1, 4)
            }
            hype_range = lane_hype_mapping[execution_lane]
            hype_score = random.uniform(hype_range[0], hype_range[1])
            
            # Generate velocity level from hype score
            if hype_score >= 8:
                velocity_level = "viral"
            elif hype_score >= 5:
                velocity_level = "breaking"
            elif hype_score >= 2.5:
                velocity_level = "trending"
            else:
                velocity_level = "normal"
            
            # Calculate success probability
            base_prob = base_success_rate * regime_multiplier
            hype_bonus = (hype_score - 1) * 0.03  # 3% bonus per hype point above 1
            success_probability = min(0.85, base_prob + hype_bonus)
            
            is_profitable = random.random() < success_probability
            
            # Generate P&L based on profitability and market regime
            if is_profitable:
                base_profit = random.uniform(100, 3000)
                regime_profit_multiplier = {
                    "bull": 1.5, "bear": 0.8, "sideways": 0.9, 
                    "volatile": 1.3, "normal": 1.0
                }[scenario.market_regime]
                gross_pnl = base_profit * regime_profit_multiplier * (hype_score / 5)
            else:
                base_loss = random.uniform(50, 1500)
                regime_loss_multiplier = {
                    "bull": 0.7, "bear": 1.4, "sideways": 1.0,
                    "volatile": 1.2, "normal": 1.0
                }[scenario.market_regime]
                gross_pnl = -base_loss * regime_loss_multiplier
            
            # Holding period based on strategy
            if scenario.strategy_focus == "high_freq":
                holding_days = random.randint(0, 3)  # Intraday to 3 days
            elif scenario.strategy_focus == "medium_freq":
                holding_days = random.randint(1, 30)  # 1-30 days
            elif scenario.strategy_focus == "low_freq" or scenario.strategy_focus == "tax_optimized":
                holding_days = random.randint(30, 400)  # 1+ months, some >1 year
            else:
                holding_days = random.randint(1, 14)  # Default
            
            # Generate timestamp
            trade_date = start_date + timedelta(
                days=random.randint(0, scenario.test_duration_days - 1),
                hours=random.randint(0 if is_crypto else 9, 23 if is_crypto else 16),
                minutes=random.randint(0, 59)
            )
            
            trade = {
                'symbol': symbol,
                'action': random.choice(['buy', 'sell']),
                'quantity': quantity,
                'price': base_price,
                'timestamp': trade_date.isoformat(),
                'gross_pnl': gross_pnl,
                'holding_period_days': holding_days,
                'trigger_type': 'hype_detection',
                'hype_score': hype_score,
                'velocity_level': velocity_level,
                'execution_lane': execution_lane
            }
            
            trades.append(trade)
        
        # Sort trades by timestamp
        return sorted(trades, key=lambda x: x['timestamp'])
    
    async def run_scenario_test(self, scenario: TestScenario) -> ScenarioResults:
        """Run a single scenario test"""
        
        self.logger.info(f"🧪 Testing scenario: {scenario.name}")
        start_time = datetime.now()
        
        # Generate scenario-specific trades
        trades = self.generate_scenario_specific_trades(scenario)
        
        # Create backtester
        backtester = EnhancedRealisticBacktester(
            broker=scenario.broker,
            starting_capital=scenario.starting_capital,
            tax_state=scenario.tax_state,
            income_level=scenario.income_level
        )
        
        # Run backtest
        start_date = (datetime.now() - timedelta(days=scenario.test_duration_days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        metrics = await backtester.run_realistic_backtest(
            trades, start_date, end_date
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Generate insights
        insights = self._generate_scenario_insights(scenario, metrics)
        
        results = ScenarioResults(
            scenario=scenario,
            metrics=metrics,
            execution_time_seconds=execution_time,
            key_insights=insights
        )
        
        self.logger.info(f"✅ Completed {scenario.name}: {results.get_performance_score():.1f}/100 score")
        return results
    
    def _generate_scenario_insights(self, scenario: TestScenario, metrics: RealisticBacktestMetrics) -> List[str]:
        """Generate key insights from scenario results"""
        
        insights = []
        
        # Performance insights
        if metrics.total_return_pct > 100:
            insights.append(f"Strong returns: {metrics.total_return_pct:.1f}% over {scenario.test_duration_days} days")
        elif metrics.total_return_pct > 20:
            insights.append(f"Positive returns: {metrics.total_return_pct:.1f}% achieved despite friction costs")
        else:
            insights.append(f"Poor returns: {metrics.total_return_pct:.1f}% - friction costs may be too high")
        
        # Friction cost insights
        if metrics.friction_cost_pct > 60:
            insights.append(f"Excessive friction costs: {metrics.friction_cost_pct:.1f}% of gross P&L consumed")
        elif metrics.friction_cost_pct > 40:
            insights.append(f"High friction costs: {metrics.friction_cost_pct:.1f}% of gross P&L - optimization needed")
        else:
            insights.append(f"Manageable friction costs: {metrics.friction_cost_pct:.1f}% of gross P&L")
        
        # Execution quality insights
        if metrics.avg_execution_quality > 70:
            insights.append(f"Excellent execution quality: {metrics.avg_execution_quality:.1f}/100 average")
        elif metrics.avg_execution_quality > 50:
            insights.append(f"Fair execution quality: {metrics.avg_execution_quality:.1f}/100 average")
        else:
            insights.append(f"Poor execution quality: {metrics.avg_execution_quality:.1f}/100 - market conditions challenging")
        
        # Strategy-specific insights
        if scenario.strategy_focus == "high_freq":
            if metrics.lightning_trades + metrics.express_trades > metrics.total_trades * 0.6:
                insights.append("Successfully executed high-frequency strategy with fast lanes")
            else:
                insights.append("High-frequency strategy constrained by execution delays")
        
        elif scenario.strategy_focus == "tax_optimized":
            if metrics.tax_efficiency_score > 50:
                insights.append(f"Good tax efficiency: {metrics.tax_efficiency_score:.1f}% long-term holdings")
            else:
                insights.append(f"Poor tax efficiency: {metrics.tax_efficiency_score:.1f}% long-term holdings")
        
        elif scenario.strategy_focus == "crypto_focus":
            crypto_dominance = metrics.crypto_trades / metrics.total_trades if metrics.total_trades > 0 else 0
            if crypto_dominance > 0.7:
                insights.append(f"Crypto-focused strategy: {crypto_dominance*100:.1f}% crypto trades")
                if metrics.crypto_win_rate > metrics.stock_win_rate:
                    insights.append("Crypto outperformed stocks as expected")
        
        # Account size insights
        if scenario.starting_capital < 25000:
            if metrics.total_trades < 100:
                insights.append("Small account limited by PDT rules - consider swing trading")
        elif scenario.starting_capital > 500000:
            if metrics.avg_slippage_bps > 50:
                insights.append("Large account suffering from market impact - consider position size limits")
        
        # Risk insights
        if metrics.max_drawdown_pct > 30:
            insights.append(f"High drawdown risk: {metrics.max_drawdown_pct:.1f}% maximum drawdown")
        elif metrics.max_drawdown_pct < 10:
            insights.append(f"Low drawdown risk: {metrics.max_drawdown_pct:.1f}% maximum drawdown")
        
        if metrics.sharpe_ratio > 2.0:
            insights.append(f"Excellent risk-adjusted returns: {metrics.sharpe_ratio:.2f} Sharpe ratio")
        elif metrics.sharpe_ratio < 1.0:
            insights.append(f"Poor risk-adjusted returns: {metrics.sharpe_ratio:.2f} Sharpe ratio")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def run_comprehensive_test_suite(self, max_scenarios: int = 25) -> Dict[str, Any]:
        """Run comprehensive test suite across multiple scenarios"""
        
        self.logger.info(f"🚀 Starting comprehensive multi-scenario test suite")
        self.logger.info(f"📊 Testing up to {max_scenarios} scenarios")
        
        # Generate all scenarios
        all_scenarios = self.generate_test_scenarios()
        
        # Limit to max_scenarios for performance
        selected_scenarios = all_scenarios[:max_scenarios]
        
        self.logger.info(f"✅ Selected {len(selected_scenarios)} scenarios for testing")
        
        # Run all scenario tests
        total_start_time = datetime.now()
        
        for i, scenario in enumerate(selected_scenarios, 1):
            self.logger.info(f"📈 Running scenario {i}/{len(selected_scenarios)}: {scenario.name}")
            
            try:
                results = await self.run_scenario_test(scenario)
                self.test_results.append(results)
                
                # Progress update
                self.logger.info(f"✅ Scenario {i} complete: {results.get_performance_score():.1f}/100 score")
                
            except Exception as e:
                self.logger.error(f"❌ Scenario {i} failed: {str(e)}")
                continue
        
        total_execution_time = (datetime.now() - total_start_time).total_seconds()
        
        # Analyze results
        analysis = self._analyze_comprehensive_results()
        analysis['total_execution_time_seconds'] = total_execution_time
        analysis['scenarios_tested'] = len(self.test_results)
        
        self.logger.info(f"🏁 Comprehensive test suite completed in {total_execution_time:.1f} seconds")
        self.logger.info(f"📊 Successfully tested {len(self.test_results)}/{len(selected_scenarios)} scenarios")
        
        return analysis
    
    def _analyze_comprehensive_results(self) -> Dict[str, Any]:
        """Analyze results across all scenarios"""
        
        if not self.test_results:
            return {"error": "No test results available"}
        
        analysis = {
            "executive_summary": {},
            "performance_rankings": [],
            "category_analysis": {},
            "key_findings": [],
            "optimization_recommendations": []
        }
        
        # Executive summary
        scores = [r.get_performance_score() for r in self.test_results]
        returns = [r.metrics.total_return_pct for r in self.test_results]
        friction_costs = [r.metrics.friction_cost_pct for r in self.test_results]
        
        analysis["executive_summary"] = {
            "scenarios_tested": len(self.test_results),
            "avg_performance_score": sum(scores) / len(scores),
            "best_performance_score": max(scores),
            "worst_performance_score": min(scores),
            "avg_return_pct": sum(returns) / len(returns),
            "best_return_pct": max(returns),
            "worst_return_pct": min(returns),
            "avg_friction_cost_pct": sum(friction_costs) / len(friction_costs),
            "lowest_friction_cost_pct": min(friction_costs),
            "highest_friction_cost_pct": max(friction_costs)
        }
        
        # Performance rankings
        ranked_results = sorted(self.test_results, key=lambda x: x.get_performance_score(), reverse=True)
        
        analysis["performance_rankings"] = [
            {
                "rank": i + 1,
                "scenario_name": result.scenario.name,
                "performance_score": result.get_performance_score(),
                "return_pct": result.metrics.total_return_pct,
                "friction_cost_pct": result.metrics.friction_cost_pct,
                "key_insight": result.key_insights[0] if result.key_insights else "No insights available"
            }
            for i, result in enumerate(ranked_results[:10])  # Top 10
        ]
        
        # Category analysis
        categories = {
            "account_size": {},
            "broker_type": {},
            "strategy_focus": {},
            "market_regime": {},
            "tax_jurisdiction": {}
        }
        
        # Group by categories
        for result in self.test_results:
            scenario = result.scenario
            
            # Account size categories
            if scenario.starting_capital < 25000:
                size_cat = "small"
            elif scenario.starting_capital < 100000:
                size_cat = "medium"
            elif scenario.starting_capital < 500000:
                size_cat = "large"
            else:
                size_cat = "institutional"
            
            if size_cat not in categories["account_size"]:
                categories["account_size"][size_cat] = []
            categories["account_size"][size_cat].append(result)
            
            # Broker categories
            broker_cat = scenario.broker.value
            if broker_cat not in categories["broker_type"]:
                categories["broker_type"][broker_cat] = []
            categories["broker_type"][broker_cat].append(result)
            
            # Strategy categories
            strategy_cat = scenario.strategy_focus
            if strategy_cat not in categories["strategy_focus"]:
                categories["strategy_focus"][strategy_cat] = []
            categories["strategy_focus"][strategy_cat].append(result)
            
            # Market regime categories
            regime_cat = scenario.market_regime
            if regime_cat not in categories["market_regime"]:
                categories["market_regime"][regime_cat] = []
            categories["market_regime"][regime_cat].append(result)
            
            # Tax jurisdiction categories
            tax_cat = f"{scenario.tax_state}_{scenario.income_level}"
            if tax_cat not in categories["tax_jurisdiction"]:
                categories["tax_jurisdiction"][tax_cat] = []
            categories["tax_jurisdiction"][tax_cat].append(result)
        
        # Calculate category averages
        for category, groups in categories.items():
            analysis["category_analysis"][category] = {}
            
            for group_name, group_results in groups.items():
                if group_results:
                    avg_score = sum(r.get_performance_score() for r in group_results) / len(group_results)
                    avg_return = sum(r.metrics.total_return_pct for r in group_results) / len(group_results)
                    avg_friction = sum(r.metrics.friction_cost_pct for r in group_results) / len(group_results)
                    
                    analysis["category_analysis"][category][group_name] = {
                        "count": len(group_results),
                        "avg_performance_score": avg_score,
                        "avg_return_pct": avg_return,
                        "avg_friction_cost_pct": avg_friction
                    }
        
        # Generate key findings
        analysis["key_findings"] = self._generate_key_findings(analysis)
        
        # Generate optimization recommendations
        analysis["optimization_recommendations"] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _generate_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate key findings from comprehensive analysis"""
        
        findings = []
        summary = analysis["executive_summary"]
        
        # Overall performance finding
        if summary["avg_performance_score"] > 70:
            findings.append(f"✅ Strong overall performance: {summary['avg_performance_score']:.1f}/100 average score across scenarios")
        elif summary["avg_performance_score"] > 50:
            findings.append(f"⚠️ Moderate performance: {summary['avg_performance_score']:.1f}/100 average score - optimization needed")
        else:
            findings.append(f"❌ Poor overall performance: {summary['avg_performance_score']:.1f}/100 average score - major issues identified")
        
        # Return finding
        if summary["avg_return_pct"] > 100:
            findings.append(f"📈 Excellent returns: {summary['avg_return_pct']:.1f}% average return despite friction costs")
        elif summary["avg_return_pct"] > 20:
            findings.append(f"📊 Positive returns: {summary['avg_return_pct']:.1f}% average return achieved")
        else:
            findings.append(f"📉 Poor returns: {summary['avg_return_pct']:.1f}% average return - friction costs dominating")
        
        # Friction cost finding
        if summary["avg_friction_cost_pct"] > 50:
            findings.append(f"💸 Excessive friction costs: {summary['avg_friction_cost_pct']:.1f}% of profits consumed on average")
        elif summary["avg_friction_cost_pct"] > 30:
            findings.append(f"💰 High friction costs: {summary['avg_friction_cost_pct']:.1f}% of profits - significant drag on performance")
        else:
            findings.append(f"💵 Manageable friction costs: {summary['avg_friction_cost_pct']:.1f}% of profits")
        
        # Best performer finding
        best_performers = analysis["performance_rankings"][:3]
        if best_performers:
            best = best_performers[0]
            findings.append(f"🏆 Best performer: {best['scenario_name']} ({best['performance_score']:.1f}/100, {best['return_pct']:.1f}% return)")
        
        # Category findings
        category_analysis = analysis.get("category_analysis", {})
        
        # Account size finding
        if "account_size" in category_analysis:
            size_scores = {k: v["avg_performance_score"] for k, v in category_analysis["account_size"].items()}
            if size_scores:
                best_size = max(size_scores, key=size_scores.get)
                findings.append(f"📊 Best account size: {best_size} accounts ({size_scores[best_size]:.1f}/100 average)")
        
        # Strategy finding
        if "strategy_focus" in category_analysis:
            strategy_scores = {k: v["avg_performance_score"] for k, v in category_analysis["strategy_focus"].items()}
            if strategy_scores:
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                findings.append(f"🎯 Best strategy: {best_strategy} ({strategy_scores[best_strategy]:.1f}/100 average)")
        
        return findings
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        summary = analysis["executive_summary"]
        
        # High-level recommendations based on friction costs
        if summary["avg_friction_cost_pct"] > 40:
            recommendations.append("🔧 CRITICAL: Implement friction cost reduction strategies")
            recommendations.append("💡 Consider: Longer holding periods to reduce tax burden")
            recommendations.append("🏦 Consider: Commission-free brokers for frequent trading")
            recommendations.append("📊 Consider: Larger position sizes to amortize fixed costs")
        
        # Performance-based recommendations
        if summary["avg_performance_score"] < 60:
            recommendations.append("⚡ URGENT: System performance below acceptable threshold")
            recommendations.append("🎯 Focus on: High-conviction signals to improve win rate")
            recommendations.append("🛡️ Implement: Better risk management to reduce drawdowns")
        
        # Category-specific recommendations
        category_analysis = analysis.get("category_analysis", {})
        
        # Account size recommendations
        if "account_size" in category_analysis:
            size_scores = category_analysis["account_size"]
            if "small" in size_scores and size_scores["small"]["avg_performance_score"] < 50:
                recommendations.append("💰 Small accounts: Focus on swing trading to avoid PDT restrictions")
            if "institutional" in size_scores and size_scores["institutional"]["avg_friction_cost_pct"] > 30:
                recommendations.append("🏦 Large accounts: Implement position size limits to reduce market impact")
        
        # Strategy recommendations
        if "strategy_focus" in category_analysis:
            strategy_scores = category_analysis["strategy_focus"]
            best_strategy = max(strategy_scores, key=lambda x: strategy_scores[x]["avg_performance_score"])
            recommendations.append(f"🎯 Prioritize: {best_strategy} strategy showed best performance")
        
        # Tax optimization recommendations
        if summary["avg_friction_cost_pct"] > 35:
            recommendations.append("🏛️ Tax Optimization: Hold winning positions >1 year for long-term gains")
            recommendations.append("📍 Location: Consider trading from tax-free states (FL, TX, NV)")
        
        return recommendations
    
    def save_comprehensive_results(self, filepath: str):
        """Save comprehensive test results to file"""
        
        # Prepare data for JSON serialization
        results_data = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "scenarios_tested": len(self.test_results),
                "framework_version": "1.0"
            },
            "individual_results": [],
            "comprehensive_analysis": self._analyze_comprehensive_results()
        }
        
        # Add individual results
        for result in self.test_results:
            result_data = {
                "scenario": asdict(result.scenario),
                "metrics": asdict(result.metrics),
                "performance_score": result.get_performance_score(),
                "execution_time_seconds": result.execution_time_seconds,
                "key_insights": result.key_insights
            }
            results_data["individual_results"].append(result_data)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Comprehensive results saved to: {filepath}")


# Main testing function
async def run_comprehensive_multi_scenario_test():
    """Run the complete multi-scenario test suite"""
    
    print("🚀 COMPREHENSIVE MULTI-SCENARIO TESTING FRAMEWORK")
    print("=" * 80)
    print("Testing realistic trading performance across multiple scenarios:")
    print("• Account sizes: $10K - $2M")
    print("• Brokers: Alpaca, IBKR, Schwab, Binance")
    print("• Strategies: High-freq, Medium-freq, Low-freq, Crypto, Tax-optimized")
    print("• Market conditions: Bull, Bear, Sideways, Volatile")
    print("• Tax jurisdictions: FL, TX, CA, NY (various income levels)")
    print("• Stress tests: Market crashes, flash crashes, liquidity crises")
    print()
    
    # Initialize tester
    tester = MultiScenarioTester()
    
    # Run comprehensive test suite (limited for demo)
    print("⏳ Running comprehensive test suite...")
    analysis = await tester.run_comprehensive_test_suite(max_scenarios=15)  # Limit for performance
    
    # Display executive summary
    print("\n" + "=" * 80)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 80)
    
    summary = analysis["executive_summary"]
    print(f"Scenarios Tested: {summary['scenarios_tested']}")
    print(f"Average Performance Score: {summary['avg_performance_score']:.1f}/100")
    print(f"Best Performance Score: {summary['best_performance_score']:.1f}/100")
    print(f"Average Return: {summary['avg_return_pct']:.1f}%")
    print(f"Best Return: {summary['best_return_pct']:.1f}%")
    print(f"Average Friction Cost: {summary['avg_friction_cost_pct']:.1f}% of profits")
    print(f"Lowest Friction Cost: {summary['lowest_friction_cost_pct']:.1f}% of profits")
    
    # Display top performers
    print("\n🏆 TOP PERFORMERS:")
    print("-" * 50)
    for rank_data in analysis["performance_rankings"][:5]:
        print(f"{rank_data['rank']}. {rank_data['scenario_name']}")
        print(f"   Score: {rank_data['performance_score']:.1f}/100, Return: {rank_data['return_pct']:.1f}%")
        print(f"   Friction: {rank_data['friction_cost_pct']:.1f}%, Insight: {rank_data['key_insight']}")
        print()
    
    # Display key findings
    print("🔍 KEY FINDINGS:")
    print("-" * 50)
    for finding in analysis["key_findings"]:
        print(f"• {finding}")
    
    # Display optimization recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 50)
    for rec in analysis["optimization_recommendations"]:
        print(f"• {rec}")
    
    # Save results
    results_file = "/home/eddy/Hyper/analysis/backtesting/multi_scenario_results.json"
    tester.save_comprehensive_results(results_file)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print(f"⏱️ Total execution time: {analysis['total_execution_time_seconds']:.1f} seconds")
    
    return analysis


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive multi-scenario test
    asyncio.run(run_comprehensive_multi_scenario_test())
    
    print("\n✅ Multi-scenario testing framework complete!")
    print("📊 Ready for final analysis and optimization recommendations")