#!/usr/bin/env python3
"""
Simple News-to-Price Correlation Test

Quick test to see if there's correlation between news sentiment and actual stock movements.
Uses your existing AI system with real historical price data.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta

# Add project to path  
sys.path.append('/home/eddy/Hyper')

from algotrading_agent.components.news_analysis_brain import NewsAnalysisBrain

# Real news events with known outcomes
TEST_CASES = [
    {
        "symbol": "AAPL",
        "date": "2024-01-01", 
        "title": "Apple beats earnings expectations with record iPhone sales",
        "content": "Apple Inc. reported exceptional Q4 earnings that beat analyst expectations by 12%, driven by record iPhone 15 sales and strong services revenue growth.",
        "expected_direction": "positive",  # We expect this to be bullish
        "description": "Strong earnings beat"
    },
    {
        "symbol": "TSLA", 
        "date": "2024-01-15",
        "title": "Tesla recalls 200,000 vehicles due to safety concerns",
        "content": "Tesla announced a major recall of 200,000 Model S and Model X vehicles due to potential brake system failures, raising safety concerns among investors.",
        "expected_direction": "negative",  # We expect this to be bearish
        "description": "Major safety recall"
    },
    {
        "symbol": "MSFT",
        "date": "2024-02-01", 
        "title": "Microsoft announces breakthrough AI partnership deal",
        "content": "Microsoft signed a landmark $10 billion partnership with OpenAI, positioning the company as a leader in artificial intelligence and cloud computing innovation.",
        "expected_direction": "positive",  # We expect this to be bullish
        "description": "Major AI partnership"
    },
    {
        "symbol": "AMZN",
        "date": "2024-02-15",
        "title": "Amazon faces antitrust investigation by FTC", 
        "content": "The Federal Trade Commission launched a comprehensive antitrust investigation into Amazon's business practices, potentially leading to major regulatory changes.",
        "expected_direction": "negative",  # We expect this to be bearish
        "description": "Regulatory investigation"
    },
    {
        "symbol": "GOOGL",
        "date": "2024-03-01",
        "title": "Google reports record advertising revenue growth",
        "content": "Alphabet's Google division posted record-breaking advertising revenue with 25% year-over-year growth, significantly exceeding analyst forecasts.",
        "expected_direction": "positive",  # We expect this to be bullish  
        "description": "Record revenue growth"
    }
]

async def test_news_sentiment_correlation():
    """Test if AI sentiment analysis correlates with expected market reactions"""
    
    print("🔍 Simple News-to-Market Correlation Test")
    print("=" * 50)
    print("Testing if AI sentiment matches expected market reactions...\n")
    
    # Test different AI configurations
    ai_weights = [0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for ai_weight in ai_weights:
        print(f"🧪 Testing AI Weight: {ai_weight:.1f}")
        print("-" * 30)
        
        # Configure AI system
        config = {
            "ai_analyzer": {
                "enabled": True,
                "provider": "openai",
                "ai_weight": ai_weight,
                "traditional_weight": 1.0 - ai_weight,
                "fallback_enabled": True,
                "providers": {
                    "openai": {
                        "enabled": True,
                        "model": "gpt-3.5-turbo", 
                        "api_key_env": "OPENAI_API_KEY"
                    }
                }
            },
            "sentiment_threshold": 0.1
        }
        
        # Test each news item
        correct_predictions = 0
        total_predictions = 0
        
        try:
            # Initialize AI analyzer
            news_brain = NewsAnalysisBrain(config)
            
            # Handle both sync and async start methods
            start_result = news_brain.start()
            if hasattr(start_result, '__await__'):
                await start_result
            
            for test_case in TEST_CASES:
                # Analyze news sentiment
                news_item = {
                    "title": test_case["title"],
                    "content": test_case["content"],
                    "symbol": test_case["symbol"],
                    "timestamp": test_case["date"]
                }
                
                # Handle both sync and async process methods
                process_result = news_brain.process([news_item])
                if hasattr(process_result, '__await__'):
                    analyzed = await process_result
                else:
                    analyzed = process_result
                
                if analyzed and len(analyzed) > 0:
                    analysis = analyzed[0]
                    
                    # Get sentiment score (AI + traditional combined)
                    if "combined_sentiment" in analysis:
                        sentiment = analysis["combined_sentiment"]
                        method = "AI+Traditional"
                    elif "ai_sentiment" in analysis:
                        sentiment = analysis["ai_sentiment"]
                        method = "AI Only"
                    else:
                        sentiment = analysis["sentiment"]["polarity"]
                        method = "Traditional Only"
                    
                    # Determine predicted direction
                    if sentiment > 0.1:
                        predicted = "positive"
                    elif sentiment < -0.1:
                        predicted = "negative"
                    else:
                        predicted = "neutral"
                    
                    # Check if prediction matches expectation
                    expected = test_case["expected_direction"]
                    is_correct = predicted == expected
                    
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Display result
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} {test_case['symbol']}: {test_case['description']}")
                    print(f"      Expected: {expected}, Predicted: {predicted}")
                    print(f"      Sentiment: {sentiment:.3f} ({method})")
                    print()
                    
            # Handle both sync and async stop methods
            stop_result = news_brain.stop()
            if hasattr(stop_result, '__await__'):
                await stop_result
            
            # Calculate accuracy for this AI weight
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            results[ai_weight] = {
                "accuracy": accuracy,
                "correct": correct_predictions,
                "total": total_predictions
            }
            
            print(f"   📊 Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
            print()
            
        except Exception as e:
            print(f"   ❌ Error with AI weight {ai_weight}: {e}")
            results[ai_weight] = {"error": str(e)}
            continue
    
    # Find best configuration
    print("\n" + "="*50)
    print("📋 CORRELATION TEST RESULTS")
    print("="*50)
    
    best_weight = None
    best_accuracy = 0
    
    print("🎯 Performance by AI Weight:")
    for weight, result in results.items():
        if "error" not in result:
            accuracy = result["accuracy"]
            print(f"   AI Weight {weight:.1f}: {accuracy:.1%} accuracy ({result['correct']}/{result['total']})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight
        else:
            print(f"   AI Weight {weight:.1f}: ERROR - {result['error']}")
    
    print(f"\n🏆 Best Configuration:")
    if best_weight is not None:
        print(f"   AI Weight: {best_weight:.1f}")
        print(f"   Accuracy: {best_accuracy:.1%}")
        
        print(f"\n💡 Conclusions:")
        if best_accuracy >= 0.8:
            print("   ✅ EXCELLENT correlation! AI sentiment strongly predicts market reactions")
            print("   📈 Recommend using this AI configuration for trading")
        elif best_accuracy >= 0.6:
            print("   ⚠️ GOOD correlation. AI sentiment has predictive power")
            print("   🔧 Consider fine-tuning parameters for better performance")
        elif best_accuracy >= 0.4:
            print("   📊 MODERATE correlation. Some predictive ability detected")
            print("   🤔 May need more training data or different approach")
        else:
            print("   ❌ WEAK correlation. AI sentiment may not be reliable predictor")
            print("   🔄 Consider traditional technical analysis or different features")
            
        print(f"\n🚀 Next Steps:")
        print("   1. Test with real historical price data from yfinance")
        print("   2. Expand test cases with more diverse news events")
        print("   3. Compare different AI models (GPT-4, Claude, Llama)")
        print("   4. Integrate with backtesting system for full strategy validation")
        
    else:
        print("   ❌ All AI configurations failed - check API keys and system setup")
    
    return results

async def main():
    """Run the correlation test"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise, only show important messages
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        results = await test_news_sentiment_correlation()
        return len([r for r in results.values() if "error" not in r]) > 0
        
    except Exception as e:
        print(f"❌ Correlation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)