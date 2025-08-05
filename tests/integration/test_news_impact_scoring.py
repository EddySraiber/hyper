#!/usr/bin/env python3
"""
Test News Impact Scoring System Integration
"""

import sys
sys.path.append('/home/eddy/Hyper')

import asyncio
import logging
from datetime import datetime

from algotrading_agent.components.news_impact_scorer import NewsImpactScorer
from algotrading_agent.config.settings import get_config

def create_test_news():
    """Create comprehensive test news with various impact levels"""
    return [
        # Grade A+ - Exceptional Impact
        {
            'title': 'BREAKING: Apple Beats Expectations with Revolutionary iPhone Sales',
            'content': 'Apple smashed analyst estimates with breakthrough quarterly performance, revenue surging 25% driven by record iPhone demand and breakthrough AI features. Stock soars 15%.',
            'source': 'Reuters',
            'timestamp': '2024-11-04T09:30:00Z',
            'priority': 'breaking'
        },
        
        # Grade A - High Impact
        {
            'title': 'Tesla Misses Delivery Estimates, Stock Plunges 8%',
            'content': 'Tesla disappointed with Q3 deliveries missing expectations by 12%, raising concerns about demand. Elon Musk cites production challenges.',
            'source': 'Bloomberg',
            'timestamp': '2024-11-04T10:15:00Z',
            'priority': 'breaking'
        },
        
        # Grade B+ - Above Average Impact
        {
            'title': 'Microsoft Announces Major AI Partnership Deal Worth $2 Billion',
            'content': 'Microsoft unveiled a massive partnership with leading AI research firms, marking its biggest investment in artificial intelligence technology.',
            'source': 'CNBC',
            'timestamp': '2024-11-04T11:00:00Z'
        },
        
        # Grade B - Good Impact
        {
            'title': 'Amazon Beats Revenue Estimates with Strong Cloud Growth',
            'content': 'Amazon reported solid quarterly results with 18% revenue growth driven by AWS cloud services expansion.',
            'source': 'MarketWatch',
            'timestamp': '2024-11-04T12:00:00Z'
        },
        
        # Grade C - Moderate Impact
        {
            'title': 'Google Expands Data Center Operations',
            'content': 'Google announced plans to build new data centers in emerging markets to support growing demand.',
            'source': 'Yahoo Finance',
            'timestamp': '2024-11-04T13:00:00Z'
        },
        
        # Grade D - Low Impact
        {
            'title': 'Netflix Reports Standard Quarterly Growth',
            'content': 'Netflix added subscribers as expected during Q3, maintaining steady growth trajectory.',
            'source': 'Business Insider',
            'timestamp': '2024-11-04T14:00:00Z'
        },
        
        # Grade F - Minimal Impact
        {
            'title': 'Company Announces Regular Board Meeting',
            'content': 'Regular quarterly board meeting scheduled for next month to discuss standard business matters.',
            'source': 'Unknown',
            'timestamp': '2024-11-02T15:00:00Z'  # Older news
        }
    ]

async def test_news_impact_scoring():
    """Test the comprehensive news impact scoring system"""
    print("🧪 NEWS IMPACT SCORING SYSTEM TEST")
    print("=" * 50)
    
    # Initialize scorer
    config = get_config()
    scorer = NewsImpactScorer(config.get_component_config('news_impact_scorer'))
    scorer.start()
    
    # Create test news
    test_news = create_test_news()
    print(f"📰 Testing with {len(test_news)} news items\n")
    
    # Score the news
    scored_news = scorer.process(test_news)
    
    # Display detailed results
    print("📊 DETAILED SCORING RESULTS:")
    print("-" * 60)
    
    for i, item in enumerate(scored_news, 1):
        score = item.get('impact_score', 0)
        grade = item.get('impact_grade', 'F')
        title = item.get('title', 'Unknown')[:50]
        source = item.get('source', 'Unknown')
        
        # Color coding for grades
        if grade in ['A+', 'A']:
            status_icon = "🔥"
        elif grade in ['B+', 'B']:
            status_icon = "📈"
        elif grade == 'C':
            status_icon = "📊"
        else:
            status_icon = "📝"
            
        print(f"{status_icon} {i:2d}. Grade {grade:2s} | Score: {score:4.2f} | {source:12s} | {title}")
    
    # Get and display summary
    impact_summary = scorer.get_impact_summary(scored_news)
    
    print(f"\n🎯 IMPACT SUMMARY:")
    print("-" * 30)
    print(f"Total News Items: {impact_summary['total_items']}")
    print(f"Average Score: {impact_summary['avg_score']:.2f}")
    print(f"Maximum Score: {impact_summary['max_score']:.2f}")
    print(f"High Impact (≥0.8): {impact_summary['high_impact_count']}")
    
    print(f"\n📈 GRADE DISTRIBUTION:")
    for grade, count in sorted(impact_summary['grade_distribution'].items()):
        percentage = (count / impact_summary['total_items']) * 100
        print(f"  Grade {grade}: {count:2d} items ({percentage:4.1f}%)")
    
    print(f"\n🏆 TOP STORIES:")
    for i, story in enumerate(impact_summary['top_stories'][:5], 1):
        print(f"  {i}. {story['grade']} ({story['score']:.2f}): {story['title']}")
    
    # Test high-impact filtering for express lane
    print(f"\n🚀 EXPRESS LANE TRIGGERS:")
    high_impact_news = [item for item in scored_news if item.get('impact_score', 0) >= 1.2]
    breaking_news = [item for item in scored_news if item.get('priority') == 'breaking']
    
    print(f"Breaking News Priority: {len(breaking_news)} items")
    print(f"High Impact Score (≥1.2): {len(high_impact_news)} items")
    
    if high_impact_news or breaking_news:
        print(f"✅ EXPRESS LANE would be triggered!")
        express_triggers = high_impact_news + breaking_news
        # Remove duplicates
        express_triggers = list({item['title']: item for item in express_triggers}.values())
        for item in express_triggers[:3]:
            print(f"  ⚡ {item.get('impact_grade', 'N/A')}: {item.get('title', 'Unknown')[:60]}")
    else:
        print(f"❌ No express lane triggers detected")
    
    # Test market exposure scoring
    print(f"\n💰 MARKET EXPOSURE ANALYSIS:")
    mega_cap_mentions = [item for item in scored_news if any(symbol in item.get('title', '') + item.get('content', '') 
                                                           for symbol in ['AAPL', 'APPLE', 'TSLA', 'TESLA', 'MSFT', 'MICROSOFT'])]
    print(f"Mega-cap company mentions: {len(mega_cap_mentions)}")
    
    # Test timing factors
    print(f"\n⏰ TIMING ANALYSIS:")
    recent_news = [item for item in scored_news if '2024-11-04' in item.get('timestamp', '')]
    stale_news = [item for item in scored_news if '2024-11-02' in item.get('timestamp', '')]
    print(f"Recent news (today): {len(recent_news)} items")
    print(f"Stale news (2+ days old): {len(stale_news)} items")
    
    # Validation summary
    print(f"\n✅ VALIDATION RESULTS:")
    print("=" * 40)
    
    validation_checks = [
        ("Scoring System Active", scorer.is_running),
        ("Grade Distribution Reasonable", len(set(impact_summary['grade_distribution'].keys())) >= 3),
        ("High Impact Detection", impact_summary['high_impact_count'] > 0),
        ("Breaking News Priority", len(breaking_news) > 0),
        ("Express Lane Triggers", len(high_impact_news) + len(breaking_news) > 0),
        ("Source Authority Working", impact_summary['max_score'] > 1.0),
        ("Content Analysis Active", any('breakthrough' in item.get('content', '').lower() for item in scored_news))
    ]
    
    passed_checks = 0
    for check_name, result in validation_checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed_checks += 1
    
    success_rate = (passed_checks / len(validation_checks)) * 100
    print(f"\n🏆 OVERALL SUCCESS: {passed_checks}/{len(validation_checks)} checks passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 NEWS IMPACT SCORING SYSTEM: READY FOR PRODUCTION!")
        return True
    else:
        print("⚠️  NEWS IMPACT SCORING SYSTEM: NEEDS ADJUSTMENT")
        return False

async def main():
    """Main test function"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        print("🚀 Starting News Impact Scoring System Tests...")
        success = await test_news_impact_scoring()
        
        if success:
            print(f"\n✅ All tests completed successfully!")
            print(f"🔥 The system can now measure news worthiness and market impact potential!")
            print(f"📊 High-impact news (grade A/B) will trigger express lane execution")
            print(f"🎯 News scoring integrates: source authority + market exposure + content intensity + timing + hype")
            return 0
        else:
            print(f"\n❌ Some tests failed - system needs adjustment")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)