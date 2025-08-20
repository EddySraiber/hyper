#!/usr/bin/env python3
"""
News Sources Validation Script

Tests all configured news sources and provides detailed reliability report.
Run this to identify broken sources and optimize configuration.
"""

import asyncio
import aiohttp
import time
import yaml
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class SourceValidator:
    def __init__(self):
        self.session = None
        self.results = {}
        
    async def start(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'AlgoTradingAgent/2.0 Source Validator'
            }
        )
        
    async def stop(self):
        """Clean up HTTP session"""
        if self.session:
            await self.session.close()
            
    async def validate_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single news source"""
        name = source.get('name', 'Unknown')
        url = source.get('url', '')
        source_type = source.get('type', 'rss')
        
        result = {
            'name': name,
            'url': url,
            'type': source_type,
            'status': 'unknown',
            'response_time': None,
            'status_code': None,
            'content_length': 0,
            'error': None,
            'recommendations': []
        }
        
        if not url:
            result['status'] = 'error'
            result['error'] = 'No URL provided'
            return result
            
        # Test HTTP connectivity
        start_time = time.time()
        try:
            headers = source.get('headers', {})
            
            async with self.session.get(url, headers=headers) as response:
                result['status_code'] = response.status
                result['response_time'] = time.time() - start_time
                
                if response.status == 200:
                    content = await response.text()
                    result['content_length'] = len(content)
                    result['status'] = 'success'
                    
                    # Validate content based on type
                    if source_type == 'rss':
                        result.update(await self._validate_rss_content(content))
                    elif source_type == 'api':
                        result.update(await self._validate_api_content(content, response.content_type))
                        
                elif response.status in [403, 401]:
                    result['status'] = 'auth_required'
                    result['error'] = f'Authentication required (HTTP {response.status})'
                    result['recommendations'].append('Check if API key is required')
                elif response.status == 404:
                    result['status'] = 'not_found'
                    result['error'] = f'URL not found (HTTP {response.status})'
                    result['recommendations'].append('Update URL - endpoint may have changed')
                elif response.status >= 500:
                    result['status'] = 'server_error'
                    result['error'] = f'Server error (HTTP {response.status})'
                    result['recommendations'].append('Temporary server issue - may recover')
                else:
                    result['status'] = 'http_error'
                    result['error'] = f'HTTP error {response.status}'
                    
        except aiohttp.ClientConnectorError as e:
            result['status'] = 'connection_error'
            result['error'] = f'Connection failed: {str(e)}'
            result['response_time'] = time.time() - start_time
            
            # Specific recommendations for DNS failures
            if 'Name or service not known' in str(e):
                result['recommendations'].append('DNS resolution failed - check domain name')
            elif 'ssl' in str(e).lower():
                result['recommendations'].append('SSL/TLS error - try HTTP version if available')
            else:
                result['recommendations'].append('Network connectivity issue')
                
        except asyncio.TimeoutError:
            result['status'] = 'timeout'
            result['error'] = 'Request timed out'
            result['response_time'] = time.time() - start_time
            result['recommendations'].append('Increase timeout or find faster alternative')
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = f'Unexpected error: {str(e)}'
            result['response_time'] = time.time() - start_time
            
        return result
        
    async def _validate_rss_content(self, content: str) -> Dict[str, Any]:
        """Validate RSS feed content"""
        try:
            import feedparser
            feed = feedparser.parse(content)
            
            validation = {
                'feed_title': getattr(feed.feed, 'title', 'Unknown'),
                'entry_count': len(feed.entries),
                'feed_valid': not feed.bozo,
                'parsing_errors': []
            }
            
            if feed.bozo:
                validation['parsing_errors'].append(f'RSS parsing error: {feed.bozo_exception}')
                
            if len(feed.entries) == 0:
                validation['parsing_errors'].append('No entries found in feed')
            elif len(feed.entries) < 5:
                validation['parsing_errors'].append(f'Only {len(feed.entries)} entries - feed may be stale')
                
            return validation
            
        except Exception as e:
            return {
                'feed_valid': False,
                'parsing_errors': [f'Content validation error: {str(e)}']
            }
            
    async def _validate_api_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Validate API response content"""
        validation = {
            'content_type': content_type,
            'api_valid': False,
            'parsing_errors': []
        }
        
        try:
            if 'json' in content_type.lower():
                import json
                data = json.loads(content)
                validation['api_valid'] = True
                validation['data_keys'] = list(data.keys()) if isinstance(data, dict) else ['array_data']
                
                # Basic data validation
                if isinstance(data, dict) and not data:
                    validation['parsing_errors'].append('Empty JSON object returned')
                elif isinstance(data, list) and len(data) == 0:
                    validation['parsing_errors'].append('Empty JSON array returned')
                    
            elif any(xml_type in content_type.lower() for xml_type in ['xml', 'rss', 'atom']):
                # XML/RSS content in API endpoint
                import feedparser
                feed = feedparser.parse(content)
                validation['api_valid'] = not feed.bozo
                validation['entry_count'] = len(feed.entries)
                
                if feed.bozo:
                    validation['parsing_errors'].append(f'XML parsing error: {feed.bozo_exception}')
            else:
                validation['parsing_errors'].append(f'Unexpected content type: {content_type}')
                
        except json.JSONDecodeError as e:
            validation['parsing_errors'].append(f'JSON parsing error: {str(e)}')
        except Exception as e:
            validation['parsing_errors'].append(f'Content validation error: {str(e)}')
            
        return validation

    def load_config(self) -> Dict[str, Any]:
        """Load news sources configuration"""
        config_path = project_root / 'config' / 'default.yml'
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('enhanced_news_scraper', {})
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
            
    async def validate_all_sources(self) -> Dict[str, Any]:
        """Validate all configured news sources"""
        config = self.load_config()
        all_sources = []
        
        # Collect all source types
        for source_type in ['sources', 'api_sources', 'social_sources', 'realtime_sources']:
            sources = config.get(source_type, [])
            for source in sources:
                source['category'] = source_type
                all_sources.append(source)
        
        print(f"Validating {len(all_sources)} news sources...")
        
        # Validate sources concurrently (in batches)
        batch_size = 10
        results = []
        
        for i in range(0, len(all_sources), batch_size):
            batch = all_sources[i:i+batch_size]
            batch_results = await asyncio.gather(
                *[self.validate_source(source) for source in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
            print(f"Completed batch {i//batch_size + 1}/{(len(all_sources)-1)//batch_size + 1}")
            
        return self._analyze_results(results)
        
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze validation results and generate report"""
        total = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        
        status_counts = {}
        error_types = {}
        slow_sources = []
        broken_sources = []
        recommendations = {}
        
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status != 'success':
                broken_sources.append({
                    'name': result['name'],
                    'url': result['url'],
                    'status': status,
                    'error': result.get('error'),
                    'recommendations': result.get('recommendations', [])
                })
                
            # Track error types
            if result.get('error'):
                error_type = status
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
            # Track slow sources
            response_time = result.get('response_time')
            if response_time and response_time > 10:  # > 10 seconds
                slow_sources.append({
                    'name': result['name'],
                    'response_time': round(response_time, 2)
                })
                
        # Generate recommendations
        if status_counts.get('connection_error', 0) > 0:
            recommendations['dns_issues'] = f"{status_counts['connection_error']} sources have DNS/connectivity issues"
            
        if status_counts.get('auth_required', 0) > 0:
            recommendations['auth_needed'] = f"{status_counts['auth_required']} sources require API keys"
            
        if status_counts.get('not_found', 0) > 0:
            recommendations['url_updates'] = f"{status_counts['not_found']} sources need URL updates"
            
        reliability_rate = (successful / total) * 100 if total > 0 else 0
        
        return {
            'summary': {
                'total_sources': total,
                'successful': successful,
                'reliability_rate': round(reliability_rate, 1),
                'failed': total - successful
            },
            'status_breakdown': status_counts,
            'error_analysis': error_types,
            'slow_sources': sorted(slow_sources, key=lambda x: x['response_time'], reverse=True),
            'broken_sources': broken_sources,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
    def print_report(self, analysis: Dict[str, Any]):
        """Print detailed validation report"""
        print("\n" + "="*60)
        print("NEWS SOURCES VALIDATION REPORT")
        print("="*60)
        
        summary = analysis['summary']
        print(f"\n📊 SUMMARY:")
        print(f"  Total Sources: {summary['total_sources']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Reliability Rate: {summary['reliability_rate']}%")
        
        if summary['reliability_rate'] >= 90:
            print("  Status: ✅ EXCELLENT")
        elif summary['reliability_rate'] >= 70:
            print("  Status: ⚠️  GOOD")
        else:
            print("  Status: ❌ NEEDS IMPROVEMENT")
            
        print(f"\n📈 STATUS BREAKDOWN:")
        for status, count in analysis['status_breakdown'].items():
            percentage = (count / summary['total_sources']) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")
            
        if analysis['slow_sources']:
            print(f"\n🐌 SLOW SOURCES (>10s):")
            for source in analysis['slow_sources'][:5]:  # Top 5
                print(f"  {source['name']}: {source['response_time']}s")
                
        if analysis['broken_sources']:
            print(f"\n❌ BROKEN SOURCES:")
            for source in analysis['broken_sources'][:10]:  # Top 10
                print(f"  {source['name']}: {source['status']}")
                if source['error']:
                    print(f"    Error: {source['error']}")
                if source['recommendations']:
                    print(f"    Fix: {source['recommendations'][0]}")
                    
        if analysis['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for category, recommendation in analysis['recommendations'].items():
                print(f"  • {recommendation}")
                
        print(f"\n⏰ Generated: {analysis['timestamp']}")
        print("="*60)

async def main():
    """Main validation function"""
    validator = SourceValidator()
    
    try:
        await validator.start()
        results = await validator.validate_all_sources()
        validator.print_report(results)
        
        # Save detailed results to file
        report_path = project_root / 'data' / 'news_sources_validation.json'
        report_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nDetailed results saved to: {report_path}")
        
    finally:
        await validator.stop()

if __name__ == "__main__":
    asyncio.run(main())