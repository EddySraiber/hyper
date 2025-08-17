import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.base import ComponentBase


class AIAnalyzer(ComponentBase):
    """
    AI-powered news analysis component that sends news to external AI services
    and retrieves advanced trading parameters and insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ai_analyzer", config)
        # Handle both ConfigManager and dict, and both full config and direct ai_analyzer config
        if hasattr(config, 'get'):  # ConfigManager or dict
            try:
                # Try to get ai_analyzer section (works for both ConfigManager and dict with nested config)
                self.ai_config = config.get("ai_analyzer", {})
                if not self.ai_config:  # Empty dict means we might have direct config
                    self.ai_config = config if isinstance(config, dict) else {}
            except:
                # Fallback: treat as direct config
                self.ai_config = config if isinstance(config, dict) else {}
        else:
            self.ai_config = config  # Direct ai_analyzer config passed
        self.primary_provider = self.ai_config.get("provider", "groq")
        self.providers_config = self.ai_config.get("providers", {})
        self.fallback_chain = self.ai_config.get("fallback_chain", ["groq", "openai", "anthropic", "traditional"])
        self.max_retries = self.ai_config.get("max_retries", 3)
        self.session = None
        
        # Load provider configurations
        self._load_provider_configs()
        
    async def start(self) -> None:
        """Start the AI analyzer component"""
        self.logger.info("Starting AI Analyzer")
        # Use a default timeout since we now have per-provider timeouts
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        self.is_running = True
        
    async def stop(self) -> None:
        """Stop the AI analyzer component"""
        self.logger.info("Stopping AI Analyzer")
        if self.session:
            await self.session.close()
        self.is_running = False
        
    async def analyze_news_batch(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of news items using AI
        
        Returns enhanced news items with AI-generated parameters:
        - market_sentiment: Advanced sentiment analysis (-1.0 to 1.0)
        - confidence_score: AI confidence in the analysis (0.0 to 1.0)
        - volatility_prediction: Predicted price volatility (0.0 to 1.0)
        - time_horizon: Expected impact duration (minutes, hours, days)
        - market_impact: Sector/market-wide impact assessment
        - trading_signals: Specific buy/sell/hold recommendations
        - risk_factors: Identified risk elements
        """
        if not self.is_running or not news_items:
            return news_items
            
        enhanced_items = []
        
        for item in news_items:
            try:
                ai_analysis = await self._analyze_single_item(item)
                item.update(ai_analysis)
                enhanced_items.append(item)
                
                # Rate limiting - small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"AI analysis failed for item: {e}")
                # Fallback to basic analysis
                item.update(self._fallback_analysis(item))
                enhanced_items.append(item)
                
        self.logger.info(f"AI analyzed {len(enhanced_items)} news items")
        return enhanced_items
        
    async def process(self, data: Any) -> Any:
        """Process method required by ComponentBase - delegates to analyze_news_batch"""
        if isinstance(data, list):
            return await self.analyze_news_batch(data)
        else:
            return await self.analyze_news_batch([data])
        
    async def _analyze_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Send single news item to AI for analysis"""
        
        # Prepare news content for AI
        title = item.get('title', '')
        content = item.get('content', '')
        symbol = item.get('symbol', 'UNKNOWN')
        
        # AI prompt for financial analysis
        prompt = self._build_analysis_prompt(title, content, symbol)
        
        # Try primary provider first, then fallback chain
        providers_to_try = [self.primary_provider] + [p for p in self.fallback_chain if p != self.primary_provider and p != "traditional"]
        
        for provider in providers_to_try:
            try:
                if provider == "openai":
                    return await self._call_openai(prompt, provider)
                elif provider == "anthropic":
                    return await self._call_anthropic(prompt, provider)
                elif provider == "groq":
                    return await self._call_groq(prompt, provider)
                elif provider == "local":
                    return await self._call_local_ai(prompt, provider)
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                continue
                
        # If all providers failed, use fallback
        self.logger.error("All AI providers failed, using fallback analysis")
        return self._fallback_analysis({})
            
    def _build_analysis_prompt(self, title: str, content: str, symbol: str) -> str:
        """Build structured prompt for AI analysis"""
        return f"""
Analyze this financial news for algorithmic trading decisions:

TITLE: {title}
CONTENT: {content[:1000]}  # Limit content length
SYMBOL: {symbol}

Provide analysis in JSON format with these fields:
{{
    "market_sentiment": <float between -1.0 and 1.0>,
    "confidence_score": <float between 0.0 and 1.0>,
    "volatility_prediction": <float between 0.0 and 1.0>,
    "time_horizon": "<minutes|hours|days>",
    "market_impact": "<sector|individual|broad>",
    "trading_signals": {{
        "action": "<buy|sell|hold>",
        "strength": <float between 0.0 and 1.0>,
        "entry_price_target": <float or null>,
        "stop_loss_suggestion": <float or null>,
        "take_profit_suggestion": <float or null>
    }},
    "risk_factors": [<list of identified risks>],
    "key_insights": [<list of key trading insights>]
}}

Focus on actionable trading intelligence, not general market commentary.
"""

    def _load_provider_configs(self):
        """Load and validate provider configurations"""
        import os
        self.provider_configs = {}
        
        for provider_name, provider_config in self.providers_config.items():
            if not provider_config.get("enabled", True):
                continue
                
            config = {
                "model": provider_config.get("model", ""),
                "base_url": provider_config.get("base_url", ""),
                "timeout": provider_config.get("timeout", 30),
                "max_tokens": provider_config.get("max_tokens", 800),
                "temperature": provider_config.get("temperature", 0.3),
                "api_key": ""
            }
            
            # Get API key from environment
            api_key_env = provider_config.get("api_key_env", "")
            if api_key_env:
                config["api_key"] = os.getenv(api_key_env, "")
                
            self.provider_configs[provider_name] = config
            
        self.logger.info(f"Loaded {len(self.provider_configs)} AI provider configurations")
        
    async def _call_openai(self, prompt: str, provider: str = "openai") -> Dict[str, Any]:
        """Call OpenAI API for analysis"""
        config = self.provider_configs.get(provider, {})
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError(f"{provider} API key not configured")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.get("model", "gpt-3.5-turbo"),
            "messages": [
                {"role": "system", "content": "You are a professional financial analyst providing structured trading insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.get("temperature", 0.3),
            "max_tokens": config.get("max_tokens", 800)
        }
        
        async with self.session.post(
            config.get("base_url", "https://api.openai.com/v1/chat/completions"),
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                try:
                    ai_analysis = json.loads(content)
                    ai_analysis["ai_provider"] = provider
                    ai_analysis["ai_model"] = config.get("model", "unknown")
                    ai_analysis["ai_timestamp"] = datetime.utcnow().isoformat()
                    return ai_analysis
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse {provider} response: {content}")
                    raise Exception(f"Invalid JSON response from {provider}")
            else:
                error_text = await response.text()
                raise Exception(f"{provider} API error {response.status}: {error_text}")
                
    async def _call_anthropic(self, prompt: str, provider: str = "anthropic") -> Dict[str, Any]:
        """Call Anthropic Claude API for analysis"""
        config = self.provider_configs.get(provider, {})
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError(f"{provider} API key not configured")
            
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": config.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": config.get("max_tokens", 800),
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with self.session.post(
            config.get("base_url", "https://api.anthropic.com/v1/messages"),
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["content"][0]["text"]
                
                try:
                    ai_analysis = json.loads(content)
                    ai_analysis["ai_provider"] = provider
                    ai_analysis["ai_model"] = config.get("model", "unknown")
                    ai_analysis["ai_timestamp"] = datetime.utcnow().isoformat()
                    return ai_analysis
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse {provider} response: {content}")
                    raise Exception(f"Invalid JSON response from {provider}")
            else:
                error_text = await response.text()
                raise Exception(f"{provider} API error {response.status}: {error_text}")
                
    async def _call_groq(self, prompt: str, provider: str = "groq") -> Dict[str, Any]:
        """Call Groq API for analysis"""
        config = self.provider_configs.get(provider, {})
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError(f"{provider} API key not configured")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.get("model", "llama3-8b-8192"),
            "messages": [
                {"role": "system", "content": "You are a professional financial analyst providing structured trading insights. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.get("temperature", 0.3),
            "max_tokens": config.get("max_tokens", 800),
            "response_format": {"type": "json_object"}  # Force JSON response
        }
        
        async with self.session.post(
            config.get("base_url", "https://api.groq.com/openai/v1/chat/completions"),
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                try:
                    ai_analysis = json.loads(content)
                    ai_analysis["ai_provider"] = provider
                    ai_analysis["ai_model"] = config.get("model", "unknown")
                    ai_analysis["ai_timestamp"] = datetime.utcnow().isoformat()
                    return ai_analysis
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse {provider} response: {content}")
                    raise Exception(f"Invalid JSON response from {provider}")
            else:
                error_text = await response.text()
                raise Exception(f"{provider} API error {response.status}: {error_text}")
                
    async def _call_local_ai(self, prompt: str, provider: str = "local") -> Dict[str, Any]:
        """Call local AI service (e.g., Ollama, local LLM)"""
        config = self.provider_configs.get(provider, {})
        
        # This would connect to a local AI service like Ollama
        # For now, return enhanced fallback analysis
        self.logger.info(f"Using local AI analysis (mock) - {config.get('model', 'unknown')}")
        
        # Mock response for local AI
        return {
            "market_sentiment": 0.1,  # Slightly positive mock
            "confidence_score": 0.6,
            "volatility_prediction": 0.4,
            "time_horizon": "hours",
            "market_impact": "individual",
            "trading_signals": {
                "action": "hold",
                "strength": 0.5,
                "entry_price_target": None,
                "stop_loss_suggestion": None,
                "take_profit_suggestion": None
            },
            "risk_factors": ["Local AI - limited training data"],
            "key_insights": ["Local analysis - consider upgrading to cloud AI"],
            "ai_provider": provider,
            "ai_model": config.get("model", "local-model"),
            "ai_timestamp": datetime.utcnow().isoformat()
        }
        
    def _fallback_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback analysis when AI is unavailable"""
        return {
            "market_sentiment": 0.0,
            "confidence_score": 0.3,  # Lower confidence for fallback
            "volatility_prediction": 0.5,
            "time_horizon": "hours",
            "market_impact": "individual",
            "trading_signals": {
                "action": "hold",
                "strength": 0.3,
                "entry_price_target": None,
                "stop_loss_suggestion": None,
                "take_profit_suggestion": None
            },
            "risk_factors": ["AI analysis unavailable"],
            "key_insights": ["Fallback analysis - limited intelligence"],
            "ai_provider": "fallback",
            "ai_timestamp": datetime.utcnow().isoformat()
        }