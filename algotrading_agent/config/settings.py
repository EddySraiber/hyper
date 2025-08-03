import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Try different possible locations
        possible_paths = [
            "/app/config/default.yml",
            "./config/default.yml", 
            "../config/default.yml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError("Could not find default configuration file")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Override with environment variables if present
            self._apply_env_overrides(config)
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
            
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides"""
        
        # Alpaca API credentials
        if os.getenv('ALPACA_API_KEY'):
            config.setdefault('alpaca', {})['api_key'] = os.getenv('ALPACA_API_KEY')
            
        if os.getenv('ALPACA_SECRET_KEY'):
            config.setdefault('alpaca', {})['secret_key'] = os.getenv('ALPACA_SECRET_KEY')
            
        # Paper trading override
        if os.getenv('ALPACA_PAPER_TRADING'):
            paper_trading = os.getenv('ALPACA_PAPER_TRADING').lower() == 'true'
            config.setdefault('alpaca', {})['paper_trading'] = paper_trading
            
        # Log level override
        if os.getenv('LOG_LEVEL'):
            config.setdefault('app', {})['log_level'] = os.getenv('LOG_LEVEL')
            config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
            
        # Data directory override
        if os.getenv('DATA_DIR'):
            config.setdefault('app', {})['data_dir'] = os.getenv('DATA_DIR')
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'news_scraper.update_interval')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.config.get(component_name, {})
        
    def get_alpaca_config(self) -> Dict[str, Any]:
        """Get Alpaca-specific configuration with credentials"""
        alpaca_config = self.config.get('alpaca', {})
        
        # Ensure we have API credentials
        if not alpaca_config.get('api_key') or not alpaca_config.get('secret_key'):
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
            
        return alpaca_config
        
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value at runtime"""
        keys = key.split('.')
        config_section = self.config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
            
        config_section[keys[-1]] = value
        
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")


# Global configuration instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    """Reload configuration from file"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager