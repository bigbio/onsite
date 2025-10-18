"""
Configuration module for pyLuciPHOr2.

This module contains the LucXorConfig class, which manages configuration settings
for the pyLuciPHOr2 algorithm.
"""

import logging
from typing import Dict, Any, Optional
from .constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class LucXorConfig:
    """
    Configuration class for pyLuciPHOr2.
    
    This class manages all configuration settings for the pyLuciPHOr2 algorithm,
    including fragment method, mass tolerance, and other parameters.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize a new LucXorConfig instance.
        
        Args:
            config_dict: Optional dictionary containing configuration settings
        """
        # Initialize with default configuration
        self.config = DEFAULT_CONFIG.copy()
        
        # Update with provided configuration
        if config_dict:
            self.update(config_dict)
            
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new settings.
        
        Args:
            config_dict: Dictionary containing new configuration settings
        """
        for key, value in config_dict.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
                
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary containing all configuration settings
        """
        return self.config.copy()
        
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.config[key]
        
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.config 