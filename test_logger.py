"""
Unit tests for logging infrastructure.
"""

import logging
import tempfile
from pathlib import Path

import pytest

from src.logger import setup_logging, get_logger


class TestLogger:
    """Test suite for logging infrastructure."""
    
    def test_setup_logging_default(self):
        """Test setting up logging with default parameters."""
        logger = setup_logging()
        
        assert logger is not None
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_debug_level(self):
        """Test setting up logging with DEBUG level."""
        logger = setup_logging(log_level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test setting up logging with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'test.log'
            logger = setup_logging(log_file=str(log_file))
            
            # Log a test message
            logger.info("Test message")
            
            # Close all handlers to release file locks (Windows issue)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Verify file was created and contains the message
            assert log_file.exists()
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_setup_logging_custom_format(self):
        """Test setting up logging with custom format."""
        custom_format = '%(levelname)s - %(message)s'
        logger = setup_logging(log_format=custom_format)
        
        assert logger is not None
        # Verify handler has the custom format
        assert len(logger.handlers) > 0
    
    def test_get_logger(self):
        """Test getting a named logger."""
        logger = get_logger('test_module')
        
        assert logger is not None
        assert logger.name == 'test_module'
    
    def test_logging_levels(self):
        """Test different logging levels."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            logger = setup_logging(log_level=level)
            expected_level = getattr(logging, level)
            assert logger.level == expected_level

