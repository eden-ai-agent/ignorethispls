#!/usr/bin/env python3
"""
Revolutionary Integrated Logging System
Patent Pending - Michael Derrick Jagneaux

Production-ready logging system specifically designed for the revolutionary O(1) 
fingerprint matching system. Provides comprehensive logging capabilities with 
performance monitoring, patent documentation, and enterprise-grade reliability.

Key Features:
- Advanced multi-level logging with O(1) performance tracking
- Real-time performance monitoring and patent validation logging
- Thread-safe logging for parallel fingerprint processing
- Integration with RevolutionaryConfigurationLoader
- Structured logging for scientific documentation
- Dynamic log rotation and archival management
- Enterprise security and compliance logging
- Statistical logging for mathematical proof generation
- GPU/CPU performance correlation logging
- Memory and resource utilization tracking
"""

import logging
import logging.handlers
import threading
import time
import json
import os
import sys
import inspect
import traceback
import queue
import atexit
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import warnings
import platform
import psutil
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import hashlib
import gzip
import shutil

# Import revolutionary configuration system
from .config_loader import RevolutionaryConfigurationLoader

# Suppress third-party logging noise
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


class LogLevel(Enum):
    """Revolutionary logging levels for comprehensive system monitoring."""
    TRACE = 5           # Ultra-detailed tracing for patent documentation
    DEBUG = 10          # Development and debugging information
    INFO = 20           # General information and system status
    WARNING = 30        # Warning conditions and non-critical issues
    ERROR = 40          # Error conditions requiring attention
    CRITICAL = 50       # Critical errors affecting system operation
    PERFORMANCE = 25    # Performance-specific logging for O(1) validation
    PATENT = 15         # Patent-specific logging for documentation
    SECURITY = 35       # Security and compliance logging


class LogCategory(Enum):
    """Log categories for organized system monitoring."""
    SYSTEM = "SYSTEM"                   # System-level operations
    FINGERPRINT = "FINGERPRINT"         # Fingerprint processing operations
    DATABASE = "DATABASE"               # Database operations
    PERFORMANCE = "PERFORMANCE"         # Performance and timing
    O1_VALIDATION = "O1_VALIDATION"     # O(1) performance validation
    PATENT = "PATENT"                   # Patent documentation
    SECURITY = "SECURITY"               # Security and access control
    API = "API"                         # API and web interface
    CACHE = "CACHE"                     # Caching operations
    IMAGE = "IMAGE"                     # Image processing
    CONFIGURATION = "CONFIGURATION"     # Configuration management
    ERROR = "ERROR"                     # Error tracking and analysis


@dataclass
class LogMetrics:
    """Performance metrics for logging operations."""
    logs_per_second: float = 0.0
    average_log_time_ms: float = 0.0
    queue_depth: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    rotation_count: int = 0
    compression_ratio: float = 1.0
    error_rate: float = 0.0


@dataclass
class PerformanceLogEntry:
    """Structured performance log entry for O(1) validation."""
    operation_name: str
    execution_time_ms: float
    memory_delta_mb: float
    cpu_usage_percent: float
    thread_id: int
    timestamp: float
    database_size: int
    is_o1_compliant: bool
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatentLogEntry:
    """Structured patent documentation log entry."""
    innovation_component: str
    technical_description: str
    performance_characteristics: Dict[str, Any]
    mathematical_proof: Optional[str]
    timestamp: float
    validation_data: Dict[str, Any] = field(default_factory=dict)


class RevolutionaryFormatter(logging.Formatter):
    """Advanced formatter for revolutionary logging system."""
    
    def __init__(self, include_performance: bool = True, include_context: bool = True):
        """Initialize the revolutionary formatter."""
        super().__init__()
        self.include_performance = include_performance
        self.include_context = include_context
        self.start_time = time.time()
        
        # Color codes for console output
        self.colors = {
            'TRACE': '\033[90m',      # Dark gray
            'DEBUG': '\033[94m',      # Blue
            'INFO': '\033[92m',       # Green
            'WARNING': '\033[93m',    # Yellow
            'ERROR': '\033[91m',      # Red
            'CRITICAL': '\033[95m',   # Magenta
            'PERFORMANCE': '\033[96m', # Cyan
            'PATENT': '\033[97m',     # White
            'SECURITY': '\033[93m',   # Yellow
            'RESET': '\033[0m'        # Reset
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with revolutionary enhancements."""
        try:
            # Base formatting
            timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Thread and process information
            thread_info = f"[T:{record.thread:04d}]" if hasattr(record, 'thread') else "[T:0000]"
            process_info = f"[P:{os.getpid():05d}]"
            
            # Performance information
            performance_info = ""
            if self.include_performance and hasattr(record, 'performance_data'):
                perf = record.performance_data
                performance_info = f"[⚡ {perf.get('execution_time_ms', 0):.2f}ms | 🧠 {perf.get('memory_mb', 0):.1f}MB]"
            
            # Context information
            context_info = ""
            if self.include_context:
                module = getattr(record, 'module', record.name)
                func = getattr(record, 'funcName', 'unknown')
                line = getattr(record, 'lineno', 0)
                context_info = f"[{module}:{func}:{line}]"
            
            # Category information
            category_info = ""
            if hasattr(record, 'category'):
                category_info = f"[{record.category.value}]"
            
            # Color formatting for console
            color = self.colors.get(record.levelname, '')
            reset = self.colors.get('RESET', '')
            
            # Build the formatted message
            parts = [
                f"{formatted_time} UTC",
                f"{color}[{record.levelname:^12}]{reset}",
                thread_info,
                process_info,
                category_info,
                context_info,
                performance_info,
                f"→ {record.getMessage()}"
            ]
            
            # Remove empty parts
            formatted_message = " ".join(part for part in parts if part.strip())
            
            # Add exception information if present
            if record.exc_info:
                formatted_message += f"\n{self.formatException(record.exc_info)}"
            
            return formatted_message
            
        except Exception as e:
            # Fallback formatting if something goes wrong
            return f"{record.created} [{record.levelname}] {record.getMessage()} [FORMATTER_ERROR: {e}]"


class PerformanceLogger:
    """Specialized performance logger for O(1) validation."""
    
    def __init__(self, logger_instance: 'RevolutionaryLogger'):
        """Initialize performance logger."""
        self.logger = logger_instance
        self.performance_cache = deque(maxlen=10000)
        self.cache_lock = threading.Lock()
    
    def log_o1_operation(self, 
                        operation_name: str,
                        execution_time_ms: float,
                        database_size: int,
                        is_constant_time: bool,
                        confidence_score: float = 1.0,
                        **metadata) -> None:
        """Log O(1) operation for patent validation."""
        
        entry = PerformanceLogEntry(
            operation_name=operation_name,
            execution_time_ms=execution_time_ms,
            memory_delta_mb=self._get_memory_delta(),
            cpu_usage_percent=psutil.cpu_percent(),
            thread_id=threading.get_ident(),
            timestamp=time.time(),
            database_size=database_size,
            is_o1_compliant=is_constant_time,
            confidence_score=confidence_score,
            metadata=metadata
        )
        
        with self.cache_lock:
            self.performance_cache.append(entry)
        
        # Log with structured data
        self.logger.performance(
            f"O(1) Operation: {operation_name} | "
            f"Time: {execution_time_ms:.3f}ms | "
            f"DB Size: {database_size:,} | "
            f"Constant Time: {'✅' if is_constant_time else '❌'} | "
            f"Confidence: {confidence_score:.3f}",
            extra={
                'performance_data': asdict(entry),
                'category': LogCategory.O1_VALIDATION
            }
        )
    
    def log_patent_validation(self,
                             innovation: str,
                             proof_data: Dict[str, Any],
                             validation_results: Dict[str, Any]) -> None:
        """Log patent validation data."""
        
        patent_entry = PatentLogEntry(
            innovation_component=innovation,
            technical_description=proof_data.get('description', ''),
            performance_characteristics=proof_data.get('performance', {}),
            mathematical_proof=proof_data.get('mathematical_proof'),
            timestamp=time.time(),
            validation_data=validation_results
        )
        
        self.logger.patent(
            f"Patent Validation: {innovation} | "
            f"Validation Score: {validation_results.get('score', 0):.3f}",
            extra={
                'patent_data': asdict(patent_entry),
                'category': LogCategory.PATENT
            }
        )
    
    def _get_memory_delta(self) -> float:
        """Get current memory delta in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self.cache_lock:
            if not self.performance_cache:
                return {}
            
            entries = list(self.performance_cache)
        
        # Calculate statistics
        execution_times = [entry.execution_time_ms for entry in entries]
        o1_compliance_rate = sum(1 for entry in entries if entry.is_o1_compliant) / len(entries)
        
        return {
            'total_operations': len(entries),
            'o1_compliance_rate': o1_compliance_rate,
            'average_execution_time_ms': sum(execution_times) / len(execution_times),
            'min_execution_time_ms': min(execution_times),
            'max_execution_time_ms': max(execution_times),
            'operations_by_type': defaultdict(int, {
                entry.operation_name: sum(1 for e in entries if e.operation_name == entry.operation_name)
                for entry in entries
            }),
            'time_range': {
                'start': min(entry.timestamp for entry in entries),
                'end': max(entry.timestamp for entry in entries)
            }
        }


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        """Initialize async handler."""
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        self.worker_thread.start()
        
        # Register cleanup
        atexit.register(self.close)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously."""
        try:
            if not self.shutdown_event.is_set():
                self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop the log if queue is full to maintain performance
            pass
    
    def _worker(self) -> None:
        """Worker thread for processing log records."""
        while not self.shutdown_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log worker errors to stderr to avoid infinite loops
                print(f"AsyncLogHandler worker error: {e}", file=sys.stderr)
    
    def close(self) -> None:
        """Shutdown the async handler."""
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            
            # Signal shutdown and wait
            try:
                self.log_queue.put_nowait(None)
                self.worker_thread.join(timeout=5.0)
            except:
                pass
            
            self.target_handler.close()
        
        super().close()


class RevolutionaryLogger:
    """
    Revolutionary logging system for the O(1) fingerprint matching system.
    
    Provides comprehensive logging capabilities with performance monitoring,
    patent documentation, and enterprise-grade reliability.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the revolutionary logging system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_loader = RevolutionaryConfigurationLoader(config_path)
        self.config = self._load_logging_config()
        
        # Initialize logging parameters
        self.log_level = getattr(logging, self.config.get('level', 'INFO').upper())
        self.log_dir = Path(self.config.get('directory', 'logs'))
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        self.backup_count = self.config.get('backup_count', 10)
        self.enable_compression = self.config.get('enable_compression', True)
        self.enable_async = self.config.get('enable_async', True)
        self.enable_performance = self.config.get('enable_performance_logging', True)
        self.enable_patent = self.config.get('enable_patent_logging', True)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = LogMetrics()
        self.metrics_lock = threading.Lock()
        
        # Performance logger
        self.performance = PerformanceLogger(self) if self.enable_performance else None
        
        # Initialize loggers
        self._setup_loggers()
        
        # Log system initialization
        self.info("🚀 Revolutionary Logging System Initialized")
        self.info(f"📁 Log Directory: {self.log_dir.absolute()}")
        self.info(f"📊 Performance Logging: {'✅' if self.enable_performance else '❌'}")
        self.info(f"📜 Patent Logging: {'✅' if self.enable_patent else '❌'}")
        self.info(f"⚡ Async Logging: {'✅' if self.enable_async else '❌'}")
    
    def _load_logging_config(self) -> Dict[str, Any]:
        """Load logging configuration from revolutionary config system."""
        try:
            app_config = self.config_loader.get_app_config()
            logging_config = app_config.get('logging', {})
            
            # Apply defaults
            defaults = {
                'level': 'INFO',
                'directory': 'logs',
                'max_file_size_mb': 100,
                'backup_count': 10,
                'enable_compression': True,
                'enable_async': True,
                'enable_performance_logging': True,
                'enable_patent_logging': True,
                'enable_console': True,
                'console_level': 'INFO',
                'file_level': 'DEBUG',
                'format': {
                    'include_performance': True,
                    'include_context': True,
                    'include_colors': True
                },
                'rotation': {
                    'mode': 'size',  # 'size' or 'time'
                    'interval': 'midnight',  # for time-based rotation
                    'when': 'midnight',
                    'backup_count': 10
                },
                'categories': {
                    'system': {'level': 'INFO', 'file': 'system.log'},
                    'performance': {'level': 'DEBUG', 'file': 'performance.log'},
                    'patent': {'level': 'DEBUG', 'file': 'patent.log'},
                    'security': {'level': 'WARNING', 'file': 'security.log'},
                    'error': {'level': 'ERROR', 'file': 'error.log'}
                }
            }
            
            # Recursively update defaults
            def update_recursive(default_dict, config_dict):
                for key, value in config_dict.items():
                    if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
                        update_recursive(default_dict[key], value)
                    else:
                        default_dict[key] = value
            
            update_recursive(defaults, logging_config)
            return defaults
            
        except Exception as e:
            # Fallback configuration
            print(f"Warning: Failed to load logging config: {e}", file=sys.stderr)
            return self._get_fallback_logging_config()
    
    def _get_fallback_logging_config(self) -> Dict[str, Any]:
        """Get fallback logging configuration."""
        return {
            'level': 'INFO',
            'directory': 'logs',
            'max_file_size_mb': 50,
            'backup_count': 5,
            'enable_compression': False,
            'enable_async': False,
            'enable_performance_logging': True,
            'enable_patent_logging': False,
            'enable_console': True,
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        }
    
    def _setup_loggers(self) -> None:
        """Setup all logging handlers and formatters."""
        # Create formatter
        self.formatter = RevolutionaryFormatter(
            include_performance=self.config.get('format', {}).get('include_performance', True),
            include_context=self.config.get('format', {}).get('include_context', True)
        )
        
        # Main logger
        self.logger = logging.getLogger('revolutionary_fingerprint')
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Console handler
        if self.config.get('enable_console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = getattr(logging, self.config.get('console_level', 'INFO').upper())
            console_handler.setLevel(console_level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        
        # File handlers
        self._setup_file_handlers()
        
        # Category-specific loggers
        self._setup_category_loggers()
    
    def _setup_file_handlers(self) -> None:
        """Setup file handlers with rotation."""
        # Main log file
        main_log_file = self.log_dir / 'revolutionary_fingerprint.log'
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=self.max_file_size_mb * 1024 * 1024,
            backupCount=self.backup_count
        )
        
        file_level = getattr(logging, self.config.get('file_level', 'DEBUG').upper())
        main_handler.setLevel(file_level)
        main_handler.setFormatter(self.formatter)
        
        # Wrap with async handler if enabled
        if self.enable_async:
            main_handler = AsyncLogHandler(main_handler)
        
        self.logger.addHandler(main_handler)
    
    def _setup_category_loggers(self) -> None:
        """Setup category-specific loggers."""
        categories = self.config.get('categories', {})
        
        for category_name, category_config in categories.items():
            if not isinstance(category_config, dict):
                continue
            
            log_file = self.log_dir / category_config.get('file', f'{category_name}.log')
            category_level = getattr(logging, category_config.get('level', 'INFO').upper())
            
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            
            handler.setLevel(category_level)
            handler.setFormatter(self.formatter)
            
            if self.enable_async:
                handler = AsyncLogHandler(handler)
            
            # Create category logger
            category_logger = logging.getLogger(f'revolutionary_fingerprint.{category_name}')
            category_logger.setLevel(category_level)
            category_logger.addHandler(handler)
            category_logger.propagate = False  # Don't propagate to parent
            
            setattr(self, f'{category_name}_logger', category_logger)
    
    def _log_with_context(self, level: int, msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
        """Log message with enhanced context information."""
        try:
            # Get caller information
            frame = inspect.currentframe().f_back.f_back
            caller_info = {
                'module': frame.f_globals.get('__name__', 'unknown'),
                'function': frame.f_code.co_name,
                'line': frame.f_lineno,
                'filename': frame.f_code.co_filename
            }
            
            # Create extra data
            extra = {
                'category': category,
                'caller_info': caller_info,
                **kwargs
            }
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.logs_per_second += 1  # This would need proper rate calculation
            
            # Log the message
            self.logger.log(level, msg, extra=extra)
            
        except Exception as e:
            # Fallback logging to prevent logging failures from breaking the application
            print(f"Logging error: {e} - Original message: {msg}", file=sys.stderr)
    
    # Public logging methods
    def trace(self, msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
        """Log trace message."""
        self._log_with_context(LogLevel.TRACE.value, msg, category, **kwargs)
    
    def debug(self, msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(LogLevel.DEBUG.value, msg, category, **kwargs)
    
    def info(self, msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(LogLevel.INFO.value, msg, category, **kwargs)
    
    def warning(self, msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(LogLevel.WARNING.value, msg, category, **kwargs)
    
    def error(self, msg: str, category: LogCategory = LogCategory.ERROR, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        if exc_info:
            kwargs['exc_info'] = True
        self._log_with_context(LogLevel.ERROR.value, msg, category, **kwargs)
    
    def critical(self, msg: str, category: LogCategory = LogCategory.ERROR, exc_info: bool = True, **kwargs) -> None:
        """Log critical message."""
        if exc_info:
            kwargs['exc_info'] = True
        self._log_with_context(LogLevel.CRITICAL.value, msg, category, **kwargs)
    
    def performance(self, msg: str, **kwargs) -> None:
        """Log performance message."""
        self._log_with_context(LogLevel.PERFORMANCE.value, msg, LogCategory.PERFORMANCE, **kwargs)
    
    def patent(self, msg: str, **kwargs) -> None:
        """Log patent documentation message."""
        if self.enable_patent:
            self._log_with_context(LogLevel.PATENT.value, msg, LogCategory.PATENT, **kwargs)
    
    def security(self, msg: str, **kwargs) -> None:
        """Log security message."""
        self._log_with_context(LogLevel.SECURITY.value, msg, LogCategory.SECURITY, **kwargs)
    
    # Context managers
    def log_execution_time(self, operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
        """Context manager for logging execution time."""
        return LogExecutionTime(self, operation_name, category)
    
    def log_memory_usage(self, operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
        """Context manager for logging memory usage."""
        return LogMemoryUsage(self, operation_name, category)
    
    # Utility methods
    def get_metrics(self) -> LogMetrics:
        """Get current logging metrics."""
        with self.metrics_lock:
            return self.metrics
    
    def rotate_logs(self) -> None:
        """Manually rotate log files."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.doRollover()
        
        self.info("Log files rotated manually")
    
    def compress_old_logs(self) -> None:
        """Compress old log files."""
        if not self.enable_compression:
            return
        
        log_files = list(self.log_dir.glob('*.log.*'))
        compressed_count = 0
        
        for log_file in log_files:
            if not log_file.name.endswith('.gz'):
                try:
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()  # Delete original
                    compressed_count += 1
                    
                except Exception as e:
                    self.warning(f"Failed to compress {log_file}: {e}")
        
        if compressed_count > 0:
            self.info(f"Compressed {compressed_count} old log files")
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """Clean up old log files."""
        import time
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        removed_count = 0
        for log_file in self.log_dir.glob('*.log*'):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            except Exception as e:
                self.warning(f"Failed to remove old log file {log_file}: {e}")
        
        if removed_count > 0:
            self.info(f"Cleaned up {removed_count} old log files")
    
    def shutdown(self) -> None:
        """Shutdown the logging system gracefully."""
        self.info("Shutting down Revolutionary Logging System...")
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            try:
                if isinstance(handler, AsyncLogHandler):
                    handler.close()
                else:
                    handler.close()
                self.logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing handler: {e}", file=sys.stderr)
        
        # Final cleanup
        self.compress_old_logs()


class LogExecutionTime:
    """Context manager for logging execution time."""
    
    def __init__(self, logger: RevolutionaryLogger, operation_name: str, category: LogCategory):
        self.logger = logger
        self.operation_name = operation_name
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
            
            if exc_type is None:
                self.logger.performance(
                    f"✅ {self.operation_name} completed in {execution_time:.3f}ms",
                    extra={'execution_time_ms': execution_time, 'category': self.category}
                )
            else:
                self.logger.error(
                    f"❌ {self.operation_name} failed after {execution_time:.3f}ms: {exc_val}",
                    category=LogCategory.ERROR,
                    extra={'execution_time_ms': execution_time}
                )


class LogMemoryUsage:
    """Context manager for logging memory usage."""
    
    def __init__(self, logger: RevolutionaryLogger, operation_name: str, category: LogCategory):
        self.logger = logger
        self.operation_name = operation_name
        self.category = category
        self.start_memory = None
    
    def __enter__(self):
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_memory is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - self.start_memory
                
                if exc_type is None:
                    self.logger.performance(
                        f"🧠 {self.operation_name} memory usage: {memory_delta:+.2f}MB (Total: {end_memory:.1f}MB)",
                        extra={'memory_delta_mb': memory_delta, 'total_memory_mb': end_memory, 'category': self.category}
                    )
                else:
                    self.logger.error(
                        f"❌ {self.operation_name} failed with memory delta: {memory_delta:+.2f}MB: {exc_val}",
                        category=LogCategory.ERROR,
                        extra={'memory_delta_mb': memory_delta}
                    )
            except Exception as e:
                self.logger.warning(f"Failed to measure memory usage for {self.operation_name}: {e}")


# Global logger instance - initialized on first import
_global_logger: Optional[RevolutionaryLogger] = None
_logger_lock = threading.Lock()


def get_revolutionary_logger(config_path: Optional[str] = None) -> RevolutionaryLogger:
    """
    Get the global revolutionary logger instance.
    
    This function provides a singleton logger that can be safely used across
    the entire revolutionary fingerprint system.
    
    Args:
        config_path: Path to configuration file (only used on first call)
    
    Returns:
        RevolutionaryLogger: The global logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = RevolutionaryLogger(config_path)
    
    return _global_logger


def setup_revolutionary_logging(config_path: Optional[str] = None) -> RevolutionaryLogger:
    """
    Setup and configure the revolutionary logging system.
    
    This is the main entry point for initializing logging in the
    revolutionary fingerprint system.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        RevolutionaryLogger: Configured logger instance
    """
    logger = get_revolutionary_logger(config_path)
    
    # Register shutdown handler
    def cleanup_logging():
        if _global_logger:
            _global_logger.shutdown()
    
    atexit.register(cleanup_logging)
    
    return logger


# Convenience functions for direct logging
def trace(msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
    """Log trace message using global logger."""
    get_revolutionary_logger().trace(msg, category, **kwargs)


def debug(msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
    """Log debug message using global logger."""
    get_revolutionary_logger().debug(msg, category, **kwargs)


def info(msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
    """Log info message using global logger."""
    get_revolutionary_logger().info(msg, category, **kwargs)


def warning(msg: str, category: LogCategory = LogCategory.SYSTEM, **kwargs) -> None:
    """Log warning message using global logger."""
    get_revolutionary_logger().warning(msg, category, **kwargs)


def error(msg: str, category: LogCategory = LogCategory.ERROR, exc_info: bool = False, **kwargs) -> None:
    """Log error message using global logger."""
    get_revolutionary_logger().error(msg, category, exc_info=exc_info, **kwargs)


def critical(msg: str, category: LogCategory = LogCategory.ERROR, exc_info: bool = True, **kwargs) -> None:
    """Log critical message using global logger."""
    get_revolutionary_logger().critical(msg, category, exc_info=exc_info, **kwargs)


def performance(msg: str, **kwargs) -> None:
    """Log performance message using global logger."""
    get_revolutionary_logger().performance(msg, **kwargs)


def patent(msg: str, **kwargs) -> None:
    """Log patent documentation using global logger."""
    get_revolutionary_logger().patent(msg, **kwargs)


def security(msg: str, **kwargs) -> None:
    """Log security message using global logger."""
    get_revolutionary_logger().security(msg, **kwargs)


def log_execution_time(operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
    """Context manager for logging execution time using global logger."""
    return get_revolutionary_logger().log_execution_time(operation_name, category)


def log_memory_usage(operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
    """Context manager for logging memory usage using global logger."""
    return get_revolutionary_logger().log_memory_usage(operation_name, category)


def demonstrate_revolutionary_logging():
    """
    Demonstrate the revolutionary logging system capabilities.
    
    Shows the comprehensive logging features designed for the
    O(1) fingerprint matching system.
    """
    print("=" * 80)
    print("🚀 REVOLUTIONARY LOGGING SYSTEM DEMONSTRATION")
    print("Patent Pending - Michael Derrick Jagneaux")
    print("=" * 80)
    
    # Initialize the logger
    logger = setup_revolutionary_logging()
    
    print(f"\n📊 Revolutionary Logging Configuration:")
    print(f"   Log Directory: {logger.log_dir}")
    print(f"   Performance Logging: {'✅' if logger.enable_performance else '❌'}")
    print(f"   Patent Documentation: {'✅' if logger.enable_patent else '❌'}")
    print(f"   Async Processing: {'✅' if logger.enable_async else '❌'}")
    
    # Demonstrate different log levels
    logger.info("🎯 Demonstrating Revolutionary Logging Capabilities", category=LogCategory.SYSTEM)
    
    # Performance logging demonstration
    with logger.log_execution_time("O(1) Fingerprint Match", LogCategory.O1_VALIDATION):
        time.sleep(0.001)  # Simulate operation
        logger.performance("Simulated O(1) operation completed successfully")
    
    # Memory usage demonstration
    with logger.log_memory_usage("Memory Intensive Operation", LogCategory.PERFORMANCE):
        # Simulate memory usage
        data = [i for i in range(1000)]
        logger.debug(f"Processed {len(data)} items in memory")
    
    # Patent logging demonstration
    if logger.enable_patent:
        logger.patent(
            "Patent Innovation: Characteristic-based O(1) addressing algorithm validated",
            extra={
                'innovation_type': 'algorithmic',
                'performance_gain': '99.9%',
                'mathematical_proof': 'Constant-time complexity proven through statistical analysis'
            }
        )
    
    # Security logging demonstration
    logger.security(
        "Security Event: Fingerprint database access validated",
        extra={'access_method': 'O(1)', 'validation_result': 'success'}
    )
    
    # Error handling demonstration
    try:
        raise ValueError("Simulated error for demonstration")
    except Exception as e:
        logger.error(
            "Demonstration error handled gracefully",
            category=LogCategory.ERROR,
            exc_info=True,
            extra={'error_type': 'demonstration', 'handled': True}
        )
    
    # Performance metrics
    if logger.performance:
        # Simulate O(1) validation logging
        logger.performance.log_o1_operation(
            operation_name="fingerprint_match",
            execution_time_ms=0.823,
            database_size=1000000,
            is_constant_time=True,
            confidence_score=0.998,
            algorithm_version="v2.1",
            patent_reference="US-PENDING-2024"
        )
        
        # Generate performance report
        report = logger.performance.generate_performance_report()
        if report:
            logger.info(f"📈 Performance Report Generated: {report.get('total_operations', 0)} operations logged")
    
    # Log completion
    logger.info("✅ Revolutionary Logging System Demonstration Complete", category=LogCategory.SYSTEM)
    
    print(f"\n📁 Log files created in: {logger.log_dir}")
    print("🔍 Check the log files to see the comprehensive logging output!")
    
    return logger


if __name__ == "__main__":
    # Run demonstration if executed directly
    demo_logger = demonstrate_revolutionary_logging()
    
    # Clean shutdown
    demo_logger.shutdown()
    print("\n🎯 Revolutionary Logging System shutdown complete!")
