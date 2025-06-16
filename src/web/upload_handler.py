#!/usr/bin/env python3
"""
Revolutionary Secure Upload Handler - Complete Fixed Version
Patent Pending - Michael Derrick Jagneaux

Production-ready file upload handler with comprehensive security validation,
image processing, batch upload capabilities, and fingerprint-specific optimizations.
"""

import os
import sys
import time
import hashlib
import tempfile
import json
import logging
import statistics
import concurrent.futures
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Flask and web components
from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Image processing
from PIL import Image, ImageStat
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import core components
try:
    from src.utils.logger import setup_logger
    from src.utils.timing_utils import HighPrecisionTimer
except ImportError:
    # Fallback logging if custom logger not available
    logging.basicConfig(level=logging.INFO)
    setup_logger = logging.getLogger

logger = setup_logger(__name__)


class FileType(Enum):
    """Supported file types for fingerprint processing."""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    UNKNOWN = "unknown"


class UploadStatus(Enum):
    """Upload processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class UploadResult:
    """Result of file upload processing."""
    success: bool
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    file_type: FileType
    processing_time_ms: float
    validation_results: Dict[str, Any]
    error_message: str = ""
    fingerprint_data: Optional[Dict[str, Any]] = None
    upload_status: UploadStatus = UploadStatus.COMPLETE


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    issues: List[str]
    recommendations: List[str]
    quality_score: float
    dimensions: Tuple[int, int]
    file_size: int
    file_type: FileType


class SecurityValidationError(Exception):
    """Exception raised for security validation failures."""
    pass


class SecureUploadHandler:
    """
    Revolutionary Secure Upload Handler
    
    Provides comprehensive file upload security, validation, and processing
    specifically optimized for fingerprint image handling with batch capabilities.
    """
    
    def __init__(self, upload_folder: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the secure upload handler."""
        self.upload_folder = Path(upload_folder)
        self.config = config or {}
        
        # Configuration with defaults
        self.max_file_size = self.config.get('max_file_size_mb', 16) * 1024 * 1024
        self.max_batch_size = self.config.get('max_batch_size', 50)
        self.processing_timeout = self.config.get('processing_timeout_seconds', 30)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 256)
        
        self.timer = HighPrecisionTimer() if 'HighPrecisionTimer' in globals() else time
        
        # Security configurations
        self.allowed_extensions = {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'}
        self.allowed_mimetypes = {
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/tiff', 'image/bmp', 'image/x-ms-bmp'
        }
        
        # Image quality requirements for fingerprints
        self.min_resolution = self.config.get('min_image_dimensions', (200, 200))
        self.max_resolution = self.config.get('max_image_dimensions', (4000, 4000))
        self.min_quality_score = self.config.get('quality_threshold', 0.3)
        
        # Create upload directory
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_bytes_processed': 0,
            'avg_processing_time': 0.0,
            'batch_uploads': 0,
            'security_rejections': 0
        }
        
        # Thread lock for statistics
        self._stats_lock = threading.Lock()
        
        # Optional fingerprint processor
        self.fingerprint_processor = None
        
        logger.info(f"SecureUploadHandler initialized: {self.upload_folder}")
    
    def set_fingerprint_processor(self, processor):
        """Set fingerprint processor for enhanced validation."""
        self.fingerprint_processor = processor
        logger.info("Fingerprint processor attached to upload handler")
    
    def process_upload(self, file: FileStorage) -> UploadResult:
        """
        Process uploaded file with comprehensive security validation.
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            UploadResult with processing details
        """
        return self.handle_single_upload(file, process_fingerprint=False)
    
    def handle_single_upload(self, file: FileStorage, 
                           process_fingerprint: bool = True,
                           store_in_database: bool = True) -> UploadResult:
        """Handle single file upload with full processing pipeline."""
        start_time = time.perf_counter()
        
        with self._stats_lock:
            self.stats['total_uploads'] += 1
        
        try:
            # Initial validation
            if not file or file.filename == '':
                return self._create_error_result("No file provided", "", "")
            
            original_filename = file.filename
            logger.info(f"Processing upload: {original_filename}")
            
            # Security validation
            validation_result = self._validate_file(file)
            if not validation_result.is_valid:
                with self._stats_lock:
                    self.stats['security_rejections'] += 1
                return self._create_error_result(
                    f"Validation failed: {', '.join(validation_result.issues)}",
                    original_filename, ""
                )
            
            # Generate secure filename
            safe_filename = self._generate_secure_filename(original_filename)
            file_path = self.upload_folder / safe_filename
            
            # Save file securely
            with self.timer.time_operation("file_save"):
                file.save(str(file_path))
            
            # Verify file integrity
            if not self._verify_file_integrity(file_path):
                file_path.unlink(missing_ok=True)
                return self._create_error_result("File integrity check failed", original_filename, "")
            
            # Additional quality validation
            quality_validation = self._perform_quality_validation(file_path)
            
            # Process fingerprint if requested
            fingerprint_data = None
            if process_fingerprint:
                try:
                    fingerprint_data = self._extract_fingerprint_preview(file_path)
                except Exception as e:
                    logger.warning(f"Fingerprint processing failed: {e}")
                    fingerprint_data = {'processing_error': str(e)}
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Update statistics
            with self._stats_lock:
                self.stats['successful_uploads'] += 1
                self.stats['total_bytes_processed'] += validation_result.file_size
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['total_uploads'] - 1) + processing_time)
                    / self.stats['total_uploads']
                )
            
            # Create successful result
            result = UploadResult(
                success=True,
                filename=safe_filename,
                original_filename=original_filename,
                file_path=str(file_path),
                file_size=validation_result.file_size,
                file_type=validation_result.file_type,
                processing_time_ms=processing_time,
                validation_results={
                    'security_validation': validation_result.__dict__,
                    'quality_validation': quality_validation,
                    'file_save_time_ms': getattr(self.timer, 'get_last_operation_time', lambda x: 0)("file_save")
                },
                fingerprint_data=fingerprint_data
            )
            
            logger.info(f"Upload successful: {original_filename} -> {safe_filename}")
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            with self._stats_lock:
                self.stats['failed_uploads'] += 1
            return self._create_error_result(f"Upload processing failed: {str(e)}", 
                                           original_filename or "unknown", "")
    
    def handle_batch_upload(self, files: List[FileStorage], 
                          max_workers: int = 4,
                          process_fingerprints: bool = True) -> Dict[str, Any]:
        """Handle batch file upload with parallel processing."""
        batch_start = time.perf_counter()
        
        logger.info(f"Starting batch upload: {len(files)} files")
        
        with self._stats_lock:
            self.stats['batch_uploads'] += 1
        
        # Validate batch size
        if len(files) > self.max_batch_size:
            return {
                'success': False,
                'error': f'Batch size too large: {len(files)} (max {self.max_batch_size})'
            }
        
        results = []
        failed_uploads = []
        total_bytes = 0
        
        # Process files with thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_index = {
                executor.submit(
                    self.handle_single_upload, 
                    file, 
                    process_fingerprint=process_fingerprints,
                    store_in_database=True
                ): i for i, file in enumerate(files)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index, timeout=self.processing_timeout):
                index = future_to_index[future]
                file = files[index]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        results.append({
                            'index': index,
                            'filename': result.filename,
                            'original_filename': result.original_filename,
                            'file_size': result.file_size,
                            'file_type': result.file_type.value,
                            'processing_time_ms': result.processing_time_ms,
                            'quality_score': result.validation_results.get('quality_validation', {}).get('quality_score', 0),
                            'fingerprint_data': result.fingerprint_data
                        })
                        total_bytes += result.file_size
                    else:
                        failed_uploads.append({
                            'index': index,
                            'filename': file.filename,
                            'error': result.error_message
                        })
                        
                except Exception as e:
                    failed_uploads.append({
                        'index': index,
                        'filename': getattr(file, 'filename', 'unknown'),
                        'error': str(e)
                    })
                
                # Progress logging
                completed = len(results) + len(failed_uploads)
                if completed % 10 == 0 or completed == len(files):
                    logger.info(f"Batch progress: {completed}/{len(files)} files processed")
        
        batch_time = (time.perf_counter() - batch_start) * 1000
        
        batch_summary = {
            'success': True,
            'total_files': len(files),
            'successful_uploads': len(results),
            'failed_uploads': len(failed_uploads),
            'success_rate': (len(results) / len(files)) * 100 if files else 0,
            'total_bytes_processed': total_bytes,
            'batch_processing_time_ms': batch_time,
            'average_time_per_file_ms': batch_time / len(files) if files else 0,
            'throughput_files_per_second': len(files) / (batch_time / 1000) if batch_time > 0 else 0,
            'results': results,
            'failed_uploads': failed_uploads
        }
        
        logger.info(f"Batch upload completed: {len(results)} successful, {len(failed_uploads)} failed")
        return batch_summary
    
    def process_upload_with_fingerprint_processing(self, file: FileStorage) -> UploadResult:
        """
        Process upload with integrated fingerprint processing.
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            UploadResult with fingerprint processing data
        """
        return self.handle_single_upload(file, process_fingerprint=True, store_in_database=True)
    
    def _validate_file(self, file: FileStorage) -> ValidationResult:
        """
        Comprehensive file security validation.
        
        Args:
            file: File to validate
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        recommendations = []
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            issues.append("File is empty")
        elif file_size > self.max_file_size:
            issues.append(f"File too large ({file_size} bytes, max {self.max_file_size})")
        
        # Check file extension
        filename = file.filename.lower()
        file_ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
        
        if file_ext not in self.allowed_extensions:
            issues.append(f"Invalid file extension: {file_ext}")
        
        # Check MIME type
        mimetype = file.mimetype
        if mimetype not in self.allowed_mimetypes:
            issues.append(f"Invalid MIME type: {mimetype}")
        
        # Determine file type
        file_type = self._determine_file_type(file_ext, mimetype)
        
        # Basic image validation
        dimensions = (0, 0)
        quality_score = 0.0
        
        try:
            # Read image header for basic validation
            file.seek(0)
            image_data = file.read(1024)  # Read first 1KB
            file.seek(0)
            
            # Try to determine dimensions using PIL
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}')
            try:
                # Write data to temp file for validation
                file.seek(0)
                temp_file.write(file.read())
                temp_file.flush()
                file.seek(0)  # Reset file pointer
                
                # Validate image with PIL
                with Image.open(temp_file.name) as img:
                    dimensions = img.size
                    
                    # Check resolution requirements
                    if dimensions[0] < self.min_resolution[0] or dimensions[1] < self.min_resolution[1]:
                        issues.append(f"Resolution too low: {dimensions[0]}x{dimensions[1]} (min {self.min_resolution[0]}x{self.min_resolution[1]})")
                    
                    if dimensions[0] > self.max_resolution[0] or dimensions[1] > self.max_resolution[1]:
                        issues.append(f"Resolution too high: {dimensions[0]}x{dimensions[1]} (max {self.max_resolution[0]}x{self.max_resolution[1]})")
                    
                    # Basic quality assessment
                    quality_score = self._assess_image_quality(img)
                    
                    if quality_score < self.min_quality_score:
                        recommendations.append(f"Image quality is low ({quality_score:.2f}). Consider using a higher quality image.")
                
            finally:
                # Clean up temp file
                temp_file.close()
                os.unlink(temp_file.name)
                
        except Exception as e:
            issues.append(f"Image validation failed: {str(e)}")
            logger.warning(f"Image validation error: {e}")
        
        # Security filename checks
        dangerous_patterns = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*', '\x00']
        for pattern in dangerous_patterns:
            if pattern in filename:
                issues.append(f"Filename contains dangerous character: {pattern}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            recommendations=recommendations,
            quality_score=quality_score,
            dimensions=dimensions,
            file_size=file_size,
            file_type=file_type
        )
    
    def _perform_quality_validation(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive quality validation on saved file."""
        try:
            with Image.open(file_path) as img:
                quality_score = self._assess_image_quality(img)
                
                # Additional quality metrics
                img_array = np.array(img.convert('L'))
                
                # Sharpness assessment
                laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
                
                # Contrast assessment
                contrast = np.std(img_array)
                
                # Brightness distribution
                hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
                brightness_mean = np.mean(img_array)
                
                return {
                    'quality_score': quality_score,
                    'sharpness_score': min(laplacian_var / 1000.0, 1.0),
                    'contrast_score': contrast / 255.0,
                    'brightness_mean': brightness_mean / 255.0,
                    'dimensions': img.size,
                    'mode': img.mode,
                    'format': img.format
                }
        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return {'error': str(e), 'quality_score': 0.0}
    
    def _extract_fingerprint_preview(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic fingerprint characteristics for preview."""
        try:
            if self.fingerprint_processor:
                result = self.fingerprint_processor.process_fingerprint(str(file_path))
                return {
                    'pattern_class': getattr(result, 'pattern_class', 'UNKNOWN'),
                    'primary_address': getattr(result, 'primary_address', '000.000.000.000.000'),
                    'image_quality': getattr(result, 'image_quality', 0.0),
                    'confidence_score': getattr(result, 'confidence_score', 0.0),
                    'processing_time_ms': getattr(result, 'processing_time_ms', 0.0)
                }
            else:
                # Basic analysis without full processor
                with Image.open(file_path) as img:
                    img_array = np.array(img.convert('L'))
                    
                    # Simple pattern detection placeholder
                    return {
                        'pattern_class': 'PREVIEW_MODE',
                        'image_quality': self._assess_image_quality(img),
                        'dimensions': img.size,
                        'file_size': file_path.stat().st_size,
                        'preview_available': True
                    }
        except Exception as e:
            logger.warning(f"Fingerprint preview failed: {e}")
            return {'error': str(e), 'preview_available': False}
    
    def _assess_image_quality(self, img: Image.Image) -> float:
        """
        Assess image quality for fingerprint processing.
        
        Args:
            img: PIL Image object
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Convert to grayscale for analysis
            if img.mode != 'L':
                img_gray = img.convert('L')
            else:
                img_gray = img
            
            # Calculate basic quality metrics
            img_array = np.array(img_gray)
            
            # Contrast (standard deviation of pixel values)
            contrast = np.std(img_array) / 255.0
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Brightness distribution (avoid over/under exposure)
            hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            
            # Penalize images with too much in the extremes
            extreme_pixels = hist_norm[0:10].sum() + hist_norm[246:256].sum()
            brightness_score = 1.0 - min(extreme_pixels * 2, 1.0)
            
            # Combined quality score
            quality_score = (contrast * 0.4 + sharpness * 0.4 + brightness_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    def _generate_secure_filename(self, original_filename: str) -> str:
        """
        Generate a secure filename preventing directory traversal and conflicts.
        
        Args:
            original_filename: Original uploaded filename
            
        Returns:
            Secure filename string
        """
        # Secure the filename
        safe_name = secure_filename(original_filename)
        
        # Add timestamp for uniqueness
        timestamp = str(int(time.time() * 1000))
        
        # Generate hash for additional security
        hash_source = f"{original_filename}{timestamp}{time.time()}"
        file_hash = hashlib.md5(hash_source.encode()).hexdigest()[:8]
        
        # Preserve extension
        name_part = safe_name.rsplit('.', 1)[0] if '.' in safe_name else safe_name
        ext_part = safe_name.rsplit('.', 1)[1] if '.' in safe_name else 'bin'
        
        return f"{timestamp}_{file_hash}_{name_part}.{ext_part}"
    
    def _determine_file_type(self, extension: str, mimetype: str) -> FileType:
        """Determine file type from extension and MIME type."""
        ext = extension.lower()
        
        if ext in ['jpg', 'jpeg'] or 'jpeg' in mimetype:
            return FileType.JPEG
        elif ext == 'png' or 'png' in mimetype:
            return FileType.PNG
        elif ext in ['tif', 'tiff'] or 'tiff' in mimetype:
            return FileType.TIFF
        elif ext == 'bmp' or 'bmp' in mimetype:
            return FileType.BMP
        else:
            return FileType.UNKNOWN
    
    def _verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity after upload."""
        try:
            # Check file exists and is readable
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
            
            # Try to open as image
            with Image.open(file_path) as img:
                img.verify()  # Verify image integrity
            
            return True
            
        except Exception as e:
            logger.warning(f"File integrity check failed: {e}")
            return False
    
    def _create_error_result(self, error_message: str, original_filename: str, file_path: str) -> UploadResult:
        """Create error result."""
        return UploadResult(
            success=False,
            filename="",
            original_filename=original_filename,
            file_path=file_path,
            file_size=0,
            file_type=FileType.UNKNOWN,
            processing_time_ms=0.0,
            validation_results={},
            error_message=error_message,
            upload_status=UploadStatus.FAILED
        )
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Get upload handler statistics."""
        with self._stats_lock:
            return {
                'total_uploads': self.stats['total_uploads'],
                'successful_uploads': self.stats['successful_uploads'],
                'failed_uploads': self.stats['failed_uploads'],
                'success_rate': (self.stats['successful_uploads'] / self.stats['total_uploads'] * 100) if self.stats['total_uploads'] > 0 else 0,
                'total_bytes_processed': self.stats['total_bytes_processed'],
                'average_processing_time_ms': self.stats['avg_processing_time'],
                'batch_uploads': self.stats['batch_uploads'],
                'security_rejections': self.stats['security_rejections'],
                'upload_folder': str(self.upload_folder),
                'configuration': {
                    'max_file_size_mb': self.max_file_size / (1024 * 1024),
                    'max_batch_size': self.max_batch_size,
                    'allowed_extensions': list(self.allowed_extensions),
                    'min_resolution': self.min_resolution,
                    'max_resolution': self.max_resolution,
                    'processing_timeout_seconds': self.processing_timeout
                }
            }
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old uploaded files."""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            cleaned_files = []
            total_size_cleaned = 0
            
            for file_path in self.upload_folder.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_files.append(str(file_path.name))
                        total_size_cleaned += file_size
            
            return {
                'files_cleaned': len(cleaned_files),
                'total_size_cleaned_bytes': total_size_cleaned,
                'total_size_cleaned_mb': total_size_cleaned / (1024 * 1024),
                'cleaned_files': cleaned_files
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'error': str(e)}
    
    def validate_upload_health(self) -> Dict[str, Any]:
        """Validate upload handler health and performance."""
        try:
            # Check disk space
            upload_folder_stats = os.statvfs(self.upload_folder)
            free_space_bytes = upload_folder_stats.f_frsize * upload_folder_stats.f_bavail
            free_space_mb = free_space_bytes / (1024 * 1024)
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Performance health check
            health_status = {
                'status': 'healthy',
                'timestamp': time.time(),
                'disk_space_mb': free_space_mb,
                'memory_usage_mb': memory_mb,
                'upload_folder_exists': self.upload_folder.exists(),
                'upload_folder_writable': os.access(self.upload_folder, os.W_OK),
                'statistics': self.get_upload_statistics()
            }
            
            # Health warnings
            warnings = []
            if free_space_mb < 100:
                warnings.append(f"Low disk space: {free_space_mb:.1f}MB remaining")
            if memory_mb > self.memory_limit_mb:
                warnings.append(f"High memory usage: {memory_mb:.1f}MB (limit: {self.memory_limit_mb}MB)")
            
            health_status['warnings'] = warnings
            health_status['status'] = 'degraded' if warnings else 'healthy'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }


# Example usage
if __name__ == '__main__':
    # Test the upload handler
    config = {
        'max_file_size_mb': 16,
        'max_batch_size': 50,
        'min_image_dimensions': (200, 200),
        'max_image_dimensions': (4000, 4000),
        'quality_threshold': 0.3,
        'processing_timeout_seconds': 30,
        'memory_limit_mb': 256
    }
    
    upload_handler = SecureUploadHandler('data/uploads', config)
    
    print("SecureUploadHandler initialized successfully!")
    print("Configuration:", json.dumps(upload_handler.get_upload_statistics(), indent=2))
    print("Health Status:", json.dumps(upload_handler.validate_upload_health(), indent=2))