#!/usr/bin/env python3
"""
Revolutionary Secure Upload Handler - Complete Fixed Version
Patent Pending - Michael Derrick Jagneaux

Production-ready file upload handler with comprehensive security validation,
image processing, and fingerprint-specific optimizations.
"""

import os
import sys
import time
import hashlib
import tempfile
import json
import logging
import statistics
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


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    issues: List[str]
    recommendations: List[str]
    quality_score: float
    dimensions: Tuple[int, int]


class SecureUploadHandler:
    """
    Revolutionary Secure Upload Handler
    
    Provides comprehensive file upload security, validation, and processing
    specifically optimized for fingerprint image handling.
    """
    
    def __init__(self, upload_folder: str, max_file_size: int = 16 * 1024 * 1024):
        """Initialize the secure upload handler."""
        self.upload_folder = Path(upload_folder)
        self.max_file_size = max_file_size
        self.timer = HighPrecisionTimer() if 'HighPrecisionTimer' in globals() else time
        
        # Security configurations
        self.allowed_extensions = {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'}
        self.allowed_mimetypes = {
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/tiff', 'image/bmp', 'image/x-ms-bmp'
        }
        
        # Image quality requirements for fingerprints
        self.min_resolution = (200, 200)  # Minimum DPI equivalent
        self.max_resolution = (4000, 4000)  # Maximum to prevent resource exhaustion
        self.min_quality_score = 0.3  # Minimum acceptable quality
        
        # Create upload directory
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_bytes_processed': 0,
            'avg_processing_time': 0.0
        }
        
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
        start_time = self.timer.time_ms() if hasattr(self.timer, 'time_ms') else time.time() * 1000
        self.stats['total_uploads'] += 1
        
        try:
            # Basic file validation
            if not file or not file.filename:
                return self._create_error_result("No file provided", "", "")
            
            original_filename = file.filename
            logger.info(f"Processing upload: {original_filename}")
            
            # Security validation
            validation_result = self._validate_file_security(file)
            if not validation_result.is_valid:
                error_msg = "; ".join(validation_result.issues)
                return self._create_error_result(error_msg, original_filename, "")
            
            # Generate secure filename
            secure_name = self._generate_secure_filename(original_filename)
            file_path = self.upload_folder / secure_name
            
            # Save file securely
            file.save(str(file_path))
            
            # Verify file integrity
            if not self._verify_file_integrity(file_path):
                file_path.unlink(missing_ok=True)  # Clean up
                return self._create_error_result("File integrity check failed", original_filename, str(file_path))
            
            # Get file information
            file_size = file_path.stat().st_size
            file_type = self._determine_file_type(
                file_path.suffix.lower().lstrip('.'), 
                file.mimetype
            )
            
            # Update statistics
            self.stats['successful_uploads'] += 1
            self.stats['total_bytes_processed'] += file_size
            
            processing_time = (self.timer.time_ms() if hasattr(self.timer, 'time_ms') else time.time() * 1000) - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_uploads'] - 1) + processing_time) /
                self.stats['successful_uploads']
            )
            
            logger.info(f"Upload successful: {secure_name} ({file_size} bytes, {processing_time:.2f}ms)")
            
            return UploadResult(
                success=True,
                filename=secure_name,
                original_filename=original_filename,
                file_path=str(file_path),
                file_size=file_size,
                file_type=file_type,
                processing_time_ms=processing_time,
                validation_results=validation_result.__dict__
            )
            
        except Exception as e:
            self.stats['failed_uploads'] += 1
            error_msg = f"Upload processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg, original_filename, "")
    
    def process_upload_with_fingerprint_processing(self, file: FileStorage) -> UploadResult:
        """
        Process upload with integrated fingerprint processing.
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            UploadResult with fingerprint processing data
        """
        # First process the upload normally
        upload_result = self.process_upload(file)
        
        if not upload_result.success:
            return upload_result
        
        # If fingerprint processor is available, process the fingerprint
        if self.fingerprint_processor:
            try:
                fingerprint_result = self.fingerprint_processor.process_fingerprint(upload_result.file_path)
                upload_result.fingerprint_data = {
                    'pattern_class': getattr(fingerprint_result, 'pattern_class', 'UNKNOWN'),
                    'primary_address': getattr(fingerprint_result, 'primary_address', '000.000.000.000.000'),
                    'image_quality': getattr(fingerprint_result, 'image_quality', 0.0),
                    'confidence_score': getattr(fingerprint_result, 'confidence_score', 0.0),
                    'processing_time_ms': getattr(fingerprint_result, 'processing_time_ms', 0.0)
                }
                logger.info(f"Fingerprint processing completed for {upload_result.filename}")
            except Exception as e:
                logger.warning(f"Fingerprint processing failed: {e}")
                upload_result.fingerprint_data = {'error': str(e)}
        
        return upload_result
    
    def _validate_file_security(self, file: FileStorage) -> ValidationResult:
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
            dimensions=dimensions
        )
    
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
            error_message=error_message
        )
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Get upload handler statistics."""
        return {
            'total_uploads': self.stats['total_uploads'],
            'successful_uploads': self.stats['successful_uploads'],
            'failed_uploads': self.stats['failed_uploads'],
            'success_rate': (self.stats['successful_uploads'] / self.stats['total_uploads'] * 100) if self.stats['total_uploads'] > 0 else 0,
            'total_bytes_processed': self.stats['total_bytes_processed'],
            'average_processing_time_ms': self.stats['avg_processing_time'],
            'upload_folder': str(self.upload_folder),
            'configuration': {
                'max_file_size_mb': self.max_file_size / (1024 * 1024),
                'allowed_extensions': list(self.allowed_extensions),
                'min_resolution': self.min_resolution,
                'max_resolution': self.max_resolution
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


# Example usage
if __name__ == '__main__':
    # Test the upload handler
    upload_handler = SecureUploadHandler('data/uploads')
    
    print("SecureUploadHandler initialized successfully!")
    print("Configuration:", upload_handler.get_upload_statistics())
