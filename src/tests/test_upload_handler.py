#!/usr/bin/env python3
"""
Revolutionary O(1) Fingerprint System - Upload Handler Tests
Patent Pending - Michael Derrick Jagneaux

Comprehensive tests for the secure upload handler, validating file security,
upload performance, validation accuracy, and integration with the O(1) system.
These tests ensure the upload gateway maintains security while enabling
high-throughput fingerprint processing.

Test Coverage:
- Secure file upload validation
- Multi-format image support verification
- Batch upload performance testing
- Memory management under load
- Security vulnerability prevention
- Quality validation accuracy
- Error handling robustness
- Integration with O(1) processor
"""

import pytest
import tempfile
import shutil
import os
import io
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
from werkzeug.datastructures import FileStorage

# Image processing imports
import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.web.upload_handler import (
    SecureUploadHandler,
    UploadResult,
    ValidationResult,
    UploadStatus,
    FileType
)
from src.tests import TestConfig, TestDataGenerator, TestUtils


class TestSecureUploadHandler:
    """
    Comprehensive test suite for the Revolutionary Secure Upload Handler.
    
    Validates all aspects of secure file upload processing for the O(1)
    fingerprint matching system, ensuring security, performance, and reliability.
    """
    
    @pytest.fixture
    def temp_upload_dir(self):
        """Create temporary upload directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="o1_upload_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def upload_handler(self, temp_upload_dir):
        """Create upload handler instance for testing."""
        config = {
            'max_file_size_mb': 10,
            'allowed_formats': ['JPEG', 'PNG', 'TIFF', 'BMP'],
            'min_image_dimensions': (100, 100),
            'max_image_dimensions': (4096, 4096),
            'quality_threshold': 0.3,
            'enable_virus_scanning': False,  # Disabled for testing
            'quarantine_suspicious': True,
            'processing_timeout_seconds': 30,
            'memory_limit_mb': 100
        }
        return SecureUploadHandler(temp_upload_dir, config)
    
    @pytest.fixture
    def mock_fingerprint_processor(self):
        """Create mock fingerprint processor for testing."""
        mock_processor = Mock()
        mock_processor.process_fingerprint.return_value = Mock(
            success=True,
            primary_address="FP.LOOP_R.GOOD_MED.AVG_CTR",
            confidence_score=0.89,
            processing_time_ms=45.2,
            pattern_class="LOOP_RIGHT",
            quality_score=0.82
        )
        return mock_processor
    
    @pytest.fixture
    def sample_files(self):
        """Generate sample files for upload testing."""
        files = {}
        
        # Valid fingerprint images
        for pattern in ["LOOP_RIGHT", "LOOP_LEFT", "WHORL", "ARCH"]:
            image = TestDataGenerator.create_synthetic_fingerprint(pattern)
            
            # Convert to different formats
            for fmt in ["JPEG", "PNG"]:
                _, buffer = cv2.imencode(f'.{fmt.lower()}', image)
                files[f'valid_{pattern}_{fmt.lower()}'] = {
                    'data': buffer.tobytes(),
                    'filename': f'test_{pattern}.{fmt.lower()}',
                    'content_type': f'image/{fmt.lower()}',
                    'expected_valid': True
                }
        
        # Invalid files for security testing
        files['malicious_script'] = {
            'data': b'#!/bin/bash\nrm -rf /',
            'filename': 'malicious.jpg',
            'content_type': 'image/jpeg',
            'expected_valid': False
        }
        
        files['empty_file'] = {
            'data': b'',
            'filename': 'empty.jpg',
            'content_type': 'image/jpeg',
            'expected_valid': False
        }
        
        files['oversized_file'] = {
            'data': b'fake_image_data' * 1000000,  # ~15MB
            'filename': 'huge.jpg',
            'content_type': 'image/jpeg',
            'expected_valid': False
        }
        
        # Corrupted image
        files['corrupted_image'] = {
            'data': b'\xff\xd8\xff\xe0INVALID_JPEG_DATA',
            'filename': 'corrupted.jpg',
            'content_type': 'image/jpeg',
            'expected_valid': False
        }
        
        return files
    
    # ==========================================
    # BASIC FUNCTIONALITY TESTS
    # ==========================================
    
    def test_handler_initialization(self, upload_handler, temp_upload_dir):
        """Test upload handler initializes correctly."""
        assert upload_handler.upload_folder == temp_upload_dir
        assert upload_handler.max_file_size == 10 * 1024 * 1024  # 10MB in bytes
        assert FileType.JPEG in upload_handler.allowed_formats
        assert upload_handler.upload_stats['total_uploads'] == 0
        assert os.path.exists(temp_upload_dir)
    
    def test_valid_single_upload(self, upload_handler, sample_files):
        """Test successful single file upload."""
        # Test with valid JPEG fingerprint
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        file_storage = FileStorage(
            stream=io.BytesIO(file_data['data']),
            filename=file_data['filename'],
            content_type=file_data['content_type']
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Validate successful upload
        assert result.success is True
        assert result.error_message is None
        assert result.file_type == FileType.JPEG
        assert result.file_size > 0
        assert result.processing_time_ms > 0
        assert result.validation_results['is_valid'] is True
        
        # Verify file was saved
        assert os.path.exists(result.file_path)
        
        # Verify filename security
        assert result.filename != result.original_filename  # Should be sanitized
        assert not any(char in result.filename for char in ['/', '\\', '..', '<', '>'])
    
    def test_batch_upload_performance(self, upload_handler, sample_files):
        """Test batch upload performance and memory management."""
        # Prepare batch of files
        batch_files = []
        for i in range(5):
            file_data = sample_files['valid_LOOP_RIGHT_jpeg']
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=f'batch_{i:03d}.jpg',
                content_type=file_data['content_type']
            )
            batch_files.append(file_storage)
        
        # Process batch
        start_time = time.perf_counter()
        results = upload_handler.process_batch_upload(batch_files)
        end_time = time.perf_counter()
        
        batch_time_ms = (end_time - start_time) * 1000
        
        # Validate batch results
        assert len(results) == 5
        assert all(result.success for result in results)
        
        # Performance validation - should process efficiently
        avg_time_per_file = batch_time_ms / len(batch_files)
        assert avg_time_per_file < 1000, f"Batch processing too slow: {avg_time_per_file:.2f}ms/file"
        
        # Memory management - no memory leaks
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 200, f"Memory usage too high after batch: {memory_mb:.1f}MB"
    
    # ==========================================
    # SECURITY VALIDATION TESTS
    # ==========================================
    
    def test_malicious_file_detection(self, upload_handler, sample_files):
        """Test detection and rejection of malicious files."""
        malicious_file = sample_files['malicious_script']
        file_storage = FileStorage(
            stream=io.BytesIO(malicious_file['data']),
            filename=malicious_file['filename'],
            content_type=malicious_file['content_type']
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should reject malicious file
        assert result.success is False
        assert result.error_message is not None
        assert "security" in result.error_message.lower() or "invalid" in result.error_message.lower()
        assert result.validation_results['is_valid'] is False
    
    def test_file_size_limits(self, upload_handler, sample_files):
        """Test file size limit enforcement."""
        oversized_file = sample_files['oversized_file']
        file_storage = FileStorage(
            stream=io.BytesIO(oversized_file['data']),
            filename=oversized_file['filename'],
            content_type=oversized_file['content_type']
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should reject oversized file
        assert result.success is False
        assert "size" in result.error_message.lower()
        assert result.file_size > upload_handler.max_file_size
    
    def test_filename_sanitization(self, upload_handler, sample_files):
        """Test filename sanitization for security."""
        dangerous_filenames = [
            "../../../etc/passwd.jpg",
            "file<script>alert('xss')</script>.jpg",
            "file\x00null.jpg",
            "../../windows/system32/config.jpg",
            "normal..file.jpg"
        ]
        
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        for dangerous_name in dangerous_filenames:
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=dangerous_name,
                content_type=file_data['content_type']
            )
            
            result = upload_handler.process_upload(file_storage)
            
            if result.success:  # If file passed validation
                # Filename should be sanitized
                assert not any(char in result.filename for char in ['/', '\\', '..', '<', '>', '\x00'])
                assert result.filename != dangerous_name
                assert result.filename.endswith('.jpg')
    
    def test_mime_type_validation(self, upload_handler):
        """Test MIME type validation and spoofing detection."""
        # Create fake image with wrong MIME type
        fake_image_data = b"This is not an image file"
        
        file_storage = FileStorage(
            stream=io.BytesIO(fake_image_data),
            filename="fake.jpg",
            content_type="image/jpeg"  # Claiming to be JPEG
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should detect MIME type spoofing
        assert result.success is False
        assert "format" in result.error_message.lower() or "invalid" in result.error_message.lower()
    
    # ==========================================
    # IMAGE VALIDATION TESTS
    # ==========================================
    
    def test_image_quality_assessment(self, upload_handler):
        """Test image quality assessment functionality."""
        # Create low quality image
        low_quality_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', low_quality_image, [cv2.IMWRITE_JPEG_QUALITY, 10])
        
        file_storage = FileStorage(
            stream=io.BytesIO(buffer.tobytes()),
            filename="low_quality.jpg",
            content_type="image/jpeg"
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should detect poor quality
        if result.success:
            assert result.validation_results['quality_score'] < 0.5
        else:
            assert "quality" in result.error_message.lower()
    
    def test_image_dimension_validation(self, upload_handler):
        """Test image dimension validation."""
        # Create tiny image (below minimum)
        tiny_image = np.ones((50, 50), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', tiny_image)
        
        file_storage = FileStorage(
            stream=io.BytesIO(buffer.tobytes()),
            filename="tiny.jpg",
            content_type="image/jpeg"
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should reject tiny image
        assert result.success is False
        assert "dimension" in result.error_message.lower() or "size" in result.error_message.lower()
    
    def test_corrupted_image_handling(self, upload_handler, sample_files):
        """Test handling of corrupted image files."""
        corrupted_file = sample_files['corrupted_image']
        file_storage = FileStorage(
            stream=io.BytesIO(corrupted_file['data']),
            filename=corrupted_file['filename'],
            content_type=corrupted_file['content_type']
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should handle corruption gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "corrupt" in result.error_message.lower() or "invalid" in result.error_message.lower()
    
    # ==========================================
    # PERFORMANCE AND CONCURRENCY TESTS
    # ==========================================
    
    def test_concurrent_uploads(self, upload_handler, sample_files):
        """Test concurrent upload handling."""
        num_threads = 8
        uploads_per_thread = 3
        results_queue = queue.Queue()
        
        def upload_worker(thread_id):
            """Worker function for concurrent uploads."""
            for i in range(uploads_per_thread):
                file_data = sample_files['valid_LOOP_RIGHT_jpeg']
                file_storage = FileStorage(
                    stream=io.BytesIO(file_data['data']),
                    filename=f'concurrent_{thread_id}_{i}.jpg',
                    content_type=file_data['content_type']
                )
                
                result = upload_handler.process_upload(file_storage)
                results_queue.put((thread_id, i, result))
        
        # Start concurrent uploads
        threads = []
        start_time = time.perf_counter()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=upload_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Validate concurrent processing
        assert len(results) == num_threads * uploads_per_thread
        
        successful_uploads = [r for _, _, r in results if r.success]
        assert len(successful_uploads) >= len(results) * 0.9  # At least 90% success
        
        # Performance should be reasonable under concurrency
        avg_time_per_upload = total_time_ms / len(results)
        assert avg_time_per_upload < 2000, f"Concurrent uploads too slow: {avg_time_per_upload:.2f}ms/upload"
    
    def test_memory_management_under_load(self, upload_handler, sample_files):
        """Test memory management under sustained load."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many uploads to test memory management
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        for i in range(20):
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=f'memory_test_{i:03d}.jpg',
                content_type=file_data['content_type']
            )
            
            result = upload_handler.process_upload(file_storage)
            
            # Force garbage collection periodically
            if i % 5 == 0:
                import gc
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (no major leaks)
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.1f}MB increase"
    
    # ==========================================
    # INTEGRATION TESTS
    # ==========================================
    
    def test_integration_with_fingerprint_processor(self, upload_handler, mock_fingerprint_processor, sample_files):
        """Test integration with fingerprint processor."""
        # Configure upload handler with mock processor
        upload_handler.fingerprint_processor = mock_fingerprint_processor
        
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        file_storage = FileStorage(
            stream=io.BytesIO(file_data['data']),
            filename=file_data['filename'],
            content_type=file_data['content_type']
        )
        
        result = upload_handler.process_upload_with_fingerprint_processing(file_storage)
        
        # Validate integration
        assert result.success is True
        assert result.fingerprint_data is not None
        assert 'primary_address' in result.fingerprint_data
        assert 'pattern_class' in result.fingerprint_data
        assert 'confidence_score' in result.fingerprint_data
        
        # Verify processor was called
        mock_fingerprint_processor.process_fingerprint.assert_called_once()
    
    def test_upload_pipeline_end_to_end(self, upload_handler, mock_fingerprint_processor, sample_files):
        """Test complete upload pipeline end-to-end."""
        upload_handler.fingerprint_processor = mock_fingerprint_processor
        
        # Test complete pipeline with multiple file types
        test_files = [
            sample_files['valid_LOOP_RIGHT_jpeg'],
            sample_files['valid_WHORL_png'],
            sample_files['valid_ARCH_jpeg']
        ]
        
        pipeline_results = []
        
        for i, file_data in enumerate(test_files):
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=f'pipeline_{i:02d}_{file_data["filename"]}',
                content_type=file_data['content_type']
            )
            
            result = upload_handler.process_upload_with_fingerprint_processing(file_storage)
            pipeline_results.append(result)
        
        # Validate pipeline results
        assert all(result.success for result in pipeline_results)
        assert all(result.fingerprint_data is not None for result in pipeline_results)
        
        # Validate processing performance
        processing_times = [r.processing_time_ms for r in pipeline_results]
        avg_processing_time = statistics.mean(processing_times)
        assert avg_processing_time < 1000, f"Pipeline too slow: {avg_processing_time:.2f}ms avg"
    
    # ==========================================
    # ERROR HANDLING AND EDGE CASES
    # ==========================================
    
    def test_disk_space_handling(self, upload_handler, sample_files, monkeypatch):
        """Test handling of insufficient disk space."""
        # Mock disk space check to simulate full disk
        def mock_disk_full(*args, **kwargs):
            raise OSError("No space left on device")
        
        monkeypatch.setattr('shutil.copy2', mock_disk_full)
        
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        file_storage = FileStorage(
            stream=io.BytesIO(file_data['data']),
            filename=file_data['filename'],
            content_type=file_data['content_type']
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should handle disk space error gracefully
        assert result.success is False
        assert "space" in result.error_message.lower() or "disk" in result.error_message.lower()
    
    def test_upload_timeout_handling(self, upload_handler, sample_files):
        """Test upload timeout handling."""
        # Configure short timeout
        upload_handler.config['processing_timeout_seconds'] = 0.1
        
        # Mock slow processing
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = lambda x: time.sleep(0.2)  # Simulate slow processing
            
            file_data = sample_files['valid_LOOP_RIGHT_jpeg']
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=file_data['filename'],
                content_type=file_data['content_type']
            )
            
            with patch.object(upload_handler, '_process_image_validation', side_effect=lambda x: (mock_sleep(0.2), True)[1]):
                result = upload_handler.process_upload(file_storage)
                
                # Should handle timeout gracefully
                if not result.success:
                    assert "timeout" in result.error_message.lower()
    
    def test_statistics_tracking(self, upload_handler, sample_files):
        """Test upload statistics tracking."""
        initial_stats = upload_handler.get_upload_statistics()
        assert initial_stats['total_uploads'] == 0
        assert initial_stats['successful_uploads'] == 0
        assert initial_stats['failed_uploads'] == 0
        
        # Process some uploads
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        # Successful upload
        file_storage = FileStorage(
            stream=io.BytesIO(file_data['data']),
            filename='stats_test_1.jpg',
            content_type=file_data['content_type']
        )
        upload_handler.process_upload(file_storage)
        
        # Failed upload (corrupted file)
        corrupted_data = sample_files['corrupted_image']
        file_storage = FileStorage(
            stream=io.BytesIO(corrupted_data['data']),
            filename='stats_test_2.jpg',
            content_type=corrupted_data['content_type']
        )
        upload_handler.process_upload(file_storage)
        
        # Check updated statistics
        final_stats = upload_handler.get_upload_statistics()
        assert final_stats['total_uploads'] == 2
        assert final_stats['successful_uploads'] >= 1
        assert final_stats['failed_uploads'] >= 1
        assert 'average_processing_time_ms' in final_stats
        assert 'total_data_processed_mb' in final_stats


# ==========================================
# PERFORMANCE BENCHMARK TESTS
# ==========================================

class TestUploadPerformanceBenchmarks:
    """Performance benchmark tests for upload handler."""
    
    def test_single_upload_performance_benchmark(self, upload_handler, sample_files):
        """Benchmark single upload performance."""
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        # Run multiple iterations for statistical significance
        performance_samples = []
        
        for i in range(10):
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=f'benchmark_{i:03d}.jpg',
                content_type=file_data['content_type']
            )
            
            start_time = time.perf_counter()
            result = upload_handler.process_upload(file_storage)
            end_time = time.perf_counter()
            
            upload_time_ms = (end_time - start_time) * 1000
            
            if result.success:
                performance_samples.append(upload_time_ms)
        
        # Performance analysis
        avg_time = statistics.mean(performance_samples)
        p95_time = np.percentile(performance_samples, 95)
        p99_time = np.percentile(performance_samples, 99)
        
        # Performance assertions
        assert avg_time < 500, f"Average upload time too slow: {avg_time:.2f}ms"
        assert p95_time < 1000, f"P95 upload time too slow: {p95_time:.2f}ms"
        assert p99_time < 2000, f"P99 upload time too slow: {p99_time:.2f}ms"
        
        print(f"🚀 UPLOAD PERFORMANCE BENCHMARK")
        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   P95 time: {p95_time:.2f}ms")
        print(f"   P99 time: {p99_time:.2f}ms")
    
    def test_throughput_benchmark(self, upload_handler, sample_files):
        """Benchmark upload throughput."""
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        # Process batch of uploads
        num_uploads = 20
        start_time = time.perf_counter()
        
        successful_uploads = 0
        for i in range(num_uploads):
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=f'throughput_{i:03d}.jpg',
                content_type=file_data['content_type']
            )
            
            result = upload_handler.process_upload(file_storage)
            if result.success:
                successful_uploads += 1
        
        end_time = time.perf_counter()
        total_time_seconds = end_time - start_time
        
        # Calculate throughput metrics
        uploads_per_second = successful_uploads / total_time_seconds
        
        # Throughput assertions
        assert uploads_per_second > 5, f"Upload throughput too low: {uploads_per_second:.2f} uploads/sec"
        assert successful_uploads >= num_uploads * 0.95, f"Success rate too low: {successful_uploads}/{num_uploads}"
        
        print(f"📈 UPLOAD THROUGHPUT BENCHMARK")
        print(f"   Throughput: {uploads_per_second:.2f} uploads/second")
        print(f"   Success rate: {successful_uploads}/{num_uploads} ({successful_uploads/num_uploads*100:.1f}%)")
        print(f"   Total time: {total_time_seconds:.2f} seconds")


# ==========================================
# SECURITY PENETRATION TESTS
# ==========================================

class TestUploadSecurityPenetration:
    """Security penetration tests for upload handler."""
    
    def test_path_traversal_attacks(self, upload_handler, sample_files):
        """Test protection against path traversal attacks."""
        file_data = sample_files['valid_LOOP_RIGHT_jpeg']
        
        # Various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/etc/passwd",
            "C:\\windows\\system32\\config",
            "file../../../etc/passwd",
            "file..\\..\\..\\windows\\system32",
            "./.././.././../etc/passwd",
            "....//....//....//etc/passwd"
        ]
        
        for malicious_path in malicious_paths:
            file_storage = FileStorage(
                stream=io.BytesIO(file_data['data']),
                filename=malicious_path + ".jpg",
                content_type=file_data['content_type']
            )
            
            result = upload_handler.process_upload(file_storage)
            
            if result.success:
                # If upload succeeded, verify path is sanitized
                assert not any(dangerous in result.file_path for dangerous in ['../', '..\\', '/etc/', 'C:\\'])
                assert temp_upload_dir in result.file_path
    
    def test_file_bomb_protection(self, upload_handler):
        """Test protection against zip bomb style attacks."""
        # Create highly compressible data that could cause memory issues
        repetitive_data = b"A" * 1000000  # 1MB of repeated data
        
        file_storage = FileStorage(
            stream=io.BytesIO(repetitive_data),
            filename="potential_bomb.jpg",
            content_type="image/jpeg"
        )
        
        result = upload_handler.process_upload(file_storage)
        
        # Should handle suspicious files appropriately
        # Either reject due to size or process safely without memory issues
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500, f"Memory usage too high: {memory_mb:.1f}MB"
    
    def test_executable_file_rejection(self, upload_handler):
        """Test rejection of executable files masquerading as images."""
        # Common executable headers
        executable_data = [
            b"MZ\x90\x00",  # PE executable
            b"\x7fELF",     # ELF executable
            b"#!/bin/bash", # Shell script
            b"PK\x03\x04"   # ZIP file
        ]
        
        for exe_data in executable_data:
            padded_data = exe_data + b"\x00" * 1000  # Pad to reasonable size
            
            file_storage = FileStorage(
                stream=io.BytesIO(padded_data),
                filename="fake_image.jpg",
                content_type="image/jpeg"
            )
            
            result = upload_handler.process_upload(file_storage)
            
            # Should reject executable files
            assert result.success is False
            assert any(keyword in result.error_message.lower() 
                      for keyword in ['invalid', 'format', 'security', 'corrupt'])
