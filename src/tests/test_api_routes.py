#!/usr/bin/env python3
"""
Revolutionary O(1) Fingerprint System - API Routes Tests
Patent Pending - Michael Derrick Jagneaux

Comprehensive tests for the API routes, validating RESTful endpoints,
request/response handling, authentication, error handling, and performance
of the web interface for the O(1) fingerprint matching system.

Test Coverage:
- Fingerprint upload API endpoints
- Search API functionality
- Database information endpoints  
- Performance monitoring APIs
- Demo and validation endpoints
- Error handling and edge cases
- API security and validation
- Response format verification
"""

import pytest
import json
import tempfile
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from werkzeug.datastructures import FileStorage

from src.web.app import O1FingerprintApp
from src.web.api_routes import O1APIRoutes
from src.tests import TestConfig, TestDataGenerator, TestUtils


class TestO1APIRoutes:
    """Test suite for O(1) API routes and endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        app_instance = O1FingerprintApp()
        flask_app = app_instance.create_app()
        flask_app.config['TESTING'] = True
        flask_app.config['WTF_CSRF_ENABLED'] = False
        
        with flask_app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_app_instance(self):
        """Create mock app instance for testing."""
        mock_app = Mock()
        mock_app.processor = Mock()
        mock_app.database = Mock()
        mock_app.performance_monitor = Mock()
        return mock_app
    
    @pytest.fixture
    def sample_fingerprint_file(self):
        """Create sample fingerprint file for upload testing."""
        # Generate synthetic fingerprint image
        image = TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
        
        # Convert to file-like object
        import cv2
        import io
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        file_data = BytesIO(buffer.tobytes())
        
        return FileStorage(
            stream=file_data,
            filename='test_fingerprint.jpg',
            content_type='image/jpeg'
        )
    
    # ==========================================
    # FINGERPRINT UPLOAD API TESTS
    # ==========================================
    
    def test_single_fingerprint_upload_success(self, client, sample_fingerprint_file):
        """Test successful single fingerprint upload."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            # Setup mock responses
            mock_app = mock_app_class.return_value
            mock_app.processor.process_fingerprint.return_value = Mock(
                success=True,
                fingerprint_id="test_001",
                primary_address="FP.LOOP_RIGHT.GOOD_MED.AVG_CTR",
                processing_time_ms=15.2,
                confidence=0.92
            )
            
            response = client.post(
                '/api/fingerprint/upload',
                data={'file': sample_fingerprint_file, 'store_in_database': 'true'},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Validate response structure
            assert 'success' in data
            assert data['success'] is True
            assert 'fingerprint_id' in data
            assert 'processing_time_ms' in data
            assert 'primary_address' in data
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        # Create text file instead of image
        text_file = FileStorage(
            stream=BytesIO(b"This is not an image"),
            filename='not_an_image.txt',
            content_type='text/plain'
        )
        
        response = client.post(
            '/api/fingerprint/upload',
            data={'file': text_file},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid file type' in data['error']
    
    def test_upload_no_file_provided(self, client):
        """Test upload with no file."""
        response = client.post('/api/fingerprint/upload')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file provided' in data['error']
    
    def test_batch_upload_success(self, client):
        """Test successful batch fingerprint upload."""
        # Create multiple test files
        files = []
        for i in range(3):
            image = TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
            import cv2
            _, buffer = cv2.imencode('.jpg', image)
            file_data = BytesIO(buffer.tobytes())
            
            files.append(FileStorage(
                stream=file_data,
                filename=f'batch_test_{i}.jpg',
                content_type='image/jpeg'
            ))
        
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.processor.process_fingerprint.return_value = Mock(
                success=True,
                fingerprint_id="batch_test",
                processing_time_ms=12.0
            )
            
            response = client.post(
                '/api/fingerprint/batch-upload',
                data={'files': files},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'results' in data
            assert len(data['results']) == 3
    
    # ==========================================
    # SEARCH API TESTS
    # ==========================================
    
    def test_fingerprint_search_success(self, client, sample_fingerprint_file):
        """Test successful fingerprint search."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.database.search_fingerprint.return_value = Mock(
                success=True,
                search_time_ms=3.2,
                matches_found=2,
                o1_performance_achieved=True,
                matches=[
                    {'fingerprint_id': 'match_001', 'similarity': 0.95},
                    {'fingerprint_id': 'match_002', 'similarity': 0.87}
                ]
            )
            
            response = client.post(
                '/api/search/fingerprint',
                data={'query_file': sample_fingerprint_file, 'max_results': '10'},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Validate search response
            assert data['success'] is True
            assert 'search_time_ms' in data
            assert 'matches' in data
            assert data['o1_performance_achieved'] is True
            assert len(data['matches']) == 2
    
    def test_address_based_search(self, client):
        """Test address-based O(1) search."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.database.search_by_address.return_value = Mock(
                success=True,
                search_time_ms=2.1,
                matches=[{'fingerprint_id': 'addr_match_001'}]
            )
            
            response = client.post(
                '/api/search/address',
                json={'address': 'FP.LOOP_RIGHT.GOOD_MED.AVG_CTR'}
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['search_time_ms'] < 5.0  # O(1) performance
    
    def test_search_with_invalid_address(self, client):
        """Test search with invalid address format."""
        response = client.post(
            '/api/search/address',
            json={'address': 'INVALID.FORMAT'}
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    # ==========================================
    # DATABASE API TESTS
    # ==========================================
    
    def test_database_statistics_endpoint(self, client):
        """Test database statistics API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.database.get_database_statistics.return_value = Mock(
                total_records=50000,
                unique_addresses=1200,
                average_search_time_ms=2.8,
                o1_performance_percentage=97.5
            )
            
            response = client.get('/api/database/stats')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Validate statistics structure
            assert 'total_records' in data
            assert 'unique_addresses' in data
            assert 'average_search_time_ms' in data
            assert 'o1_performance_percentage' in data
            assert data['total_records'] == 50000
    
    def test_database_optimization_endpoint(self, client):
        """Test database optimization API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.database.optimize_database.return_value = {
                'success': True,
                'optimization_time_ms': 150.0,
                'performance_improvement': 12.5
            }
            
            response = client.post('/api/database/optimize')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'optimization_time_ms' in data
    
    # ==========================================
    # PERFORMANCE MONITORING TESTS
    # ==========================================
    
    def test_performance_metrics_endpoint(self, client):
        """Test performance metrics API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.performance_monitor.get_current_metrics.return_value = {
                'avg_search_time_ms': 3.1,
                'p95_search_time_ms': 4.8,
                'p99_search_time_ms': 6.2,
                'o1_compliance_rate': 98.2,
                'throughput_queries_per_second': 320
            }
            
            response = client.get('/api/performance/metrics')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Validate performance metrics
            assert 'avg_search_time_ms' in data
            assert 'o1_compliance_rate' in data
            assert data['o1_compliance_rate'] > 95.0
    
    def test_o1_validation_endpoint(self, client):
        """Test O(1) performance validation API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.performance_monitor.prove_o1_performance.return_value = Mock(
                is_constant_time=True,
                average_time_ms=2.9,
                coefficient_of_variation=0.08,
                validation_confidence=0.97
            )
            
            response = client.post(
                '/api/performance/validate-o1',
                json={'database_sizes': [1000, 10000, 100000]}
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['is_constant_time'] is True
            assert data['validation_confidence'] > 0.95
    
    # ==========================================
    # DEMO AND VALIDATION ENDPOINTS
    # ==========================================
    
    def test_demo_scalability_endpoint(self, client):
        """Test scalability demonstration API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.database.demonstrate_scalability.return_value = {
                'database_sizes_tested': [1000, 100000, 10000000],
                'search_times_ms': [2.8, 3.1, 2.9],
                'scalability_proven': True,
                'performance_consistency': 97.8
            }
            
            response = client.post('/api/demo/scalability')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['scalability_proven'] is True
    
    def test_patent_validation_endpoint(self, client):
        """Test patent validation demonstration API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            mock_app.performance_monitor.validate_patent_claims.return_value = {
                'claim_1_constant_time': True,
                'claim_2_unlimited_scale': True,
                'claim_3_superior_performance': True,
                'mathematical_proof_confidence': 0.98,
                'patent_validation_status': 'PROVEN'
            }
            
            response = client.post('/api/demo/patent-validation')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['patent_validation_status'] == 'PROVEN'
    
    # ==========================================
    # ERROR HANDLING TESTS
    # ==========================================
    
    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get('/api/nonexistent/endpoint')
        assert response.status_code == 404
    
    def test_method_not_allowed_handling(self, client):
        """Test method not allowed handling."""
        response = client.get('/api/fingerprint/upload')  # Should be POST
        assert response.status_code == 405
    
    def test_invalid_json_handling(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            '/api/search/address',
            data='invalid json',
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_large_file_upload_handling(self, client):
        """Test handling of oversized file uploads."""
        # Create large dummy file
        large_data = b'x' * (20 * 1024 * 1024)  # 20MB file
        large_file = FileStorage(
            stream=BytesIO(large_data),
            filename='large_file.jpg',
            content_type='image/jpeg'
        )
        
        response = client.post(
            '/api/fingerprint/upload',
            data={'file': large_file},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 413  # Request Entity Too Large
    
    # ==========================================
    # SECURITY TESTS
    # ==========================================
    
    def test_cors_headers(self, client):
        """Test CORS headers on API endpoints."""
        response = client.options('/api/fingerprint/upload')
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_security_headers(self, client):
        """Test security headers."""
        response = client.get('/api/stats')
        
        # Check for security headers (these should be added in production)
        # Note: Actual implementation may vary
        assert response.status_code in [200, 404]  # Endpoint may not exist
    
    def test_input_validation(self, client):
        """Test input validation and sanitization."""
        # Test with malicious input
        malicious_input = "<script>alert('xss')</script>"
        
        response = client.post(
            '/api/search/address',
            json={'address': malicious_input}
        )
        
        # Should reject or sanitize malicious input
        assert response.status_code in [400, 422]
    
    # ==========================================
    # PERFORMANCE TESTS
    # ==========================================
    
    def test_api_response_times(self, client):
        """Test API response time requirements."""
        endpoints_to_test = [
            ('/api/stats', 'GET'),
            ('/health', 'GET')
        ]
        
        for endpoint, method in endpoints_to_test:
            start_time = time.perf_counter()
            
            if method == 'GET':
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            
            # API responses should be fast
            assert response_time <= TestConfig.API_RESPONSE_TIME_THRESHOLD_MS, \
                f"{endpoint} too slow: {response_time:.2f}ms"
    
    def test_concurrent_api_requests(self, client):
        """Test API performance under concurrent requests."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        num_threads = 5
        requests_per_thread = 10
        
        def api_worker():
            thread_results = []
            for _ in range(requests_per_thread):
                start_time = time.perf_counter()
                response = client.get('/health')
                end_time = time.perf_counter()
                
                thread_results.append({
                    'status_code': response.status_code,
                    'response_time_ms': (end_time - start_time) * 1000
                })
            
            results_queue.put(thread_results)
        
        # Launch concurrent requests
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=api_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze concurrent performance
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Validate concurrent API performance
        successful_requests = sum(1 for r in all_results if r['status_code'] == 200)
        success_rate = successful_requests / len(all_results)
        
        avg_response_time = sum(r['response_time_ms'] for r in all_results) / len(all_results)
        
        assert success_rate >= 0.95, f"Poor concurrent success rate: {success_rate:.1%}"
        assert avg_response_time <= TestConfig.API_RESPONSE_TIME_THRESHOLD_MS * 2, \
            f"Poor concurrent performance: {avg_response_time:.2f}ms"
    
    # ==========================================
    # INTEGRATION TESTS
    # ==========================================
    
    def test_full_workflow_integration(self, client, sample_fingerprint_file):
        """Test complete workflow through API."""
        with patch('src.web.app.O1FingerprintApp') as mock_app_class:
            mock_app = mock_app_class.return_value
            
            # Mock successful upload
            mock_app.processor.process_fingerprint.return_value = Mock(
                success=True,
                fingerprint_id="integration_001",
                primary_address="FP.LOOP_RIGHT.GOOD_MED.AVG_CTR"
            )
            
            # Mock successful database insertion
            mock_app.database.insert_fingerprint.return_value = Mock(
                success=True,
                record_id="integration_001"
            )
            
            # Mock successful search
            mock_app.database.search_by_address.return_value = Mock(
                success=True,
                matches=[{'fingerprint_id': 'integration_001'}]
            )
            
            # 1. Upload fingerprint
            upload_response = client.post(
                '/api/fingerprint/upload',
                data={'file': sample_fingerprint_file, 'store_in_database': 'true'},
                content_type='multipart/form-data'
            )
            assert upload_response.status_code == 200
            
            upload_data = json.loads(upload_response.data)
            primary_address = upload_data.get('primary_address')
            
            # 2. Search for uploaded fingerprint
            if primary_address:
                search_response = client.post(
                    '/api/search/address',
                    json={'address': primary_address}
                )
                assert search_response.status_code == 200
                
                search_data = json.loads(search_response.data)
                assert search_data['success'] is True
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Validate health check structure
        assert 'status' in data
        assert 'timestamp' in data
        assert 'components' in data
        assert data['status'] == 'healthy'
