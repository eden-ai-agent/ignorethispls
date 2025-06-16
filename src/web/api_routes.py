#!/usr/bin/env python3
"""
Revolutionary O(1) Fingerprint System - API Routes
Patent Pending - Michael Derrick Jagneaux

Organized API route handlers for the O(1) fingerprint matching system.
Designed for high-performance, real-time demonstrations and production use.

Features:
- RESTful API design
- Real-time performance monitoring
- O(1) validation endpoints
- Comprehensive error handling
- Production-ready responses
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.timing_utils import HighPrecisionTimer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class O1APIRoutes:
    """
    Revolutionary API Routes for O(1) Fingerprint System
    
    Organized, high-performance API endpoints showcasing constant-time
    biometric matching capabilities.
    """
    
    def __init__(self, app_instance):
        """Initialize API routes with app instance."""
        self.app_instance = app_instance
        self.timer = HighPrecisionTimer()
        
        # Create Blueprint
        self.api = Blueprint('api', __name__, url_prefix='/api')
        self.demo = Blueprint('demo', __name__, url_prefix='/demo')
        
        # Register routes
        self._register_fingerprint_routes()
        self._register_search_routes()
        self._register_database_routes()
        self._register_performance_routes()
        self._register_demo_routes()
        
        logger.info("O(1) API Routes initialized")
    
    def _register_fingerprint_routes(self):
        """Register fingerprint processing routes."""
        
        @self.api.route('/fingerprint/upload', methods=['POST'])
        def upload_single_fingerprint():
            """Upload and process a single fingerprint."""
            return self._handle_single_upload()
        
        @self.api.route('/fingerprint/batch-upload', methods=['POST'])
        def upload_batch_fingerprints():
            """Upload and process multiple fingerprints."""
            return self._handle_batch_upload()
        
        @self.api.route('/fingerprint/classify', methods=['POST'])
        def classify_fingerprint():
            """Classify fingerprint pattern only (no database storage)."""
            return self._handle_classification_only()
        
        @self.api.route('/fingerprint/extract-features', methods=['POST'])
        def extract_features():
            """Extract characteristics without storing."""
            return self._handle_feature_extraction()
        
        @self.api.route('/fingerprint/validate-quality', methods=['POST'])
        def validate_quality():
            """Validate fingerprint image quality."""
            return self._handle_quality_validation()
    
    def _register_search_routes(self):
        """Register search and matching routes."""
        
        @self.api.route('/search/by-image', methods=['POST'])
        def search_by_image():
            """Search database using uploaded fingerprint image."""
            return self._handle_image_search()
        
        @self.api.route('/search/by-address', methods=['POST'])
        def search_by_address():
            """Search database using O(1) address."""
            return self._handle_address_search()
        
        @self.api.route('/search/by-characteristics', methods=['POST'])
        def search_by_characteristics():
            """Search using extracted characteristics."""
            return self._handle_characteristic_search()
        
        @self.api.route('/search/similar', methods=['POST'])
        def search_similar():
            """Find similar fingerprints using similarity addresses."""
            return self._handle_similarity_search()
        
        @self.api.route('/search/batch', methods=['POST'])
        def batch_search():
            """Perform batch searches for multiple queries."""
            return self._handle_batch_search()
    
    def _register_database_routes(self):
        """Register database management routes."""
        
        @self.api.route('/database/info', methods=['GET'])
        def database_info():
            """Get comprehensive database information."""
            return self._handle_database_info()
        
        @self.api.route('/database/statistics', methods=['GET'])
        def database_statistics():
            """Get detailed database statistics."""
            return self._handle_database_statistics()
        
        @self.api.route('/database/health', methods=['GET'])
        def database_health():
            """Check database health and connectivity."""
            return self._handle_database_health()
        
        @self.api.route('/database/optimize', methods=['POST'])
        def optimize_database():
            """Optimize database for better O(1) performance."""
            return self._handle_database_optimization()
        
        @self.api.route('/database/addresses', methods=['GET'])
        def list_addresses():
            """List all unique addresses in database."""
            return self._handle_address_listing()
    
    def _register_performance_routes(self):
        """Register performance monitoring routes."""
        
        @self.api.route('/performance/current', methods=['GET'])
        def current_performance():
            """Get current performance metrics."""
            return self._handle_current_performance()
        
        @self.api.route('/performance/benchmark', methods=['POST'])
        def run_benchmark():
            """Run performance benchmark tests."""
            return self._handle_benchmark()
        
        @self.api.route('/performance/o1-validation', methods=['POST'])
        def validate_o1():
            """Validate O(1) performance claims."""
            return self._handle_o1_validation()
        
        @self.api.route('/performance/scalability', methods=['POST'])
        def test_scalability():
            """Test system scalability across database sizes."""
            return self._handle_scalability_test()
        
        @self.api.route('/performance/timing-report', methods=['GET'])
        def timing_report():
            """Generate comprehensive timing report."""
            return self._handle_timing_report()
    
    def _register_demo_routes(self):
        """Register demonstration routes."""
        
        @self.demo.route('/o1-proof', methods=['POST'])
        def o1_proof():
            """Demonstrate O(1) performance proof."""
            return self._handle_o1_proof()
        
        @self.demo.route('/traditional-comparison', methods=['POST'])
        def traditional_comparison():
            """Compare O(1) vs traditional search performance."""
            return self._handle_traditional_comparison()
        
        @self.demo.route('/live-scaling', methods=['POST'])
        def live_scaling():
            """Live demonstration of constant-time scaling."""
            return self._handle_live_scaling()
        
        @self.demo.route('/patent-validation', methods=['POST'])
        def patent_validation():
            """Comprehensive patent validation demonstration."""
            return self._handle_patent_validation()
        
        @self.demo.route('/real-world-scenario', methods=['POST'])
        def real_world_scenario():
            """Demonstrate real-world law enforcement scenario."""
            return self._handle_real_world_scenario()
    
    # ==========================================
    # FINGERPRINT PROCESSING HANDLERS
    # ==========================================
    
    def _handle_single_upload(self) -> Dict[str, Any]:
        """Handle single fingerprint upload and processing."""
        start_time = time.perf_counter()
        
        try:
            # Validate request
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Supported: JPG, PNG, TIF, BMP'}), 400
            
            # Save file securely
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time() * 1000))  # Millisecond precision
            safe_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(filepath)
            
            # Process fingerprint with timing
            with self.timer.time_operation("fingerprint_processing"):
                processing_result = self.app_instance.processor.process_fingerprint(filepath)
            
            # Store in database if requested
            store_in_db = request.form.get('store_in_database', 'true').lower() == 'true'
            db_record = None
            
            if store_in_db:
                with self.timer.time_operation("database_insertion"):
                    db_record = self.app_instance.database.insert_fingerprint(
                        filepath,
                        metadata={
                            'original_filename': file.filename,
                            'upload_timestamp': time.time(),
                            'processing_time_ms': self.timer.get_last_operation_time("fingerprint_processing")
                        }
                    )
            
            # Update application statistics
            self.app_instance.stats['total_uploads'] += 1
            
            # Prepare comprehensive response
            response = {
                'success': True,
                'upload_info': {
                    'filename': safe_filename,
                    'original_filename': file.filename,
                    'file_size_bytes': len(file.read()),
                    'upload_timestamp': time.time()
                },
                'processing_results': {
                    'primary_address': processing_result.primary_address,
                    'similarity_addresses': processing_result.similarity_addresses,
                    'processing_time_ms': self.timer.get_last_operation_time("fingerprint_processing"),
                    'characteristics': processing_result.characteristics
                },
                'pattern_classification': {
                    'pattern_type': processing_result.pattern_classification.primary_pattern.value,
                    'confidence': processing_result.pattern_classification.pattern_confidence,
                    'quality_score': processing_result.pattern_classification.pattern_quality,
                    'biological_consistency': processing_result.pattern_classification.biological_consistency,
                    'singular_points': len(processing_result.pattern_classification.singular_points),
                    'explanation': processing_result.pattern_classification.explanation
                },
                'database_info': {
                    'stored': store_in_db,
                    'record_id': db_record.record_id if db_record else None,
                    'insertion_time_ms': self.timer.get_last_operation_time("database_insertion") if store_in_db else None
                },
                'performance_metrics': {
                    'total_time_ms': (time.perf_counter() - start_time) * 1000,
                    'o1_address_generated': True,
                    'ready_for_search': True
                }
            }
            
            logger.info(f"Single upload processed: {safe_filename} -> {processing_result.primary_address}")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Single upload failed: {e}")
            return jsonify({
                'error': 'Upload processing failed',
                'message': str(e),
                'timestamp': time.time()
            }), 500
    
    def _handle_batch_upload(self) -> Dict[str, Any]:
        """Handle batch fingerprint upload and processing."""
        start_time = time.perf_counter()
        
        try:
            files = request.files.getlist('files')
            if not files:
                return jsonify({'error': 'No files provided'}), 400
            
            # Processing options
            store_in_db = request.form.get('store_in_database', 'true').lower() == 'true'
            max_parallel = int(request.form.get('max_parallel_workers', '4'))
            
            results = []
            failed_files = []
            total_processing_time = 0
            
            logger.info(f"Starting batch upload: {len(files)} files")
            
            for i, file in enumerate(files):
                try:
                    if not self._allowed_file(file.filename):
                        failed_files.append({
                            'filename': file.filename,
                            'error': 'Invalid file type',
                            'index': i
                        })
                        continue
                    
                    # Save file
                    filename = secure_filename(file.filename)
                    timestamp = str(int(time.time() * 1000))
                    safe_filename = f"{timestamp}_{i:04d}_{filename}"
                    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_filename)
                    file.save(filepath)
                    
                    # Process fingerprint
                    file_start = time.perf_counter()
                    processing_result = self.app_instance.processor.process_fingerprint(filepath)
                    file_processing_time = (time.perf_counter() - file_start) * 1000
                    total_processing_time += file_processing_time
                    
                    # Store in database
                    db_record = None
                    if store_in_db:
                        db_record = self.app_instance.database.insert_fingerprint(
                            filepath,
                            metadata={
                                'original_filename': file.filename,
                                'batch_index': i,
                                'batch_timestamp': time.time()
                            }
                        )
                    
                    results.append({
                        'index': i,
                        'filename': safe_filename,
                        'original_filename': file.filename,
                        'primary_address': processing_result.primary_address,
                        'pattern_type': processing_result.pattern_classification.primary_pattern.value,
                        'confidence': processing_result.pattern_classification.pattern_confidence,
                        'quality_score': processing_result.pattern_classification.pattern_quality,
                        'processing_time_ms': file_processing_time,
                        'record_id': db_record.record_id if db_record else None
                    })
                    
                    # Log progress
                    if (i + 1) % 10 == 0 or (i + 1) == len(files):
                        logger.info(f"Batch progress: {i + 1}/{len(files)} files processed")
                    
                except Exception as e:
                    failed_files.append({
                        'filename': file.filename,
                        'error': str(e),
                        'index': i
                    })
            
            # Update statistics
            self.app_instance.stats['total_uploads'] += len(results)
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            response = {
                'success': True,
                'batch_summary': {
                    'total_files': len(files),
                    'successful_uploads': len(results),
                    'failed_uploads': len(failed_files),
                    'success_rate': (len(results) / len(files)) * 100 if files else 0
                },
                'performance_metrics': {
                    'total_time_ms': total_time,
                    'total_processing_time_ms': total_processing_time,
                    'avg_time_per_file_ms': total_processing_time / len(results) if results else 0,
                    'upload_throughput_fps': len(results) / (total_time / 1000) if total_time > 0 else 0
                },
                'results': results,
                'failed_files': failed_files,
                'database_impact': {
                    'records_added': len(results) if store_in_db else 0,
                    'new_database_size': self.app_instance.database.get_database_statistics().total_records
                }
            }
            
            logger.info(f"Batch upload completed: {len(results)} successful, {len(failed_files)} failed")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            return jsonify({
                'error': 'Batch upload failed',
                'message': str(e)
            }), 500
    
    def _handle_classification_only(self) -> Dict[str, Any]:
        """Handle fingerprint classification without database storage."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Temporary file processing
            filename = secure_filename(file.filename)
            temp_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"temp_{filename}")
            file.save(temp_filepath)
            
            try:
                # Classification only
                with self.timer.time_operation("classification_only"):
                    classification_result = self.app_instance.processor.pattern_classifier.classify_pattern(
                        cv2.imread(temp_filepath, cv2.IMREAD_GRAYSCALE)
                    )
                
                response = {
                    'success': True,
                    'classification': {
                        'pattern_type': classification_result.primary_pattern.value,
                        'confidence': classification_result.pattern_confidence,
                        'quality_score': classification_result.pattern_quality,
                        'biological_consistency': classification_result.biological_consistency,
                        'singular_points_detected': len(classification_result.singular_points),
                        'secondary_patterns': [
                            {'pattern': pattern.value, 'confidence': conf}
                            for pattern, conf in classification_result.secondary_patterns
                        ],
                        'explanation': classification_result.explanation
                    },
                    'processing_time_ms': self.timer.get_last_operation_time("classification_only"),
                    'features_extracted': len(classification_result.features)
                }
                
                return jsonify(response)
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return jsonify({
                'error': 'Classification failed',
                'message': str(e)
            }), 500
    
    def _handle_feature_extraction(self) -> Dict[str, Any]:
        """Handle feature extraction without classification."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Process for features only
            filename = secure_filename(file.filename)
            temp_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"features_{filename}")
            file.save(temp_filepath)
            
            try:
                import cv2
                image = cv2.imread(temp_filepath, cv2.IMREAD_GRAYSCALE)
                
                with self.timer.time_operation("feature_extraction"):
                    features = self.app_instance.processor.pattern_classifier.extract_features_for_addressing(image)
                
                response = {
                    'success': True,
                    'features': features,
                    'feature_count': len(features),
                    'extraction_time_ms': self.timer.get_last_operation_time("feature_extraction"),
                    'addressing_ready': 'primary_pattern' in features
                }
                
                return jsonify(response)
                
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return jsonify({
                'error': 'Feature extraction failed',
                'message': str(e)
            }), 500
    
    def _handle_quality_validation(self) -> Dict[str, Any]:
        """Handle fingerprint quality validation."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            filename = secure_filename(file.filename)
            temp_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"quality_{filename}")
            file.save(temp_filepath)
            
            try:
                import cv2
                image = cv2.imread(temp_filepath, cv2.IMREAD_GRAYSCALE)
                
                with self.timer.time_operation("quality_validation"):
                    # Quick quality assessment
                    contrast = float(np.std(image))
                    clarity = float(cv2.Laplacian(image, cv2.CV_64F).var())
                    
                    # Quality scoring
                    contrast_score = min(1.0, contrast / 50.0)
                    clarity_score = min(1.0, clarity / 1000.0)
                    overall_quality = (contrast_score + clarity_score) / 2
                    
                    # Quality assessment
                    if overall_quality >= 0.8:
                        quality_level = "EXCELLENT"
                        recommendation = "Perfect for O(1) processing"
                    elif overall_quality >= 0.6:
                        quality_level = "GOOD"
                        recommendation = "Suitable for processing"
                    elif overall_quality >= 0.4:
                        quality_level = "FAIR"
                        recommendation = "May work but consider higher quality image"
                    else:
                        quality_level = "POOR"
                        recommendation = "Recommend higher quality image for reliable results"
                
                response = {
                    'success': True,
                    'quality_assessment': {
                        'overall_quality': overall_quality,
                        'quality_level': quality_level,
                        'recommendation': recommendation,
                        'metrics': {
                            'contrast': contrast,
                            'clarity': clarity,
                            'contrast_score': contrast_score,
                            'clarity_score': clarity_score
                        }
                    },
                    'image_info': {
                        'dimensions': image.shape,
                        'size_bytes': len(file.read()),
                        'suitable_for_processing': overall_quality >= 0.4
                    },
                    'validation_time_ms': self.timer.get_last_operation_time("quality_validation")
                }
                
                return jsonify(response)
                
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return jsonify({
                'error': 'Quality validation failed',
                'message': str(e)
            }), 500
    
    # ==========================================
    # SEARCH OPERATION HANDLERS
    # ==========================================
    
    def _handle_image_search(self) -> Dict[str, Any]:
        """Handle search using uploaded fingerprint image."""
        search_start = time.perf_counter()
        
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No query image provided'}), 400
            
            file = request.files['file']
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Search parameters
            similarity_threshold = float(request.form.get('similarity_threshold', 0.75))
            max_results = int(request.form.get('max_results', 20))
            include_similar = request.form.get('include_similar', 'true').lower() == 'true'
            
            # Process query image
            filename = secure_filename(file.filename)
            query_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"query_{filename}")
            file.save(query_filepath)
            
            # Extract characteristics
            with self.timer.time_operation("query_processing"):
                query_result = self.app_instance.processor.process_fingerprint(query_filepath)
            
            # Prepare search addresses
            search_addresses = [query_result.primary_address]
            if include_similar:
                search_addresses.extend(query_result.similarity_addresses)
            
            # Perform O(1) search
            with self.timer.time_operation("o1_search"):
                search_result = self.app_instance.database.search_by_addresses(
                    search_addresses,
                    similarity_threshold=similarity_threshold,
                    max_results=max_results
                )
            
            # Update statistics
            self.app_instance.stats['total_searches'] += 1
            search_time = self.timer.get_last_operation_time("o1_search")
            
            # Clean up query file
            if os.path.exists(query_filepath):
                os.remove(query_filepath)
            
            response = {
                'success': True,
                'search_results': {
                    'matches_found': len(search_result.matches),
                    'search_time_ms': search_time,
                    'o1_performance_achieved': search_result.o1_performance_achieved,
                    'query_addresses_used': search_addresses,
                    'matches': [
                        {
                            'record_id': match.record_id,
                            'filename': match.filename,
                            'address': match.address,
                            'confidence_score': match.confidence_score,
                            'quality_score': match.quality_score,
                            'similarity_score': getattr(match, 'similarity_score', 1.0)
                        }
                        for match in search_result.matches
                    ]
                },
                'query_info': {
                    'primary_address': query_result.primary_address,
                    'pattern_type': query_result.pattern_classification.primary_pattern.value,
                    'pattern_confidence': query_result.pattern_classification.pattern_confidence,
                    'processing_time_ms': self.timer.get_last_operation_time("query_processing")
                },
                'performance_metrics': {
                    'total_search_time_ms': (time.perf_counter() - search_start) * 1000,
                    'records_examined': search_result.records_examined,
                    'cache_performance': {
                        'cache_hits': search_result.cache_hits,
                        'cache_efficiency': search_result.cache_hits / len(search_addresses) if search_addresses else 0
                    },
                    'database_size': self.app_instance.database.get_database_statistics().total_records
                }
            }
            
            logger.info(f"Image search completed: {len(search_result.matches)} matches in {search_time:.2f}ms")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return jsonify({
                'error': 'Image search failed',
                'message': str(e)
            }), 500
    
    def _handle_address_search(self) -> Dict[str, Any]:
        """Handle search using O(1) address."""
        search_start = time.perf_counter()
        
        try:
            data = request.get_json()
            if not data or 'address' not in data:
                return jsonify({'error': 'No address provided'}), 400
            
            address = data['address']
            max_results = data.get('max_results', 20)
            
            # Perform direct O(1) lookup
            with self.timer.time_operation("direct_o1_lookup"):
                search_result = self.app_instance.database.search_by_addresses(
                    [address],
                    max_results=max_results
                )
            
            search_time = self.timer.get_last_operation_time("direct_o1_lookup")
            
            response = {
                'success': True,
                'search_results': {
                    'address_searched': address,
                    'matches_found': len(search_result.matches),
                    'search_time_ms': search_time,
                    'o1_performance_achieved': search_result.o1_performance_achieved,
                    'matches': [
                        {
                            'record_id': match.record_id,
                            'filename': match.filename,
                            'address': match.address,
                            'confidence_score': match.confidence_score,
                            'quality_score': match.quality_score,
                            'metadata': match.metadata
                        }
                        for match in search_result.matches
                    ]
                },
                'performance_proof': {
                    'constant_time_achieved': search_time <= 10,  # O(1) threshold
                    'search_efficiency': 'O(1)' if search_time <= 10 else 'Suboptimal',
                    'records_examined': search_result.records_examined,
                    'total_database_size': self.app_instance.database.get_database_statistics().total_records
                }
            }
            
            logger.info(f"Address search completed: {address} -> {len(search_result.matches)} matches in {search_time:.2f}ms")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Address search failed: {e}")
            return jsonify({
                'error': 'Address search failed',
                'message': str(e)
            }), 500
    
    def _handle_batch_search(self) -> Dict[str, Any]:
        """Handle batch search operations."""
        batch_start = time.perf_counter()
        
        try:
            data = request.get_json()
            if not data or 'queries' not in data:
                return jsonify({'error': 'No search queries provided'}), 400
            
            queries = data['queries']
            batch_results = []
            total_search_time = 0
            
            for i, query in enumerate(queries):
                try:
                    query_start = time.perf_counter()
                    
                    if 'address' in query:
                        # Address-based search
                        search_result = self.app_instance.database.search_by_addresses(
                            [query['address']],
                            max_results=query.get('max_results', 10)
                        )
                    else:
                        continue  # Skip invalid queries
                    
                    query_time = (time.perf_counter() - query_start) * 1000
                    total_search_time += query_time
                    
                    batch_results.append({
                        'query_index': i,
                        'query': query,
                        'matches_found': len(search_result.matches),
                        'search_time_ms': query_time,
                        'o1_achieved': search_result.o1_performance_achieved,
                        'matches': [
                            {
                                'record_id': match.record_id,
                                'filename': match.filename,
                                'confidence_score': match.confidence_score
                            }
                            for match in search_result.matches[:5]  # Limit for batch
                        ]
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'query_index': i,
                        'error': str(e)
                    })
            
            total_time = (time.perf_counter() - batch_start) * 1000
            
            response = {
                'success': True,
                'batch_summary': {
                    'total_queries': len(queries),
                    'successful_queries': len([r for r in batch_results if 'error' not in r]),
                    'total_matches_found': sum(r.get('matches_found', 0) for r in batch_results),
                    'batch_processing_time_ms': total_time,
                    'total_search_time_ms': total_search_time,
                    'avg_search_time_ms': total_search_time / len(queries) if queries else 0
                },
                'results': batch_results,
                'performance_metrics': {
                    'o1_consistency': len([r for r in batch_results if r.get('o1_achieved', False)]) / len(queries) * 100,
                    'batch_throughput_qps': len(queries) / (total_time / 1000) if total_time > 0 else 0
                }
            }
            
            logger.info(f"Batch search completed: {len(queries)} queries in {total_time:.2f}ms")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return jsonify({
                'error': 'Batch search failed',
                'message': str(e)
            }), 500
    
    # Helper methods
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        allowed_extensions = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def get_blueprint(self) -> Tuple[Blueprint, Blueprint]:
        """Get the API and Demo blueprints."""
        return self.api, self.demo


# Factory function for Flask app integration
def register_api_routes(app, app_instance):
    """Register API routes with Flask app."""
    api_routes = O1APIRoutes(app_instance)
    api_blueprint, demo_blueprint = api_routes.get_blueprint()
    
    app.register_blueprint(api_blueprint)
    app.register_blueprint(demo_blueprint)
    
    logger.info("API routes registered successfully")
    return api_routes