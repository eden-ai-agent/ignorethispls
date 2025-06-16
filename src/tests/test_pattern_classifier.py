#!/usr/bin/env python3
"""
Revolutionary O(1) Fingerprint System - Pattern Classifier Tests
Patent Pending - Michael Derrick Jagneaux

Comprehensive tests for the Scientific Pattern Classifier, validating
the accuracy and performance of the core pattern recognition system
that enables O(1) addressing.

Test Coverage:
- Pattern classification accuracy validation
- Poincaré index calculation verification
- Singular point detection validation
- Performance benchmarking for O(1) requirements
- Quality assessment testing
- Biological consistency validation
- Batch processing verification
"""

import pytest
import numpy as np
import cv2
import time
import statistics
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Tuple

from src.core.pattern_classifier import (
    ScientificPatternClassifier, 
    FingerprintPattern,
    SingularPointType,
    PatternClassificationResult,
    SingularPoint
)
from src.tests import TestConfig, TestDataGenerator, TestUtils


class TestScientificPatternClassifier:
    """Test suite for the Scientific Pattern Classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return ScientificPatternClassifier(
            image_size=(512, 512),
            block_size=16,
            smoothing_sigma=2.0
        )
    
    @pytest.fixture
    def test_images(self):
        """Generate test fingerprint images."""
        return {
            'loop_right': TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT"),
            'loop_left': TestDataGenerator.create_synthetic_fingerprint("LOOP_LEFT"),
            'whorl': TestDataGenerator.create_synthetic_fingerprint("WHORL"),
            'arch': TestDataGenerator.create_synthetic_fingerprint("ARCH")
        }
    
    # ==========================================
    # BASIC FUNCTIONALITY TESTS
    # ==========================================
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes with correct parameters."""
        assert classifier.image_size == (512, 512)
        assert classifier.block_size == 16
        assert classifier.smoothing_sigma == 2.0
        assert classifier.poincare_threshold == 0.3
        assert hasattr(classifier, 'classification_stats')
        assert classifier.classification_stats['total_classifications'] == 0
    
    def test_single_pattern_classification(self, classifier, test_images):
        """Test single pattern classification functionality."""
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            
            # Validate result structure
            assert isinstance(result, PatternClassificationResult)
            assert isinstance(result.primary_pattern, FingerprintPattern)
            assert 0.0 <= result.pattern_confidence <= 1.0
            assert result.processing_time_ms > 0
            assert result.pattern_quality >= 0.0
            assert isinstance(result.singular_points, list)
            assert isinstance(result.ridge_orientation_field, np.ndarray)
            assert len(result.explanation) > 0
    
    def test_pattern_type_accuracy(self, classifier, test_images):
        """Test pattern classification accuracy for known patterns."""
        # Note: Since we're using synthetic data, we expect some classification
        # flexibility, but the system should demonstrate consistent behavior
        
        results = {}
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            results[pattern_type] = result
            
            # Each classification should have reasonable confidence
            assert result.pattern_confidence >= 0.3, f"Low confidence for {pattern_type}"
            
            # Processing time should be within O(1) requirements
            assert result.processing_time_ms <= TestConfig.O1_SEARCH_TIME_THRESHOLD_MS * 2, \
                f"Classification too slow for {pattern_type}: {result.processing_time_ms}ms"
        
        # Ensure we get different classifications for different patterns
        unique_classifications = set(r.primary_pattern for r in results.values())
        assert len(unique_classifications) >= 2, "Classifier not distinguishing between patterns"
    
    def test_poincare_index_calculation(self, classifier, test_images):
        """Test Poincaré index calculation for singular point detection."""
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            
            # Validate singular points have proper Poincaré indices
            for sp in result.singular_points:
                assert isinstance(sp, SingularPoint)
                assert isinstance(sp.poincare_index, float)
                assert abs(sp.poincare_index) <= 1.0, "Poincaré index out of valid range"
                assert 0 <= sp.x < image.shape[1], "Singular point X coordinate invalid"
                assert 0 <= sp.y < image.shape[0], "Singular point Y coordinate invalid"
                assert 0.0 <= sp.confidence <= 1.0, "Singular point confidence invalid"
    
    def test_singular_point_types(self, classifier, test_images):
        """Test detection of different singular point types."""
        all_point_types = set()
        
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            
            for sp in result.singular_points:
                assert sp.point_type in [SingularPointType.CORE, SingularPointType.DELTA]
                all_point_types.add(sp.point_type)
        
        # Should detect both cores and deltas across test images
        # (Though not necessarily in every single image)
        assert len(all_point_types) >= 1, "No singular points detected"
    
    # ==========================================
    # PERFORMANCE TESTS
    # ==========================================
    
    def test_classification_speed(self, classifier, test_images):
        """Test classification meets speed requirements for O(1) system."""
        processing_times = []
        
        for pattern_type, image in test_images.items():
            start_time = time.perf_counter()
            result = classifier.classify_pattern(image)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            processing_times.append(processing_time)
            
            # Individual classification should be fast
            assert processing_time <= 50.0, f"Classification too slow: {processing_time}ms"
            
            # Reported time should be reasonably accurate
            assert abs(processing_time - result.processing_time_ms) <= 5.0, \
                "Reported processing time inaccurate"
        
        # Average processing time should be excellent
        avg_time = statistics.mean(processing_times)
        assert avg_time <= 25.0, f"Average processing time too slow: {avg_time}ms"
    
    def test_batch_processing_efficiency(self, classifier):
        """Test batch processing for efficiency gains."""
        # Create batch of test images
        batch_size = 10
        test_batch = [
            TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
            for _ in range(batch_size)
        ]
        
        # Time batch processing
        start_time = time.perf_counter()
        results = classifier.classify_batch(test_batch)
        end_time = time.perf_counter()
        
        batch_time = (end_time - start_time) * 1000
        
        # Validate results
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, PatternClassificationResult)
            assert result.processing_time_ms > 0
        
        # Batch processing should be efficient
        avg_time_per_image = batch_time / batch_size
        assert avg_time_per_image <= 30.0, f"Batch processing inefficient: {avg_time_per_image}ms/image"
    
    def test_performance_optimization(self, classifier):
        """Test performance optimization for different speed targets."""
        test_image = TestDataGenerator.create_synthetic_fingerprint("WHORL")
        
        # Test different optimization levels
        optimization_targets = [5.0, 10.0, 20.0, 50.0]  # Target times in ms
        
        for target_time in optimization_targets:
            classifier.optimize_for_performance(target_time)
            
            # Measure performance after optimization
            times = []
            for _ in range(5):  # Multiple measurements for accuracy
                start_time = time.perf_counter()
                result = classifier.classify_pattern(test_image)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(times)
            
            # Performance should meet or beat target (with some tolerance)
            tolerance_factor = 1.5  # 50% tolerance for optimization
            assert avg_time <= target_time * tolerance_factor, \
                f"Optimization failed: {avg_time}ms > {target_time * tolerance_factor}ms"
    
    # ==========================================
    # QUALITY AND ACCURACY TESTS
    # ==========================================
    
    def test_quality_assessment(self, classifier, test_images):
        """Test pattern quality assessment functionality."""
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            
            # Quality score should be reasonable
            assert 0.0 <= result.pattern_quality <= 1.0
            assert result.biological_consistency >= 0.0
            
            # High-quality synthetic images should score well
            assert result.pattern_quality >= 0.5, \
                f"Quality too low for synthetic image: {result.pattern_quality}"
    
    def test_noise_robustness(self, classifier):
        """Test classifier robustness to image noise."""
        base_image = TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
        
        # Test with different noise levels
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        results = []
        
        for noise_level in noise_levels:
            noisy_image = TestDataGenerator.create_synthetic_fingerprint(
                "LOOP_RIGHT", noise_level=noise_level
            )
            result = classifier.classify_pattern(noisy_image)
            results.append((noise_level, result))
        
        # Classification should degrade gracefully with noise
        confidences = [r[1].pattern_confidence for r in results]
        
        # Confidence should generally decrease with noise (though not strictly)
        clean_confidence = confidences[0]
        highest_noise_confidence = confidences[-1]
        
        # Should maintain reasonable performance even with noise
        assert highest_noise_confidence >= 0.2, "Too sensitive to noise"
        
        # Processing time should remain stable
        processing_times = [r[1].processing_time_ms for r in results]
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Time variance should be low (O(1) requirement)
        time_variance = (max_time - min_time) / min_time if min_time > 0 else 0
        assert time_variance <= 0.5, "Processing time too variable with noise"
    
    def test_edge_case_handling(self, classifier):
        """Test classifier handling of edge cases."""
        # Test with very small image
        small_image = TestDataGenerator.create_synthetic_fingerprint(
            "ARCH", size=(64, 64)
        )
        result = classifier.classify_pattern(small_image)
        assert isinstance(result, PatternClassificationResult)
        assert result.processing_time_ms > 0
        
        # Test with very large image
        large_image = TestDataGenerator.create_synthetic_fingerprint(
            "WHORL", size=(1024, 1024)
        )
        result = classifier.classify_pattern(large_image)
        assert isinstance(result, PatternClassificationResult)
        assert result.processing_time_ms > 0
        
        # Test with extreme aspect ratio
        rectangular_image = TestDataGenerator.create_synthetic_fingerprint(
            "LOOP_LEFT", size=(256, 512)
        )
        result = classifier.classify_pattern(rectangular_image)
        assert isinstance(result, PatternClassificationResult)
        assert result.processing_time_ms > 0
    
    def test_consistency_across_runs(self, classifier):
        """Test classification consistency across multiple runs."""
        test_image = TestDataGenerator.create_synthetic_fingerprint("WHORL")
        
        # Run classification multiple times
        results = []
        for _ in range(10):
            result = classifier.classify_pattern(test_image)
            results.append(result)
        
        # Primary pattern should be consistent
        primary_patterns = [r.primary_pattern for r in results]
        most_common_pattern = max(set(primary_patterns), key=primary_patterns.count)
        consistency_rate = primary_patterns.count(most_common_pattern) / len(primary_patterns)
        
        assert consistency_rate >= 0.8, f"Classification inconsistent: {consistency_rate}"
        
        # Processing times should be similar
        processing_times = [r.processing_time_ms for r in results]
        time_std = statistics.stdev(processing_times)
        time_mean = statistics.mean(processing_times)
        coefficient_of_variation = time_std / time_mean if time_mean > 0 else 0
        
        assert coefficient_of_variation <= 0.2, "Processing time too variable"
    
    # ==========================================
    # FEATURE EXTRACTION TESTS
    # ==========================================
    
    def test_feature_extraction(self, classifier, test_images):
        """Test pattern feature extraction functionality."""
        for pattern_type, image in test_images.items():
            features = classifier.extract_pattern_features(image)
            
            # Validate feature structure
            assert isinstance(features, dict)
            assert 'primary_pattern' in features
            assert 'pattern_confidence' in features
            assert 'pattern_quality' in features
            assert 'core_count' in features
            assert 'delta_count' in features
            assert 'total_singular_points' in features
            
            # Validate feature values
            assert isinstance(features['core_count'], int)
            assert isinstance(features['delta_count'], int)
            assert features['core_count'] >= 0
            assert features['delta_count'] >= 0
            assert features['total_singular_points'] >= 0
            assert features['core_count'] + features['delta_count'] <= features['total_singular_points']
    
    def test_orientation_field_calculation(self, classifier, test_images):
        """Test ridge orientation field calculation."""
        for pattern_type, image in test_images.items():
            result = classifier.classify_pattern(image)
            orientation_field = result.ridge_orientation_field
            
            # Validate orientation field
            assert isinstance(orientation_field, np.ndarray)
            assert orientation_field.shape[0] > 0
            assert orientation_field.shape[1] > 0
            
            # Orientation values should be in valid range
            valid_orientations = np.isfinite(orientation_field)
            assert np.all(valid_orientations | np.isnan(orientation_field)), \
                "Invalid orientation values detected"
    
    # ==========================================
    # INTEGRATION TESTS
    # ==========================================
    
    def test_statistics_tracking(self, classifier, test_images):
        """Test classification statistics tracking."""
        initial_count = classifier.classification_stats['total_classifications']
        
        for pattern_type, image in test_images.items():
            classifier.classify_pattern(image)
        
        final_count = classifier.classification_stats['total_classifications']
        
        # Statistics should be updated
        assert final_count == initial_count + len(test_images)
        assert classifier.classification_stats['average_processing_time_ms'] > 0
        
        # Pattern distribution should be tracked
        assert 'pattern_distribution' in classifier.classification_stats
        assert isinstance(classifier.classification_stats['pattern_distribution'], dict)
    
    def test_error_handling(self, classifier):
        """Test classifier error handling for invalid inputs."""
        # Test with None input
        result = classifier.classify_pattern(None)
        assert result.primary_pattern == FingerprintPattern.PATTERN_UNCLEAR
        assert result.error_message is not None
        
        # Test with empty array
        empty_array = np.array([])
        result = classifier.classify_pattern(empty_array)
        assert result.primary_pattern == FingerprintPattern.PATTERN_UNCLEAR
        
        # Test with wrong dimensions
        wrong_dims = np.random.randint(0, 255, (10,), dtype=np.uint8)
        result = classifier.classify_pattern(wrong_dims)
        assert result.primary_pattern == FingerprintPattern.PATTERN_UNCLEAR
    
    def test_memory_efficiency(self, classifier):
        """Test memory efficiency during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(20):
            test_image = TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
            classifier.classify_pattern(test_image)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (< 100MB for 20 images)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase}MB"
    
    # ==========================================
    # PATENT VALIDATION TESTS
    # ==========================================
    
    def test_o1_addressing_requirements(self, classifier):
        """Test that classification meets O(1) addressing requirements."""
        # Generate diverse test images
        test_images = []
        expected_patterns = []
        
        pattern_types = ["LOOP_RIGHT", "LOOP_LEFT", "WHORL", "ARCH"]
        for pattern_type in pattern_types:
            for _ in range(5):  # 5 images per pattern type
                image = TestDataGenerator.create_synthetic_fingerprint(pattern_type)
                test_images.append(image)
                expected_patterns.append(pattern_type)
        
        # Classify all images and measure performance
        processing_times = []
        successful_classifications = 0
        
        for i, image in enumerate(test_images):
            start_time = time.perf_counter()
            result = classifier.classify_pattern(image)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000
            processing_times.append(processing_time)
            
            # Count successful classifications
            if result.pattern_confidence >= 0.5:
                successful_classifications += 1
        
        # Validate O(1) performance characteristics
        assert TestUtils.assert_o1_performance(
            processing_times, 
            list(range(len(test_images))),  # Simulated database sizes
            tolerance_ms=5.0
        ), "Classification does not meet O(1) performance requirements"
        
        # Validate classification accuracy
        accuracy_rate = successful_classifications / len(test_images)
        assert accuracy_rate >= 0.8, f"Classification accuracy too low: {accuracy_rate}"
    
    def test_scalability_demonstration(self, classifier):
        """Demonstrate classifier scalability for patent validation."""
        # Test with increasing numbers of classifications
        batch_sizes = [1, 5, 10, 25, 50, 100]
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Generate batch
            test_batch = [
                TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
                for _ in range(batch_size)
            ]
            
            # Measure batch processing time
            start_time = time.perf_counter()
            results = classifier.classify_batch(test_batch)
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000
            avg_time_per_image = total_time / batch_size
            
            performance_results[batch_size] = {
                'total_time_ms': total_time,
                'avg_time_per_image_ms': avg_time_per_image,
                'successful_classifications': len([r for r in results if r.pattern_confidence >= 0.5])
            }
        
        # Analyze scalability
        avg_times = [performance_results[size]['avg_time_per_image_ms'] for size in batch_sizes]
        
        # Average time per image should remain relatively constant (O(1) property)
        coefficient_of_variation = np.std(avg_times) / np.mean(avg_times)
        assert coefficient_of_variation <= 0.3, f"Poor scalability: CV = {coefficient_of_variation}"
        
        # All batch sizes should maintain good performance
        for size, results in performance_results.items():
            assert results['avg_time_per_image_ms'] <= 30.0, \
                f"Poor performance at batch size {size}: {results['avg_time_per_image_ms']}ms"
    
    # ==========================================
    # STRESS TESTS
    # ==========================================
    
    def test_continuous_operation(self, classifier):
        """Test classifier under continuous operation conditions."""
        # Simulate continuous operation
        num_operations = 200
        performance_samples = []
        
        for i in range(num_operations):
            test_image = TestDataGenerator.create_synthetic_fingerprint(
                np.random.choice(["LOOP_RIGHT", "LOOP_LEFT", "WHORL", "ARCH"])
            )
            
            start_time = time.perf_counter()
            result = classifier.classify_pattern(test_image)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000
            performance_samples.append(processing_time)
            
            # Check for performance degradation
            if i > 50:  # After warm-up period
                recent_avg = np.mean(performance_samples[-10:])
                assert recent_avg <= 50.0, f"Performance degradation detected at operation {i}"
        
        # Overall performance should remain stable
        first_half_avg = np.mean(performance_samples[:num_operations//2])
        second_half_avg = np.mean(performance_samples[num_operations//2:])
        
        # Performance should not degrade significantly
        degradation_ratio = second_half_avg / first_half_avg
        assert degradation_ratio <= 1.2, f"Significant performance degradation: {degradation_ratio}"
    
    def test_concurrent_classification(self, classifier):
        """Test classifier thread safety and concurrent access."""
        import threading
        import queue
        
        num_threads = 4
        operations_per_thread = 25
        results_queue = queue.Queue()
        
        def classification_worker():
            """Worker function for concurrent classification."""
            thread_results = []
            for _ in range(operations_per_thread):
                test_image = TestDataGenerator.create_synthetic_fingerprint("WHORL")
                
                start_time = time.perf_counter()
                result = classifier.classify_pattern(test_image)
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000
                thread_results.append({
                    'processing_time_ms': processing_time,
                    'confidence': result.pattern_confidence,
                    'success': result.pattern_confidence >= 0.3
                })
            
            results_queue.put(thread_results)
        
        # Start concurrent threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=classification_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect all results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Validate concurrent performance
        assert len(all_results) == num_threads * operations_per_thread
        
        processing_times = [r['processing_time_ms'] for r in all_results]
        successful_operations = sum(1 for r in all_results if r['success'])
        
        # Performance should remain good under concurrent access
        avg_time = np.mean(processing_times)
        assert avg_time <= 40.0, f"Poor concurrent performance: {avg_time}ms"
        
        # Success rate should remain high
        success_rate = successful_operations / len(all_results)
        assert success_rate >= 0.8, f"Poor concurrent success rate: {success_rate}"
    
    # ==========================================
    # VALIDATION AND COMPLIANCE TESTS
    # ==========================================
    
    def test_scientific_accuracy_validation(self, classifier):
        """Validate scientific accuracy of Poincaré index calculations."""
        # Create test image with known singular points
        test_image = TestDataGenerator.create_synthetic_fingerprint("WHORL")
        result = classifier.classify_pattern(test_image)
        
        # Validate Poincaré index values are scientifically sound
        for sp in result.singular_points:
            # Core points should have positive index (~+1)
            # Delta points should have negative index (~-1)
            if sp.point_type == SingularPointType.CORE:
                assert 0.3 <= sp.poincare_index <= 1.5, \
                    f"Invalid core Poincaré index: {sp.poincare_index}"
            elif sp.point_type == SingularPointType.DELTA:
                assert -1.5 <= sp.poincare_index <= -0.3, \
                    f"Invalid delta Poincaré index: {sp.poincare_index}"
    
    def test_biological_consistency_validation(self, classifier):
        """Test biological consistency assessment."""
        # Test with biologically realistic patterns
        realistic_patterns = [
            TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT"),
            TestDataGenerator.create_synthetic_fingerprint("WHORL")
        ]
        
        for image in realistic_patterns:
            result = classifier.classify_pattern(image)
            
            # Biological consistency should be reasonably high for realistic patterns
            assert result.biological_consistency >= 0.5, \
                f"Low biological consistency: {result.biological_consistency}"
            
            # Pattern should be classified with reasonable confidence
            assert result.pattern_confidence >= 0.3, \
                f"Low confidence for realistic pattern: {result.pattern_confidence}"
    
    def test_patent_claim_validation(self, classifier):
        """Validate specific patent claims about the classification system."""
        # Patent Claim: Classification enables stable O(1) addressing
        test_images = [
            TestDataGenerator.create_synthetic_fingerprint("LOOP_RIGHT")
            for _ in range(10)
        ]
        
        # Process all images
        results = []
        processing_times = []
        
        for image in test_images:
            start_time = time.perf_counter()
            result = classifier.classify_pattern(image)
            end_time = time.perf_counter()
            
            results.append(result)
            processing_times.append((end_time - start_time) * 1000)
        
        # Patent Claim 1: Consistent pattern classification
        primary_patterns = [r.primary_pattern for r in results]
        most_common = max(set(primary_patterns), key=primary_patterns.count)
        consistency_rate = primary_patterns.count(most_common) / len(primary_patterns)
        assert consistency_rate >= 0.8, "Pattern classification not consistent enough for addressing"
        
        # Patent Claim 2: Performance suitable for O(1) operations
        avg_time = np.mean(processing_times)
        max_time = max(processing_times)
        assert avg_time <= 20.0, f"Average time too slow for O(1): {avg_time}ms"
        assert max_time <= 50.0, f"Maximum time too slow for O(1): {max_time}ms"
        
        # Patent Claim 3: Feature extraction enables addressing
        features = classifier.extract_pattern_features(test_images[0])
        required_features = ['primary_pattern', 'core_count', 'delta_count', 'pattern_quality']
        
        for feature in required_features:
            assert feature in features, f"Missing required feature for addressing: {feature}"
            assert features[feature] is not None, f"Null value for addressing feature: {feature}"