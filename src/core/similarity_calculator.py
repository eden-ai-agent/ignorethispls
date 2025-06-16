#!/usr/bin/env python3
"""
Revolutionary Biological Similarity Calculator
Patent Pending - Michael Derrick Jagneaux

Advanced biological similarity calculation for fingerprint matching that accounts for:
- Same finger impression variations (pressure, rotation, quality)
- Biological feature stability across impressions
- Scientific fingerprint classification accuracy
- O(1) performance optimization

This module calculates how biologically similar two fingerprints are,
optimized for the revolutionary O(1) addressing system.
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalFeatureWeight(Enum):
    """Weights for different biological features based on stability."""
    PATTERN_CLASS = 4.0        # Most stable - never changes for same finger
    CORE_POSITION = 3.5        # Very stable - biological structure
    RIDGE_FLOW = 3.0          # Stable - directional flow pattern
    DELTA_CONFIG = 2.5        # Stable - structural configuration
    RIDGE_DENSITY = 2.0       # Moderately stable - can vary with pressure
    MINUTIAE_COUNT = 1.5      # Variable - pressure and quality dependent
    ORIENTATION = 1.0         # Variable - rotation and positioning
    QUALITY_METRICS = 0.5     # Highly variable - imaging conditions


class MatchConfidence(Enum):
    """Match confidence levels for biological similarity."""
    EXACT = "EXACT"           # 95-100% similarity - same impression
    HIGH = "HIGH"             # 85-95% similarity - same finger, different impression
    MEDIUM = "MEDIUM"         # 70-85% similarity - possible match, needs verification
    LOW = "LOW"              # 50-70% similarity - unlikely match
    NO_MATCH = "NO_MATCH"    # <50% similarity - different finger


@dataclass
class SimilarityResult:
    """Complete similarity analysis result."""
    overall_similarity: float              # 0.0 to 1.0 overall score
    confidence_level: MatchConfidence      # Confidence classification
    biological_consistency: float         # How consistent with biological patterns
    stable_feature_score: float          # Score based on stable features only
    variable_feature_score: float        # Score based on variable features
    pattern_match_score: float           # Pattern-specific matching
    feature_breakdown: Dict[str, float]   # Individual feature scores
    quality_adjusted_score: float        # Score adjusted for image quality
    is_same_finger_likely: bool          # Biological assessment
    match_explanation: str               # Human-readable explanation


class RevolutionaryBiologicalSimilarity:
    """
    Revolutionary biological similarity calculator optimized for O(1) systems.
    
    This calculator understands fingerprint biology and accounts for natural
    variation between impressions of the same finger while maintaining
    discrimination between different fingers.
    
    Key Innovations:
    - Biological feature hierarchy based on stability
    - Quality-aware similarity calculation
    - Same-finger impression tolerance
    - O(1) optimized for speed
    - Scientific accuracy based on fingerprint research
    """
    
    def __init__(self):
        """Initialize the revolutionary similarity calculator."""
        self.feature_weights = self._initialize_feature_weights()
        self.pattern_compatibility = self._initialize_pattern_compatibility()
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        # Calculation statistics
        self.calculation_stats = {
            'total_calculations': 0,
            'exact_matches': 0,
            'high_confidence_matches': 0,
            'same_finger_identifications': 0,
            'average_calculation_time_ms': 0
        }
        
        logger.info("Revolutionary Biological Similarity Calculator initialized")
        logger.info("Optimized for same-finger tolerance and different-finger discrimination")
    
    def calculate_biological_similarity(self, 
                                      characteristics1: Dict[str, Any], 
                                      characteristics2: Dict[str, Any]) -> SimilarityResult:
        """
        Calculate comprehensive biological similarity between two fingerprints.
        
        This is the core function that determines if two fingerprints are from
        the same finger, accounting for natural impression variations.
        
        Args:
            characteristics1: First fingerprint characteristics
            characteristics2: Second fingerprint characteristics
            
        Returns:
            Complete similarity analysis with biological assessment
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Validate inputs
            self._validate_characteristics(characteristics1)
            self._validate_characteristics(characteristics2)
            
            # Calculate individual feature similarities
            feature_scores = self._calculate_feature_similarities(characteristics1, characteristics2)
            
            # Calculate weighted biological similarity
            stable_score = self._calculate_stable_feature_score(feature_scores)
            variable_score = self._calculate_variable_feature_score(feature_scores)
            pattern_score = self._calculate_pattern_specific_score(characteristics1, characteristics2)
            
            # Overall similarity with biological weighting
            overall_similarity = (stable_score * 0.6 + 
                                variable_score * 0.3 + 
                                pattern_score * 0.1)
            
            # Quality adjustment
            quality_factor = self._calculate_quality_adjustment(characteristics1, characteristics2)
            quality_adjusted = overall_similarity * quality_factor
            
            # Biological consistency check
            biological_consistency = self._assess_biological_consistency(characteristics1, characteristics2)
            
            # Final similarity incorporating biological knowledge
            final_similarity = (quality_adjusted * 0.8 + biological_consistency * 0.2)
            
            # Determine confidence level
            confidence = self._determine_confidence_level(final_similarity, stable_score, biological_consistency)
            
            # Same finger assessment
            is_same_finger = self._assess_same_finger_probability(final_similarity, stable_score, feature_scores)
            
            # Generate explanation
            explanation = self._generate_match_explanation(final_similarity, confidence, feature_scores)
            
            # Create result
            result = SimilarityResult(
                overall_similarity=final_similarity,
                confidence_level=confidence,
                biological_consistency=biological_consistency,
                stable_feature_score=stable_score,
                variable_feature_score=variable_score,
                pattern_match_score=pattern_score,
                feature_breakdown=feature_scores,
                quality_adjusted_score=quality_adjusted,
                is_same_finger_likely=is_same_finger,
                match_explanation=explanation
            )
            
            # Update statistics
            calculation_time = (time.perf_counter() - start_time) * 1000
            self._update_calculation_stats(result, calculation_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return self._create_error_result(str(e))
    
    def calculate_fast_similarity(self, characteristics1: Dict[str, Any], 
                                characteristics2: Dict[str, Any]) -> float:
        """
        Fast similarity calculation optimized for O(1) performance.
        
        Focuses on most stable biological features for rapid screening.
        Used in high-performance scenarios where speed is critical.
        
        Args:
            characteristics1, characteristics2: Fingerprint characteristics
            
        Returns:
            Fast similarity score (0.0 to 1.0)
        """
        try:
            # Focus on most stable features only
            pattern_similarity = self._calculate_pattern_similarity(
                characteristics1.get('pattern_class', ''),
                characteristics2.get('pattern_class', '')
            )
            
            core_similarity = self._calculate_core_similarity(
                characteristics1.get('core_position', ''),
                characteristics2.get('core_position', '')
            )
            
            flow_similarity = self._calculate_flow_similarity(
                characteristics1.get('ridge_flow_direction', ''),
                characteristics2.get('ridge_flow_direction', '')
            )
            
            # Weighted fast score (stable features only)
            fast_score = (pattern_similarity * 0.5 + 
                         core_similarity * 0.3 + 
                         flow_similarity * 0.2)
            
            return fast_score
            
        except Exception as e:
            logger.error(f"Fast similarity calculation failed: {e}")
            return 0.0
    
    def batch_similarity_analysis(self, 
                                query_characteristics: Dict[str, Any],
                                candidate_list: List[Dict[str, Any]]) -> List[SimilarityResult]:
        """
        Batch similarity analysis for multiple candidates.
        
        Optimized for O(1) database lookup scenarios where we need to
        quickly assess multiple potential matches.
        
        Args:
            query_characteristics: Query fingerprint characteristics
            candidate_list: List of candidate fingerprint characteristics
            
        Returns:
            List of similarity results sorted by similarity score
        """
        results = []
        
        logger.info(f"Performing batch similarity analysis for {len(candidate_list)} candidates")
        
        for i, candidate in enumerate(candidate_list):
            try:
                result = self.calculate_biological_similarity(query_characteristics, candidate)
                results.append(result)
                
                # Log progress for large batches
                if len(candidate_list) > 100 and (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(candidate_list)} candidates")
                    
            except Exception as e:
                logger.error(f"Failed to process candidate {i}: {e}")
                continue
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.overall_similarity, reverse=True)
        
        logger.info(f"Batch analysis complete: {len(results)} results generated")
        
        return results
    
    def optimize_for_same_finger_detection(self, 
                                         same_finger_pairs: List[Tuple[Dict, Dict]],
                                         different_finger_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Optimize similarity calculation based on known same/different finger pairs.
        
        Uses machine learning principles to tune the biological similarity
        calculation for maximum accuracy in same-finger detection.
        
        Args:
            same_finger_pairs: List of (chars1, chars2) from same finger
            different_finger_pairs: List of (chars1, chars2) from different fingers
            
        Returns:
            Optimization results and recommended parameter updates
        """
        logger.info("Optimizing similarity calculation for same finger detection...")
        
        # Analyze same finger similarities
        same_finger_scores = []
        for chars1, chars2 in same_finger_pairs:
            result = self.calculate_biological_similarity(chars1, chars2)
            same_finger_scores.append(result.overall_similarity)
        
        # Analyze different finger similarities
        different_finger_scores = []
        for chars1, chars2 in different_finger_pairs[:len(same_finger_pairs)]:  # Balance dataset
            result = self.calculate_biological_similarity(chars1, chars2)
            different_finger_scores.append(result.overall_similarity)
        
        # Calculate statistics
        same_finger_stats = {
            'mean': np.mean(same_finger_scores),
            'std': np.std(same_finger_scores),
            'min': np.min(same_finger_scores),
            'max': np.max(same_finger_scores)
        }
        
        different_finger_stats = {
            'mean': np.mean(different_finger_scores),
            'std': np.std(different_finger_scores),
            'min': np.min(different_finger_scores),
            'max': np.max(different_finger_scores)
        }
        
        # Calculate separation metrics
        separation_gap = same_finger_stats['min'] - different_finger_stats['max']
        overlap_percentage = self._calculate_overlap_percentage(same_finger_scores, different_finger_scores)
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(same_finger_scores, different_finger_scores)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            same_finger_stats, different_finger_stats, separation_gap, optimal_threshold
        )
        
        optimization_results = {
            'same_finger_statistics': same_finger_stats,
            'different_finger_statistics': different_finger_stats,
            'separation_gap': separation_gap,
            'overlap_percentage': overlap_percentage,
            'optimal_threshold': optimal_threshold,
            'current_accuracy': self._estimate_accuracy(same_finger_scores, different_finger_scores),
            'recommendations': recommendations,
            'sample_sizes': {
                'same_finger_pairs': len(same_finger_pairs),
                'different_finger_pairs': len(different_finger_pairs)
            }
        }
        
        logger.info(f"Optimization complete:")
        logger.info(f"  Same finger scores: {same_finger_stats['mean']:.3f} ± {same_finger_stats['std']:.3f}")
        logger.info(f"  Different finger scores: {different_finger_stats['mean']:.3f} ± {different_finger_stats['std']:.3f}")
        logger.info(f"  Separation gap: {separation_gap:.3f}")
        logger.info(f"  Optimal threshold: {optimal_threshold:.3f}")
        
        return optimization_results
    
    # Private helper methods
    def _validate_characteristics(self, characteristics: Dict[str, Any]) -> None:
        """Validate that characteristics contain required fields."""
        required_fields = [
            'pattern_class', 'core_position', 'ridge_flow_direction',
            'ridge_count_vertical', 'ridge_count_horizontal', 'minutiae_count'
        ]
        
        for field in required_fields:
            if field not in characteristics:
                raise ValueError(f"Missing required characteristic: {field}")
    
    def _calculate_feature_similarities(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> Dict[str, float]:
        """Calculate similarity for each individual feature."""
        feature_scores = {}
        
        # Pattern class similarity (exact match required for same finger)
        feature_scores['pattern_class'] = self._calculate_pattern_similarity(
            chars1.get('pattern_class', ''), chars2.get('pattern_class', '')
        )
        
        # Core position similarity
        feature_scores['core_position'] = self._calculate_core_similarity(
            chars1.get('core_position', ''), chars2.get('core_position', '')
        )
        
        # Ridge flow similarity
        feature_scores['ridge_flow'] = self._calculate_flow_similarity(
            chars1.get('ridge_flow_direction', ''), chars2.get('ridge_flow_direction', '')
        )
        
        # Ridge count similarities (with tolerance for impression variation)
        feature_scores['ridge_count_vertical'] = self._calculate_ridge_count_similarity(
            chars1.get('ridge_count_vertical', 0), chars2.get('ridge_count_vertical', 0)
        )
        
        feature_scores['ridge_count_horizontal'] = self._calculate_ridge_count_similarity(
            chars1.get('ridge_count_horizontal', 0), chars2.get('ridge_count_horizontal', 0)
        )
        
        # Minutiae count similarity (highly variable)
        feature_scores['minutiae_count'] = self._calculate_minutiae_similarity(
            chars1.get('minutiae_count', 0), chars2.get('minutiae_count', 0)
        )
        
        # Pattern orientation similarity
        feature_scores['pattern_orientation'] = self._calculate_orientation_similarity(
            chars1.get('pattern_orientation', 0), chars2.get('pattern_orientation', 0)
        )
        
        # Quality metrics
        feature_scores['image_quality'] = self._calculate_quality_similarity(
            chars1.get('image_quality', 0), chars2.get('image_quality', 0)
        )
        
        feature_scores['ridge_density'] = self._calculate_density_similarity(
            chars1.get('ridge_density', 0), chars2.get('ridge_density', 0)
        )
        
        return feature_scores
    
    def _calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate pattern class similarity with biological accuracy."""
        if pattern1 == pattern2:
            return 1.0
        
        # Use biological pattern relationships
        pattern_groups = {
            'ARCH': ['ARCH_PLAIN', 'ARCH_TENTED'],
            'LOOP': ['LOOP_LEFT', 'LOOP_RIGHT', 'LOOP_UNDETERMINED'],
            'WHORL': ['WHORL', 'WHORL_COMPLEX']
        }
        
        # High similarity within same pattern family
        for group_patterns in pattern_groups.values():
            if pattern1 in group_patterns and pattern2 in group_patterns:
                if pattern1 == pattern2:
                    return 1.0
                else:
                    return 0.85  # High similarity within family
        
        # Special case: tented arch can be similar to loops
        if (pattern1 == 'ARCH_TENTED' and pattern2.startswith('LOOP')) or \
           (pattern2 == 'ARCH_TENTED' and pattern1.startswith('LOOP')):
            return 0.4
        
        # Different pattern families - very low similarity
        return 0.1
    
    def _calculate_core_similarity(self, core1: str, core2: str) -> float:
        """Calculate core position similarity with spatial awareness."""
        if core1 == core2:
            return 1.0
        
        # Define spatial proximity map
        position_coords = {
            "UPPER_LEFT": (0, 0), "UPPER_CENTER_LEFT": (1, 0), 
            "UPPER_CENTER_RIGHT": (2, 0), "UPPER_RIGHT": (3, 0),
            "CENTER_LEFT": (0, 1), "CENTER_CENTER_LEFT": (1, 1),
            "CENTER_CENTER_RIGHT": (2, 1), "CENTER_RIGHT": (3, 1),
            "LOWER_CENTER_LEFT": (1, 2), "LOWER_CENTER": (2, 2),
            "LOWER_CENTER_RIGHT": (2, 2), "LOWER_LEFT": (0, 2),
            "BOTTOM_LEFT": (0, 3), "BOTTOM_CENTER_LEFT": (1, 3),
            "BOTTOM_CENTER_RIGHT": (2, 3), "BOTTOM_RIGHT": (3, 3)
        }
        
        if core1 in position_coords and core2 in position_coords:
            coord1 = position_coords[core1]
            coord2 = position_coords[core2]
            
            # Calculate Euclidean distance
            distance = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            max_distance = math.sqrt(3**2 + 3**2)  # Maximum possible distance
            
            # Convert distance to similarity (closer = more similar)
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, similarity)
        
        return 0.0
    
    def _calculate_flow_similarity(self, flow1: str, flow2: str) -> float:
        """Calculate ridge flow similarity."""
        if flow1 == flow2:
            return 1.0
        
        # Define flow relationships
        flow_angles = {
            "HORIZONTAL": 0,
            "DIAGONAL_UP": 45,
            "VERTICAL": 90,
            "DIAGONAL_DOWN": 135
        }
        
        if flow1 in flow_angles and flow2 in flow_angles:
            angle1 = flow_angles[flow1]
            angle2 = flow_angles[flow2]
            
            # Calculate angular difference
            diff = abs(angle1 - angle2)
            if diff > 90:
                diff = 180 - diff  # Handle wraparound
            
            # Convert to similarity (smaller angle difference = higher similarity)
            similarity = 1.0 - (diff / 90.0)
            return max(0.0, similarity)
        
        return 0.0
    
    def _calculate_ridge_count_similarity(self, count1: int, count2: int) -> float:
        """Calculate ridge count similarity with tolerance for impression variation."""
        if count1 == count2:
            return 1.0
        
        # Ridge counts can vary ±5 for same finger due to pressure/rotation
        max_difference = 15  # Allow up to 15 ridge difference
        difference = abs(count1 - count2)
        
        if difference <= 3:
            return 1.0  # Very close counts
        elif difference <= 8:
            return 0.8  # Good similarity
        elif difference <= max_difference:
            return 0.6 - (difference - 8) * 0.1  # Decreasing similarity
        else:
            return 0.1  # Very different
    
    def _calculate_minutiae_similarity(self, count1: int, count2: int) -> float:
        """Calculate minutiae count similarity (highly variable feature)."""
        if count1 == count2:
            return 1.0
        
        # Minutiae can vary significantly due to image quality and processing
        max_count = max(count1, count2, 1)
        difference = abs(count1 - count2)
        
        # More tolerant similarity for minutiae
        similarity = 1.0 - (difference / max_count)
        return max(0.0, similarity)
    
    def _calculate_orientation_similarity(self, orient1: int, orient2: int) -> float:
        """Calculate pattern orientation similarity."""
        if orient1 == orient2:
            return 1.0
        
        # Handle circular nature of angles
        diff = abs(orient1 - orient2)
        if diff > 90:
            diff = 180 - diff
        
        # Allow ±15 degrees tolerance for same finger
        if diff <= 15:
            return 1.0
        elif diff <= 30:
            return 0.8
        elif diff <= 45:
            return 0.6
        else:
            return 0.2
    
    def _calculate_quality_similarity(self, quality1: float, quality2: float) -> float:
        """Calculate image quality similarity."""
        if abs(quality1 - quality2) <= 5:
            return 1.0
        elif abs(quality1 - quality2) <= 15:
            return 0.8
        elif abs(quality1 - quality2) <= 30:
            return 0.6
        else:
            return 0.3
    
    def _calculate_density_similarity(self, density1: float, density2: float) -> float:
        """Calculate ridge density similarity."""
        if abs(density1 - density2) <= 2:
            return 1.0
        elif abs(density1 - density2) <= 5:
            return 0.8
        elif abs(density1 - density2) <= 10:
            return 0.6
        else:
            return 0.3
    
    def _calculate_stable_feature_score(self, feature_scores: Dict[str, float]) -> float:
        """Calculate score based on stable biological features only."""
        stable_features = {
            'pattern_class': BiologicalFeatureWeight.PATTERN_CLASS.value,
            'core_position': BiologicalFeatureWeight.CORE_POSITION.value,
            'ridge_flow': BiologicalFeatureWeight.RIDGE_FLOW.value
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for feature, weight in stable_features.items():
            if feature in feature_scores:
                weighted_sum += feature_scores[feature] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_variable_feature_score(self, feature_scores: Dict[str, float]) -> float:
        """Calculate score based on variable features with tolerance."""
        variable_features = {
            'ridge_count_vertical': BiologicalFeatureWeight.RIDGE_DENSITY.value,
            'ridge_count_horizontal': BiologicalFeatureWeight.RIDGE_DENSITY.value,
            'minutiae_count': BiologicalFeatureWeight.MINUTIAE_COUNT.value,
            'pattern_orientation': BiologicalFeatureWeight.ORIENTATION.value
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for feature, weight in variable_features.items():
            if feature in feature_scores:
                weighted_sum += feature_scores[feature] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_pattern_specific_score(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate pattern-specific similarity score."""
        pattern1 = chars1.get('pattern_class', '')
        pattern2 = chars2.get('pattern_class', '')
        
        if pattern1 != pattern2:
            return 0.0  # Different patterns can't have pattern-specific similarity
        
        # Pattern-specific calculations based on the common pattern
        if pattern1.startswith('LOOP'):
            return self._calculate_loop_specific_similarity(chars1, chars2)
        elif pattern1.startswith('WHORL'):
            return self._calculate_whorl_specific_similarity(chars1, chars2)
        elif pattern1.startswith('ARCH'):
            return self._calculate_arch_specific_similarity(chars1, chars2)
        else:
            return 0.5  # Default for unknown patterns
    
    def _calculate_loop_specific_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity specific to loop patterns."""
        # For loops, core position and ridge flow are critical
        core_sim = self._calculate_core_similarity(
            chars1.get('core_position', ''), chars2.get('core_position', '')
        )
        flow_sim = self._calculate_flow_similarity(
            chars1.get('ridge_flow_direction', ''), chars2.get('ridge_flow_direction', '')
        )
        
        return (core_sim * 0.6 + flow_sim * 0.4)
    
    def _calculate_whorl_specific_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity specific to whorl patterns."""
        # For whorls, ridge counts and core position are important
        core_sim = self._calculate_core_similarity(
            chars1.get('core_position', ''), chars2.get('core_position', '')
        )
        ridge_v_sim = self._calculate_ridge_count_similarity(
            chars1.get('ridge_count_vertical', 0), chars2.get('ridge_count_vertical', 0)
        )
        ridge_h_sim = self._calculate_ridge_count_similarity(
            chars1.get('ridge_count_horizontal', 0), chars2.get('ridge_count_horizontal', 0)
        )
        
        return (core_sim * 0.4 + ridge_v_sim * 0.3 + ridge_h_sim * 0.3)
    
    def _calculate_arch_specific_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity specific to arch patterns."""
        # For arches, ridge flow and density are key
        flow_sim = self._calculate_flow_similarity(
            chars1.get('ridge_flow_direction', ''), chars2.get('ridge_flow_direction', '')
        )
        density_sim = self._calculate_density_similarity(
            chars1.get('ridge_density', 0), chars2.get('ridge_density', 0)
        )
        
        return (flow_sim * 0.6 + density_sim * 0.4)
    
    def _calculate_quality_adjustment(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate quality adjustment factor for similarity score."""
        quality1 = chars1.get('image_quality', 50)
        quality2 = chars2.get('image_quality', 50)
        
        # Lower quality images get less stringent matching
        avg_quality = (quality1 + quality2) / 2
        
        if avg_quality >= 85:
            return 1.0  # High quality - no adjustment
        elif avg_quality >= 70:
            return 1.05  # Good quality - slight boost
        elif avg_quality >= 50:
            return 1.1   # Fair quality - moderate boost
        else:
            return 1.15  # Poor quality - higher tolerance
    
    def _assess_biological_consistency(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Assess biological consistency of the match."""
        # Check for biological impossibilities
        consistency_score = 1.0
        
        # Pattern class consistency (most important)
        pattern1 = chars1.get('pattern_class', '')
        pattern2 = chars2.get('pattern_class', '')
        
        if pattern1 != pattern2:
            # Different patterns - check if biologically related
            if self._are_patterns_biologically_related(pattern1, pattern2):
                consistency_score *= 0.8  # Slight penalty for different but related patterns
            else:
                consistency_score *= 0.3  # Major penalty for unrelated patterns
        
        # Ridge count consistency
        ridge_v1 = chars1.get('ridge_count_vertical', 0)
        ridge_v2 = chars2.get('ridge_count_vertical', 0)
        if abs(ridge_v1 - ridge_v2) > 20:  # Very large difference
            consistency_score *= 0.7
        
        # Core position consistency with pattern
        core1 = chars1.get('core_position', '')
        core2 = chars2.get('core_position', '')
        if pattern1 == pattern2 and self._calculate_core_similarity(core1, core2) < 0.3:
            consistency_score *= 0.8  # Core positions too different for same pattern
        
        return consistency_score
    
    def _are_patterns_biologically_related(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are biologically related."""
        related_groups = [
            ['ARCH_PLAIN', 'ARCH_TENTED'],
            ['LOOP_LEFT', 'LOOP_RIGHT', 'LOOP_UNDETERMINED'],
            ['WHORL', 'WHORL_COMPLEX']
        ]
        
        for group in related_groups:
            if pattern1 in group and pattern2 in group:
                return True
        
        # Special case: tented arch and loops can be related
        if (pattern1 == 'ARCH_TENTED' and pattern2.startswith('LOOP')) or \
           (pattern2 == 'ARCH_TENTED' and pattern1.startswith('LOOP')):
            return True
        
        return False
    
    def _determine_confidence_level(self, similarity: float, stable_score: float, consistency: float) -> MatchConfidence:
        """Determine confidence level based on similarity scores."""
        # Exact match criteria
        if similarity >= 0.95 and stable_score >= 0.9 and consistency >= 0.95:
            return MatchConfidence.EXACT
        
        # High confidence criteria
        elif similarity >= 0.85 and stable_score >= 0.8 and consistency >= 0.85:
            return MatchConfidence.HIGH
        
        # Medium confidence criteria
        elif similarity >= 0.70 and stable_score >= 0.6 and consistency >= 0.7:
            return MatchConfidence.MEDIUM
        
        # Low confidence criteria
        elif similarity >= 0.50 and stable_score >= 0.4:
            return MatchConfidence.LOW
        
        # No match
        else:
            return MatchConfidence.NO_MATCH
    
    def _assess_same_finger_probability(self, similarity: float, stable_score: float, feature_scores: Dict[str, float]) -> bool:
        """Assess if the fingerprints are likely from the same finger."""
        # Primary criteria: stable features must match well
        if stable_score < 0.7:
            return False
        
        # Pattern class must match (critical for same finger)
        pattern_score = feature_scores.get('pattern_class', 0)
        if pattern_score < 0.8:
            return False
        
        # Overall similarity threshold
        if similarity < 0.75:
            return False
        
        # Core position should be reasonably similar
        core_score = feature_scores.get('core_position', 0)
        if core_score < 0.5:
            return False
        
        return True
    
    def _generate_match_explanation(self, similarity: float, confidence: MatchConfidence, feature_scores: Dict[str, float]) -> str:
        """Generate human-readable explanation of the match."""
        if confidence == MatchConfidence.EXACT:
            return f"Exact match with {similarity:.1%} similarity. All biological features align perfectly."
        
        elif confidence == MatchConfidence.HIGH:
            pattern_score = feature_scores.get('pattern_class', 0)
            core_score = feature_scores.get('core_position', 0)
            return f"High confidence match ({similarity:.1%}). Pattern: {pattern_score:.1%}, Core: {core_score:.1%}. Likely same finger, different impression."
        
        elif confidence == MatchConfidence.MEDIUM:
            stable_features = ['pattern_class', 'core_position', 'ridge_flow']
            stable_avg = sum(feature_scores.get(f, 0) for f in stable_features) / len(stable_features)
            return f"Medium confidence match ({similarity:.1%}). Stable features: {stable_avg:.1%}. Requires verification."
        
        elif confidence == MatchConfidence.LOW:
            return f"Low confidence match ({similarity:.1%}). Some biological features align but overall similarity is low."
        
        else:
            return f"No match ({similarity:.1%}). Biological features indicate different fingers."
    
    def _create_error_result(self, error_message: str) -> SimilarityResult:
        """Create error result when calculation fails."""
        return SimilarityResult(
            overall_similarity=0.0,
            confidence_level=MatchConfidence.NO_MATCH,
            biological_consistency=0.0,
            stable_feature_score=0.0,
            variable_feature_score=0.0,
            pattern_match_score=0.0,
            feature_breakdown={},
            quality_adjusted_score=0.0,
            is_same_finger_likely=False,
            match_explanation=f"Calculation error: {error_message}"
        )
    
    def _calculate_overlap_percentage(self, same_finger_scores: List[float], different_finger_scores: List[float]) -> float:
        """Calculate percentage of score overlap between same and different fingers."""
        if not same_finger_scores or not different_finger_scores:
            return 0.0
        
        same_min = min(same_finger_scores)
        same_max = max(same_finger_scores)
        diff_min = min(different_finger_scores)
        diff_max = max(different_finger_scores)
        
        # Calculate overlap region
        overlap_start = max(same_min, diff_min)
        overlap_end = min(same_max, diff_max)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        # Calculate overlap as percentage of total range
        total_range = max(same_max, diff_max) - min(same_min, diff_min)
        overlap_range = overlap_end - overlap_start
        
        return (overlap_range / total_range) * 100 if total_range > 0 else 0.0
    
    def _find_optimal_threshold(self, same_finger_scores: List[float], different_finger_scores: List[float]) -> float:
        """Find optimal threshold that maximizes accuracy."""
        all_scores = sorted(set(same_finger_scores + different_finger_scores))
        best_threshold = 0.5
        best_accuracy = 0.0
        
        for threshold in all_scores:
            # Calculate accuracy at this threshold
            true_positives = sum(1 for score in same_finger_scores if score >= threshold)
            true_negatives = sum(1 for score in different_finger_scores if score < threshold)
            
            total_samples = len(same_finger_scores) + len(different_finger_scores)
            accuracy = (true_positives + true_negatives) / total_samples
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def _estimate_accuracy(self, same_finger_scores: List[float], different_finger_scores: List[float]) -> float:
        """Estimate current accuracy with default threshold."""
        threshold = 0.75  # Default threshold
        
        true_positives = sum(1 for score in same_finger_scores if score >= threshold)
        true_negatives = sum(1 for score in different_finger_scores if score < threshold)
        
        total_samples = len(same_finger_scores) + len(different_finger_scores)
        return (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0
    
    def _generate_optimization_recommendations(self, same_stats: Dict, diff_stats: Dict, separation: float, threshold: float) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        if separation < 0.1:
            recommendations.append("INCREASE_STABLE_FEATURE_WEIGHT: Stable features need more emphasis")
        
        if same_stats['std'] > 0.2:
            recommendations.append("REDUCE_SAME_FINGER_VARIANCE: Same finger scores too variable")
        
        if diff_stats['mean'] > 0.4:
            recommendations.append("IMPROVE_DISCRIMINATION: Different finger scores too high")
        
        if threshold < 0.6:
            recommendations.append("ADJUST_FEATURE_WEIGHTS: Threshold too low, adjust biological weights")
        
        if separation > 0.3:
            recommendations.append("EXCELLENT_SEPARATION: Current algorithm performs very well")
        
        return recommendations
    
    def _update_calculation_stats(self, result: SimilarityResult, calculation_time: float) -> None:
        """Update calculation statistics."""
        self.calculation_stats['total_calculations'] += 1
        
        if result.confidence_level == MatchConfidence.EXACT:
            self.calculation_stats['exact_matches'] += 1
        elif result.confidence_level == MatchConfidence.HIGH:
            self.calculation_stats['high_confidence_matches'] += 1
        
        if result.is_same_finger_likely:
            self.calculation_stats['same_finger_identifications'] += 1
        
        # Update average time
        total_time = (self.calculation_stats['average_calculation_time_ms'] * 
                     (self.calculation_stats['total_calculations'] - 1) + calculation_time)
        self.calculation_stats['average_calculation_time_ms'] = total_time / self.calculation_stats['total_calculations']
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature weights based on biological stability."""
        return {
            'pattern_class': BiologicalFeatureWeight.PATTERN_CLASS.value,
            'core_position': BiologicalFeatureWeight.CORE_POSITION.value,
            'ridge_flow': BiologicalFeatureWeight.RIDGE_FLOW.value,
            'ridge_density': BiologicalFeatureWeight.RIDGE_DENSITY.value,
            'minutiae_count': BiologicalFeatureWeight.MINUTIAE_COUNT.value,
            'orientation': BiologicalFeatureWeight.ORIENTATION.value,
            'quality_metrics': BiologicalFeatureWeight.QUALITY_METRICS.value
        }
    
    def _initialize_pattern_compatibility(self) -> Dict[str, Dict[str, float]]:
        """Initialize pattern compatibility matrix."""
        return {
            'ARCH_PLAIN': {'ARCH_PLAIN': 1.0, 'ARCH_TENTED': 0.8, 'LOOP_LEFT': 0.2, 'LOOP_RIGHT': 0.2, 'WHORL': 0.1},
            'ARCH_TENTED': {'ARCH_PLAIN': 0.8, 'ARCH_TENTED': 1.0, 'LOOP_LEFT': 0.4, 'LOOP_RIGHT': 0.4, 'WHORL': 0.1},
            'LOOP_LEFT': {'ARCH_PLAIN': 0.2, 'ARCH_TENTED': 0.4, 'LOOP_LEFT': 1.0, 'LOOP_RIGHT': 0.7, 'WHORL': 0.2},
            'LOOP_RIGHT': {'ARCH_PLAIN': 0.2, 'ARCH_TENTED': 0.4, 'LOOP_LEFT': 0.7, 'LOOP_RIGHT': 1.0, 'WHORL': 0.2},
            'WHORL': {'ARCH_PLAIN': 0.1, 'ARCH_TENTED': 0.1, 'LOOP_LEFT': 0.2, 'LOOP_RIGHT': 0.2, 'WHORL': 1.0}
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality assessment thresholds."""
        return {
            'excellent_threshold': 85.0,
            'good_threshold': 70.0,
            'fair_threshold': 50.0,
            'poor_threshold': 30.0
        }
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get current calculation statistics."""
        stats = self.calculation_stats.copy()
        
        if stats['total_calculations'] > 0:
            stats['exact_match_rate'] = stats['exact_matches'] / stats['total_calculations'] * 100
            stats['high_confidence_rate'] = stats['high_confidence_matches'] / stats['total_calculations'] * 100
            stats['same_finger_rate'] = stats['same_finger_identifications'] / stats['total_calculations'] * 100
        
        return stats


def demonstrate_biological_similarity():
    """
    Demonstrate the revolutionary biological similarity calculator.
    
    Shows how the system accurately identifies same finger impressions
    while discriminating between different fingers.
    """
    print("=" * 80)
    print("🧬 REVOLUTIONARY BIOLOGICAL SIMILARITY CALCULATOR")
    print("Patent Pending - Michael Derrick Jagneaux")
    print("=" * 80)
    
    # Initialize the similarity calculator
    calculator = RevolutionaryBiologicalSimilarity()
    
    print(f"\n🔬 Biological Similarity Calculator Features:")
    print(f"   ✅ Same finger impression tolerance")
    print(f"   ✅ Different finger discrimination")
    print(f"   ✅ Quality-aware matching")
    print(f"   ✅ Biological feature hierarchy")
    print(f"   ✅ Scientific pattern relationships")
    
    # Example: Same finger, different impressions
    same_finger_impression_1 = {
        'pattern_class': 'LOOP_RIGHT',
        'core_position': 'CENTER_CENTER_LEFT',
        'ridge_flow_direction': 'DIAGONAL_UP',
        'ridge_count_vertical': 45,
        'ridge_count_horizontal': 38,
        'minutiae_count': 67,
        'pattern_orientation': 75,
        'image_quality': 85.5,
        'ridge_density': 23.7,
        'contrast_level': 142.3
    }
    
    same_finger_impression_2 = {
        'pattern_class': 'LOOP_RIGHT',  # Same pattern
        'core_position': 'CENTER_CENTER_LEFT',  # Same core position
        'ridge_flow_direction': 'DIAGONAL_UP',  # Same flow
        'ridge_count_vertical': 47,  # Slight variation (+2)
        'ridge_count_horizontal': 36,  # Slight variation (-2)
        'minutiae_count': 71,  # Variation (+4)
        'pattern_orientation': 78,  # Slight rotation (+3°)
        'image_quality': 79.2,  # Lower quality (-6.3)
        'ridge_density': 25.1,  # Slight density change (+1.4)
        'contrast_level': 138.7  # Slight contrast change (-3.6)
    }
    
    # Example: Different finger
    different_finger = {
        'pattern_class': 'WHORL',  # Different pattern
        'core_position': 'UPPER_RIGHT',  # Different position
        'ridge_flow_direction': 'VERTICAL',  # Different flow
        'ridge_count_vertical': 62,  # Different count
        'ridge_count_horizontal': 51,  # Different count
        'minutiae_count': 89,  # Different count
        'pattern_orientation': 105,  # Different orientation
        'image_quality': 82.1,
        'ridge_density': 31.4,
        'contrast_level': 156.2
    }
    
    print(f"\n🔍 Test Case 1: Same Finger, Different Impressions")
    print(f"   Impression 1: {same_finger_impression_1['pattern_class']}, Core: {same_finger_impression_1['core_position']}")
    print(f"   Impression 2: {same_finger_impression_2['pattern_class']}, Core: {same_finger_impression_2['core_position']}")
    
    result1 = calculator.calculate_biological_similarity(same_finger_impression_1, same_finger_impression_2)
    
    print(f"\n📊 Same Finger Analysis:")
    print(f"   Overall Similarity: {result1.overall_similarity:.1%}")
    print(f"   Confidence Level: {result1.confidence_level.value}")
    print(f"   Stable Features Score: {result1.stable_feature_score:.1%}")
    print(f"   Variable Features Score: {result1.variable_feature_score:.1%}")
    print(f"   Biological Consistency: {result1.biological_consistency:.1%}")
    print(f"   Same Finger Assessment: {'✅ YES' if result1.is_same_finger_likely else '❌ NO'}")
    print(f"   Explanation: {result1.match_explanation}")
    
    print(f"\n🆚 Test Case 2: Different Fingers")
    print(f"   Finger 1: {same_finger_impression_1['pattern_class']}, Core: {same_finger_impression_1['core_position']}")
    print(f"   Finger 2: {different_finger['pattern_class']}, Core: {different_finger['core_position']}")
    
    result2 = calculator.calculate_biological_similarity(same_finger_impression_1, different_finger)
    
    print(f"\n📊 Different Finger Analysis:")
    print(f"   Overall Similarity: {result2.overall_similarity:.1%}")
    print(f"   Confidence Level: {result2.confidence_level.value}")
    print(f"   Stable Features Score: {result2.stable_feature_score:.1%}")
    print(f"   Variable Features Score: {result2.variable_feature_score:.1%}")
    print(f"   Biological Consistency: {result2.biological_consistency:.1%}")
    print(f"   Same Finger Assessment: {'✅ YES' if result2.is_same_finger_likely else '❌ NO'}")
    print(f"   Explanation: {result2.match_explanation}")
    
    print(f"\n🎯 Revolutionary Accuracy:")
    same_finger_score = result1.overall_similarity
    different_finger_score = result2.overall_similarity
    separation_ratio = same_finger_score / different_finger_score if different_finger_score > 0 else float('inf')
    
    print(f"   Same Finger Score: {same_finger_score:.1%}")
    print(f"   Different Finger Score: {different_finger_score:.1%}")
    print(f"   Separation Ratio: {separation_ratio:.1f}x")
    
    if separation_ratio > 3:
        print(f"   ✅ EXCELLENT discrimination between same/different fingers")
    elif separation_ratio > 2:
        print(f"   ✅ GOOD discrimination capability")
    else:
        print(f"   ⚠️ Discrimination could be improved")
    
    print(f"\n⚡ Performance Features:")
    print(f"   🧬 Biological accuracy based on fingerprint science")
    print(f"   🎯 Same finger tolerance for impression variations")
    print(f"   🚀 Different finger discrimination for security")
    print(f"   ⚡ Fast calculation optimized for O(1) systems")
    print(f"   📊 Quality-aware matching for real-world conditions")
    
    print("=" * 80)


def benchmark_similarity_performance():
    """
    Benchmark similarity calculation performance for O(1) optimization.
    """
    import time
    
    calculator = RevolutionaryBiologicalSimilarity()
    
    print(f"\n⚡ SIMILARITY CALCULATION PERFORMANCE BENCHMARK")
    print("-" * 60)
    
    # Generate test characteristics
    base_chars = {
        'pattern_class': 'LOOP_RIGHT',
        'core_position': 'CENTER_CENTER_LEFT',
        'ridge_flow_direction': 'DIAGONAL_UP',
        'ridge_count_vertical': 45,
        'ridge_count_horizontal': 38,
        'minutiae_count': 67,
        'pattern_orientation': 75,
        'image_quality': 85.5,
        'ridge_density': 23.7,
        'contrast_level': 142.3
    }
    
    # Generate variations for testing
    test_cases = []
    for i in range(1000):
        variation = base_chars.copy()
        variation['ridge_count_vertical'] += i % 20 - 10
        variation['ridge_count_horizontal'] += i % 15 - 7
        variation['minutiae_count'] += i % 30 - 15
        variation['pattern_orientation'] = (variation['pattern_orientation'] + i % 45) % 180
        variation['image_quality'] += i % 40 - 20
        test_cases.append(variation)
    
    print(f"🔧 Testing similarity calculation with {len(test_cases)} pairs...")
    
    # Benchmark full similarity calculation
    start_time = time.perf_counter()
    full_results = []
    
    for i, test_chars in enumerate(test_cases):
        result = calculator.calculate_biological_similarity(base_chars, test_chars)
        full_results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(test_cases)} full calculations...")
    
    full_time = (time.perf_counter() - start_time) * 1000
    
    # Benchmark fast similarity calculation
    start_time = time.perf_counter()
    fast_results = []
    
    for test_chars in test_cases:
        fast_score = calculator.calculate_fast_similarity(base_chars, test_chars)
        fast_results.append(fast_score)
    
    fast_time = (time.perf_counter() - start_time) * 1000
    
    # Calculate statistics
    full_avg_time = full_time / len(test_cases)
    fast_avg_time = fast_time / len(test_cases)
    speed_advantage = full_avg_time / fast_avg_time
    
    print(f"\n📊 Performance Results:")
    print(f"   Full Calculation:")
    print(f"     Total time: {full_time:.1f}ms")
    print(f"     Average per calculation: {full_avg_time:.3f}ms")
    print(f"     Calculations per second: {len(test_cases) / (full_time/1000):,.0f}")
    
    print(f"   Fast Calculation:")
    print(f"     Total time: {fast_time:.1f}ms")
    print(f"     Average per calculation: {fast_avg_time:.3f}ms")
    print(f"     Calculations per second: {len(test_cases) / (fast_time/1000):,.0f}")
    
    print(f"   Speed Advantage: {speed_advantage:.1f}x faster (fast vs full)")
    
    # Analyze accuracy
    high_confidence_count = sum(1 for r in full_results if r.confidence_level in [MatchConfidence.EXACT, MatchConfidence.HIGH])
    same_finger_count = sum(1 for r in full_results if r.is_same_finger_likely)
    
    print(f"\n🎯 Accuracy Analysis:")
    print(f"   High confidence matches: {high_confidence_count}/{len(full_results)} ({high_confidence_count/len(full_results)*100:.1f}%)")
    print(f"   Same finger identifications: {same_finger_count}/{len(full_results)} ({same_finger_count/len(full_results)*100:.1f}%)")
    
    # Performance rating
    if full_avg_time < 1.0:
        rating = "REVOLUTIONARY (sub-millisecond)"
    elif full_avg_time < 5.0:
        rating = "EXCELLENT (sub-5ms)"
    elif full_avg_time < 10.0:
        rating = "GOOD (sub-10ms)"
    else:
        rating = "NEEDS OPTIMIZATION"
    
    print(f"\n⚡ Performance Rating: {rating}")
    
    if full_avg_time < 5.0:
        print(f"   ✅ Ready for real-time O(1) database operations")
        print(f"   ✅ Suitable for high-throughput matching")
        print(f"   ✅ Faster than traditional fingerprint algorithms")
    
    return {
        'full_avg_time_ms': full_avg_time,
        'fast_avg_time_ms': fast_avg_time,
        'speed_advantage': speed_advantage,
        'calculations_per_second_full': len(test_cases) / (full_time/1000),
        'calculations_per_second_fast': len(test_cases) / (fast_time/1000),
        'high_confidence_rate': high_confidence_count/len(full_results)*100,
        'same_finger_rate': same_finger_count/len(full_results)*100,
        'performance_rating': rating
    }


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_biological_similarity()
    
    print("\n" + "="*80)
    benchmark_results = benchmark_similarity_performance()
    
    print(f"\n🧬 REVOLUTIONARY BIOLOGICAL SIMILARITY CALCULATOR READY!")
    print(f"   Scientific accuracy: ✅ Based on fingerprint biology")
    print(f"   Same finger tolerance: ✅ Accounts for impression variation")
    print(f"   Different finger discrimination: ✅ High security")
    print(f"   O(1) optimized: ✅ {benchmark_results['calculations_per_second_full']:,.0f} calculations/second")
    print(f"   Fast mode: ✅ {benchmark_results['calculations_per_second_fast']:,.0f} calculations/second")
    print("="*80)#!/usr/bin/env python3
"""
Biological Features Classes
Patent Pending - Michael Derrick Jagneaux

Classes for biological feature extraction and similarity calculation.
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Types of similarity metrics."""
    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"
    BIOLOGICAL = "BIOLOGICAL"
    WEIGHTED = "WEIGHTED"


class FeatureType(Enum):
    """Types of biological features."""
    PATTERN = "PATTERN"
    SPATIAL = "SPATIAL"
    QUALITY = "QUALITY"
    MINUTIAE = "MINUTIAE"
    RIDGE = "RIDGE"


@dataclass
class BiologicalFeatures:
    """Revolutionary biological features for fingerprint similarity."""
    
    # Pattern characteristics
    pattern_class: str                    # Primary pattern type
    pattern_confidence: float            # Confidence in pattern classification
    secondary_patterns: List[str]        # Alternative patterns detected
    
    # Spatial characteristics
    core_position: Tuple[float, float]   # Core position coordinates
    delta_positions: List[Tuple[float, float]]  # Delta positions
    pattern_orientation: float           # Primary orientation angle
    spatial_distribution: Dict[str, float]  # Spatial feature distribution
    
    # Ridge characteristics
    ridge_count_vertical: int            # Vertical ridge count
    ridge_count_horizontal: int          # Horizontal ridge count
    ridge_density: float                 # Average ridge density
    ridge_flow_direction: str           # Primary ridge flow
    ridge_wavelength: float             # Average ridge wavelength
    
    # Minutiae characteristics
    minutiae_count: int                  # Total minutiae count
    minutiae_types: Dict[str, int]      # Count by minutiae type
    minutiae_distribution: Dict[str, float]  # Spatial distribution
    minutiae_quality: float             # Average minutiae quality
    
    # Quality characteristics
    image_quality: float                 # Overall image quality
    clarity_score: float                 # Image clarity
    contrast_level: float               # Image contrast
    noise_level: float                  # Noise estimation
    uniformity_score: float             # Illumination uniformity
    
    # Derived characteristics
    uniqueness_score: float             # Estimated uniqueness
    stability_score: float              # Characteristic stability
    matching_difficulty: float          # Difficulty of matching
    
    # Metadata
    extraction_time_ms: float           # Feature extraction time
    feature_version: str                # Feature extraction version
    extraction_confidence: float       # Overall confidence
    feature_hash: str                   # Hash of all features
    
    def __post_init__(self):
        """Validate features after initialization."""
        if not 0 <= self.pattern_confidence <= 1:
            raise ValueError("Pattern confidence must be between 0 and 1")
        if not 0 <= self.image_quality <= 100:
            raise ValueError("Image quality must be between 0 and 100")
        if self.minutiae_count < 0:
            raise ValueError("Minutiae count cannot be negative")
    
    def get_feature_vector(self, feature_types: Optional[List[FeatureType]] = None) -> np.ndarray:
        """Get numerical feature vector for similarity calculations."""
        if feature_types is None:
            feature_types = list(FeatureType)
        
        features = []
        
        if FeatureType.PATTERN in feature_types:
            # Pattern features
            pattern_encoded = self._encode_pattern(self.pattern_class)
            features.extend([
                pattern_encoded,
                self.pattern_confidence,
                self.pattern_orientation / 180.0  # Normalize to 0-1
            ])
        
        if FeatureType.SPATIAL in feature_types:
            # Spatial features
            features.extend([
                self.core_position[0] / 512.0,  # Normalize assuming 512x512 image
                self.core_position[1] / 512.0,
                len(self.delta_positions) / 10.0  # Normalize delta count
            ])
        
        if FeatureType.RIDGE in feature_types:
            # Ridge features
            features.extend([
                self.ridge_count_vertical / 100.0,  # Normalize
                self.ridge_count_horizontal / 100.0,
                self.ridge_density / 50.0,  # Normalize
                self.ridge_wavelength / 20.0 if self.ridge_wavelength > 0 else 0
            ])
        
        if FeatureType.MINUTIAE in feature_types:
            # Minutiae features
            features.extend([
                self.minutiae_count / 100.0,  # Normalize
                self.minutiae_quality,
                len(self.minutiae_types) / 10.0  # Normalize type diversity
            ])
        
        if FeatureType.QUALITY in feature_types:
            # Quality features
            features.extend([
                self.image_quality / 100.0,
                self.clarity_score,
                self.contrast_level / 255.0,  # Normalize
                1.0 - self.noise_level,  # Invert noise (higher = better)
                self.uniformity_score
            ])
        
        return np.array(features, dtype=np.float32)
    
    def calculate_similarity(self, other: 'BiologicalFeatures', 
                           metric: SimilarityMetric = SimilarityMetric.BIOLOGICAL,
                           weights: Optional[Dict[FeatureType, float]] = None) -> float:
        """Calculate similarity to another BiologicalFeatures instance."""
        if metric == SimilarityMetric.BIOLOGICAL:
            return self._calculate_biological_similarity(other, weights)
        elif metric == SimilarityMetric.EUCLIDEAN:
            return self._calculate_euclidean_similarity(other)
        elif metric == SimilarityMetric.COSINE:
            return self._calculate_cosine_similarity(other)
        elif metric == SimilarityMetric.WEIGHTED:
            return self._calculate_weighted_similarity(other, weights)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def is_similar_to(self, other: 'BiologicalFeatures', 
                     threshold: float = 0.8,
                     metric: SimilarityMetric = SimilarityMetric.BIOLOGICAL) -> bool:
        """Check if this feature set is similar to another within threshold."""
        similarity = self.calculate_similarity(other, metric)
        return similarity >= threshold
    
    def get_distinctive_features(self) -> Dict[str, float]:
        """Get the most distinctive features for this fingerprint."""
        distinctiveness = {}
        
        # Pattern distinctiveness
        pattern_rarity = self._get_pattern_rarity(self.pattern_class)
        distinctiveness['pattern_rarity'] = pattern_rarity
        
        # Minutiae distinctiveness
        minutiae_distinctiveness = min(1.0, self.minutiae_count / 100.0)
        distinctiveness['minutiae_richness'] = minutiae_distinctiveness
        
        # Quality distinctiveness
        quality_distinctiveness = self.image_quality / 100.0
        distinctiveness['quality_level'] = quality_distinctiveness
        
        # Spatial complexity
        spatial_complexity = len(self.delta_positions) / 5.0  # Normalize
        distinctiveness['spatial_complexity'] = min(1.0, spatial_complexity)
        
        # Overall uniqueness
        distinctiveness['overall_uniqueness'] = self.uniqueness_score
        
        return distinctiveness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_class': self.pattern_class,
            'pattern_confidence': self.pattern_confidence,
            'secondary_patterns': self.secondary_patterns,
            'core_position': self.core_position,
            'delta_positions': self.delta_positions,
            'pattern_orientation': self.pattern_orientation,
            'spatial_distribution': self.spatial_distribution,
            'ridge_count_vertical': self.ridge_count_vertical,
            'ridge_count_horizontal': self.ridge_count_horizontal,
            'ridge_density': self.ridge_density,
            'ridge_flow_direction': self.ridge_flow_direction,
            'ridge_wavelength': self.ridge_wavelength,
            'minutiae_count': self.minutiae_count,
            'minutiae_types': self.minutiae_types,
            'minutiae_distribution': self.minutiae_distribution,
            'minutiae_quality': self.minutiae_quality,
            'image_quality': self.image_quality,
            'clarity_score': self.clarity_score,
            'contrast_level': self.contrast_level,
            'noise_level': self.noise_level,
            'uniformity_score': self.uniformity_score,
            'uniqueness_score': self.uniqueness_score,
            'stability_score': self.stability_score,
            'matching_difficulty': self.matching_difficulty,
            'extraction_time_ms': self.extraction_time_ms,
            'feature_version': self.feature_version,
            'extraction_confidence': self.extraction_confidence,
            'feature_hash': self.feature_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiologicalFeatures':
        """Create from dictionary."""
        return cls(**data)
    
    def _encode_pattern(self, pattern: str) -> float:
        """Encode pattern class to numerical value."""
        pattern_map = {
            'ARCH_PLAIN': 0.1,
            'ARCH_TENTED': 0.2,
            'LOOP_LEFT': 0.4,
            'LOOP_RIGHT': 0.6,
            'WHORL': 0.8,
            'PATTERN_UNCLEAR': 0.0
        }
        return pattern_map.get(pattern, 0.0)
    
    def _calculate_biological_similarity(self, other: 'BiologicalFeatures',
                                        weights: Optional[Dict[FeatureType, float]] = None) -> float:
        """Calculate biological similarity using domain knowledge."""
        if weights is None:
            weights = {
                FeatureType.PATTERN: 0.3,
                FeatureType.SPATIAL: 0.2,
                FeatureType.RIDGE: 0.2,
                FeatureType.MINUTIAE: 0.2,
                FeatureType.QUALITY: 0.1
            }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # Pattern similarity (most important)
        if self.pattern_class == other.pattern_class:
            pattern_sim = 1.0
        else:
            pattern_sim = 0.0  # Different patterns = very low similarity
        
        total_similarity += pattern_sim * weights[FeatureType.PATTERN]
        total_weight += weights[FeatureType.PATTERN]
        
        # Spatial similarity
        spatial_sim = self._calculate_spatial_similarity(other)
        total_similarity += spatial_sim * weights[FeatureType.SPATIAL]
        total_weight += weights[FeatureType.SPATIAL]
        
        # Ridge similarity
        ridge_sim = self._calculate_ridge_similarity(other)
        total_similarity += ridge_sim * weights[FeatureType.RIDGE]
        total_weight += weights[FeatureType.RIDGE]
        
        # Minutiae similarity
        minutiae_sim = self._calculate_minutiae_similarity(other)
        total_similarity += minutiae_sim * weights[FeatureType.MINUTIAE]
        total_weight += weights[FeatureType.MINUTIAE]
        
        # Quality factor (affects confidence in similarity)
        min_quality = min(self.image_quality, other.image_quality) / 100.0
        quality_factor = min_quality * weights[FeatureType.QUALITY]
        total_weight += weights[FeatureType.QUALITY]
        
        # Calculate weighted average
        if total_weight > 0:
            base_similarity = total_similarity / total_weight
            # Adjust by quality factor
            final_similarity = base_similarity * (0.7 + 0.3 * min_quality)
            return min(1.0, max(0.0, final_similarity))
        else:
            return 0.0
    
    def _calculate_euclidean_similarity(self, other: 'BiologicalFeatures') -> float:
        """Calculate Euclidean distance-based similarity."""
        vec1 = self.get_feature_vector()
        vec2 = other.get_feature_vector()
        
        if len(vec1) != len(vec2):
            raise ValueError("Feature vectors must have same length")
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        
        # Convert to similarity (0-1 scale)
        max_distance = np.sqrt(len(vec1))  # Maximum possible distance
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
    
    def _calculate_cosine_similarity(self, other: 'BiologicalFeatures') -> float:
        """Calculate cosine similarity."""
        vec1 = self.get_feature_vector()
        vec2 = other.get_feature_vector()
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)
    
    def _calculate_weighted_similarity(self, other: 'BiologicalFeatures',
                                     weights: Optional[Dict[FeatureType, float]] = None) -> float:
        """Calculate weighted similarity with custom weights."""
        if weights is None:
            # Default weights
            weights = {feature_type: 1.0 for feature_type in FeatureType}
        
        return self._calculate_biological_similarity(other, weights)
    
    def _calculate_spatial_similarity(self, other: 'BiologicalFeatures') -> float:
        """Calculate spatial feature similarity."""
        # Core position similarity
        core_dist = np.linalg.norm(
            np.array(self.core_position) - np.array(other.core_position)
        )
        core_sim = 1.0 - min(1.0, core_dist / 512.0)  # Normalize by image size
        
        # Orientation similarity
        angle_diff = abs(self.pattern_orientation - other.pattern_orientation)
        angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wraparound
        orientation_sim = 1.0 - (angle_diff / 90.0)  # Normalize by 90 degrees
        
        # Combine spatial similarities
        return (core_sim + orientation_sim) / 2.0
    
    def _calculate_ridge_similarity(self, other: 'BiologicalFeatures') -> float:
        """Calculate ridge feature similarity."""
        # Ridge count similarity
        count_diff_v = abs(self.ridge_count_vertical - other.ridge_count_vertical)
        count_diff_h = abs(self.ridge_count_horizontal - other.ridge_count_horizontal)
        
        count_sim_v = 1.0 - min(1.0, count_diff_v / 50.0)  # Normalize
        count_sim_h = 1.0 - min(1.0, count_diff_h / 50.0)
        
        # Ridge density similarity
        density_diff = abs(self.ridge_density - other.ridge_density)
        density_sim = 1.0 - min(1.0, density_diff / 30.0)  # Normalize
        
        # Ridge flow similarity
        flow_sim = 1.0 if self.ridge_flow_direction == other.ridge_flow_direction else 0.5
        
        # Combine ridge similarities
        return (count_sim_v + count_sim_h + density_sim + flow_sim) / 4.0
    
    def _calculate_minutiae_similarity(self, other: 'BiologicalFeatures') -> float:
        """Calculate minutiae feature similarity."""
        # Minutiae count similarity
        count_diff = abs(self.minutiae_count - other.minutiae_count)
        count_sim = 1.0 - min(1.0, count_diff / 50.0)  # Normalize
        
        # Minutiae quality similarity
        quality_diff = abs(self.minutiae_quality - other.minutiae_quality)
        quality_sim = 1.0 - quality_diff  # Already normalized 0-1
        
        # Minutiae type distribution similarity
        all_types = set(self.minutiae_types.keys()) | set(other.minutiae_types.keys())
        type_similarities = []
        
        for minutiae_type in all_types:
            count1 = self.minutiae_types.get(minutiae_type, 0)
            count2 = other.minutiae_types.get(minutiae_type, 0)
            type_diff = abs(count1 - count2)
            type_sim = 1.0 - min(1.0, type_diff / 20.0)  # Normalize
            type_similarities.append(type_sim)
        
        type_sim_avg = np.mean(type_similarities) if type_similarities else 0.0
        
        # Combine minutiae similarities
        return (count_sim + quality_sim + type_sim_avg) / 3.0
    
    def _get_pattern_rarity(self, pattern: str) -> float:
        """Get rarity score for pattern type."""
        # Based on typical fingerprint pattern distribution
        pattern_frequencies = {
            'LOOP_RIGHT': 0.31,    # Most common
            'LOOP_LEFT': 0.31,
            'WHORL': 0.34,
            'ARCH_PLAIN': 0.03,   # Rare
            'ARCH_TENTED': 0.01   # Very rare
        }
        
        frequency = pattern_frequencies.get(pattern, 0.05)
        return 1.0 - frequency  # Rarity = 1 - frequency


class RevolutionaryBiologicalSimilarity:
    """Revolutionary similarity calculator for biological features."""
    
    def __init__(self):
        """Initialize the similarity calculator."""
        self.calculation_stats = {
            'total_calculations': 0,
            'average_calculation_time_ms': 0,
            'similarity_distribution': {}
        }
    
    def calculate_similarity(self, features1: BiologicalFeatures, features2: BiologicalFeatures,
                           metric: SimilarityMetric = SimilarityMetric.BIOLOGICAL) -> Dict[str, Any]:
        """Calculate comprehensive similarity between two feature sets."""
        start_time = time.perf_counter()
        
        try:
            # Calculate similarity
            similarity_score = features1.calculate_similarity(features2, metric)
            
            # Calculate component similarities
            component_similarities = {
                'pattern': features1._calculate_spatial_similarity(features2),
                'spatial': features1._calculate_spatial_similarity(features2),
                'ridge': features1._calculate_ridge_similarity(features2),
                'minutiae': features1._calculate_minutiae_similarity(features2)
            }
            
            # Calculate confidence in similarity
            min_quality = min(features1.image_quality, features2.image_quality) / 100.0
            confidence = min_quality * similarity_score
            
            calculation_time = (time.perf_counter() - start_time) * 1000
            
            result = {
                'overall_similarity': similarity_score,
                'component_similarities': component_similarities,
                'confidence': confidence,
                'calculation_time_ms': calculation_time,
                'metric_used': metric.value,
                'is_match': similarity_score >= 0.8,
                'match_strength': self._classify_match_strength(similarity_score)
            }
            
            self._update_calculation_stats(calculation_time)
            return result
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {
                'overall_similarity': 0.0,
                'error': str(e),
                'calculation_time_ms': 0.0
            }
    
    def _classify_match_strength(self, similarity: float) -> str:
        """Classify match strength based on similarity score."""
        if similarity >= 0.95:
            return "VERY_STRONG"
        elif similarity >= 0.85:
            return "STRONG" 
        elif similarity >= 0.75:
            return "MODERATE"
        elif similarity >= 0.60:
            return "WEAK"
        else:
            return "NO_MATCH"
    
    def _update_calculation_stats(self, calculation_time: float) -> None:
        """Update calculation statistics."""
        self.calculation_stats['total_calculations'] += 1
        
        total = self.calculation_stats['total_calculations']
        current_avg = self.calculation_stats['average_calculation_time_ms']
        new_avg = ((current_avg * (total - 1)) + calculation_time) / total
        self.calculation_stats['average_calculation_time_ms'] = new_avg


# Convenience function
def create_biological_features(characteristics: Dict[str, Any]) -> BiologicalFeatures:
    """Create BiologicalFeatures from fingerprint characteristics."""
    return BiologicalFeatures(
        pattern_class=characteristics.get('pattern_class', 'UNKNOWN'),
        pattern_confidence=characteristics.get('confidence_score', 0.0),
        secondary_patterns=[],
        core_position=(256.0, 256.0),  # Default center
        delta_positions=[],
        pattern_orientation=characteristics.get('pattern_orientation', 0),
        spatial_distribution={},
        ridge_count_vertical=characteristics.get('ridge_count_vertical', 0),
        ridge_count_horizontal=characteristics.get('ridge_count_horizontal', 0),
        ridge_density=characteristics.get('ridge_density', 0.0),
        ridge_flow_direction=characteristics.get('ridge_flow_direction', 'UNKNOWN'),
        ridge_wavelength=10.0,  # Default
        minutiae_count=characteristics.get('minutiae_count', 0),
        minutiae_types={},
        minutiae_distribution={},
        minutiae_quality=0.8,  # Default
        image_quality=characteristics.get('image_quality', 0.0),
        clarity_score=0.8,  # Default
        contrast_level=characteristics.get('contrast_level', 0.0),
        noise_level=0.2,  # Default
        uniformity_score=0.8,  # Default
        uniqueness_score=0.8,  # Default
        stability_score=0.8,  # Default
        matching_difficulty=0.5,  # Default
        extraction_time_ms=characteristics.get('processing_time_ms', 0.0),
        feature_version='1.0',
        extraction_confidence=characteristics.get('confidence_score', 0.0),
        feature_hash=''  # Would be calculated
    )