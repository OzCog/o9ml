"""
Tests for Embodiment Integration

Comprehensive test suite for embodiment tensor processing,
Unity3D/ROS bridges, and sensory-motor feedback loops.
"""

import pytest
import numpy as np
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cogml.embodiment.embodiment_tensor import (
    EmbodimentTensorSignature, 
    EmbodimentTensorProcessor
)

class TestEmbodimentTensorSignature:
    """Test embodiment tensor signature validation and creation"""
    
    def test_valid_signature_creation(self):
        """Test creating a valid embodiment tensor signature"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "auditory"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        assert signature.sensory_modality == ["visual", "auditory"]
        assert signature.motor_command == [0.1, 0.2, 0.3]
        assert signature.action_confidence == 0.8
        assert signature.embodiment_state == "virtual"
    
    def test_invalid_sensory_modality(self):
        """Test validation of invalid sensory modality"""
        with pytest.raises(ValueError, match="Invalid sensory modalities"):
            EmbodimentTensorSignature(
                sensory_modality=["invalid_modality"],
                motor_command=[0.1, 0.2, 0.3],
                spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
                temporal_context=["present"],
                action_confidence=0.8,
                embodiment_state="virtual",
                interaction_mode="active",
                feedback_loop="closed"
            )
    
    def test_invalid_motor_command_dimensions(self):
        """Test validation of motor command dimensions"""
        with pytest.raises(ValueError, match="Motor command must have 3 dimensions"):
            EmbodimentTensorSignature(
                sensory_modality=["visual"],
                motor_command=[0.1, 0.2],  # Only 2 dimensions
                spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
                temporal_context=["present"],
                action_confidence=0.8,
                embodiment_state="virtual",
                interaction_mode="active",
                feedback_loop="closed"
            )
    
    def test_invalid_spatial_coordinates(self):
        """Test validation of spatial coordinates"""
        with pytest.raises(ValueError, match="Spatial coordinates must have 4 dimensions"):
            EmbodimentTensorSignature(
                sensory_modality=["visual"],
                motor_command=[0.1, 0.2, 0.3],
                spatial_coordinates=[1.0, 2.0, 3.0],  # Only 3 dimensions
                temporal_context=["present"],
                action_confidence=0.8,
                embodiment_state="virtual",
                interaction_mode="active",
                feedback_loop="closed"
            )
    
    def test_invalid_action_confidence_range(self):
        """Test validation of action confidence range"""
        with pytest.raises(ValueError, match="Action confidence must be in range"):
            EmbodimentTensorSignature(
                sensory_modality=["visual"],
                motor_command=[0.1, 0.2, 0.3],
                spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
                temporal_context=["present"],
                action_confidence=1.5,  # Out of range
                embodiment_state="virtual",
                interaction_mode="active",
                feedback_loop="closed"
            )
    
    def test_invalid_embodiment_state(self):
        """Test validation of embodiment state"""
        with pytest.raises(ValueError, match="Embodiment state must be one of"):
            EmbodimentTensorSignature(
                sensory_modality=["visual"],
                motor_command=[0.1, 0.2, 0.3],
                spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
                temporal_context=["present"],
                action_confidence=0.8,
                embodiment_state="invalid_state",
                interaction_mode="active",
                feedback_loop="closed"
            )
    
    def test_all_valid_modalities(self):
        """Test all valid sensory modalities"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "auditory", "tactile", "proprioceptive"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["past", "present", "future"],
            action_confidence=0.8,
            embodiment_state="hybrid",
            interaction_mode="adaptive",
            feedback_loop="predictive"
        )
        
        assert len(signature.sensory_modality) == 4
        assert len(signature.temporal_context) == 3

class TestEmbodimentTensorProcessor:
    """Test embodiment tensor processing functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.processor = EmbodimentTensorProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.attention_tensor_dims == 327
        assert self.processor.embodiment_tensor_dims == 37
        assert self.processor.total_dims == 364
    
    def test_create_embodiment_tensor(self):
        """Test embodiment tensor creation"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "auditory"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        tensor = self.processor.create_embodiment_tensor(signature)
        
        assert tensor.shape[0] == 37
        assert isinstance(tensor, np.ndarray)
        
        # Check motor actions (first 6 dimensions)
        assert tensor[0] == 0.1
        assert tensor[1] == 0.2
        assert tensor[2] == 0.3
        
        # Check sensory modalities encoding
        assert tensor[6] == 1.0  # visual
        assert tensor[7] == 1.0  # auditory
        assert tensor[8] == 0.0  # tactile (not active)
        
        # Check spatial coordinates
        assert tensor[14] == 1.0  # x
        assert tensor[15] == 2.0  # y
        assert tensor[16] == 3.0  # z
        assert tensor[17] == 0.5  # orientation
    
    def test_integrate_with_attention_tensor(self):
        """Test integration with attention tensor"""
        # Create mock attention tensor
        attention_vector = np.random.random(327)
        
        # Create embodiment tensor
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        embodiment_tensor = self.processor.create_embodiment_tensor(signature)
        
        # Integrate tensors
        unified_tensor = self.processor.integrate_with_attention_tensor(
            attention_vector, embodiment_tensor
        )
        
        assert unified_tensor.shape[0] == 364
        
        # Check that attention part is preserved
        np.testing.assert_array_equal(unified_tensor[:327], attention_vector)
        
        # Check that embodiment part is preserved
        np.testing.assert_array_equal(unified_tensor[327:], embodiment_tensor)
    
    def test_process_sensory_input(self):
        """Test processing raw sensory input"""
        sensory_data = {
            "visual_frames": [[255, 128, 64]],
            "audio_data": [0.1, 0.2, 0.3],
            "spatial_coords": [1.0, 2.0, 3.0, 0.5],
            "motor_command": [0.1, 0.2, 0.3],
            "embodiment_state": "physical",
            "temporal_context": ["present"],
            "action_confidence": 0.9,
            "interaction_mode": "adaptive",
            "feedback_loop": "closed"
        }
        
        signature = self.processor.process_sensory_input(sensory_data)
        
        assert "visual" in signature.sensory_modality
        assert "auditory" in signature.sensory_modality
        assert signature.embodiment_state == "physical"
        assert signature.action_confidence == 0.9
        assert signature.spatial_coordinates == [1.0, 2.0, 3.0, 0.5]
    
    def test_generate_motor_response(self):
        """Test motor response generation"""
        # Create unified tensor
        attention_vector = np.random.random(327)
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        embodiment_tensor = self.processor.create_embodiment_tensor(signature)
        unified_tensor = self.processor.integrate_with_attention_tensor(
            attention_vector, embodiment_tensor
        )
        
        # Generate motor response
        goal_state = {"target_position": [5.0, 6.0, 7.0]}
        motor_response = self.processor.generate_motor_response(unified_tensor, goal_state)
        
        assert "linear_velocity" in motor_response
        assert "angular_velocity" in motor_response
        assert "target_position" in motor_response
        assert "current_position" in motor_response
        assert "confidence" in motor_response
        assert "timestamp" in motor_response
        
        assert motor_response["target_position"] == [5.0, 6.0, 7.0]
        assert len(motor_response["linear_velocity"]) == 3
        assert len(motor_response["angular_velocity"]) == 3
    
    def test_validate_embodiment_dataflow(self):
        """Test embodiment dataflow validation"""
        result = self.processor.validate_embodiment_dataflow()
        assert result is True
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization"""
        original_signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "tactile"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present", "future"],
            action_confidence=0.8,
            embodiment_state="hybrid",
            interaction_mode="adaptive",
            feedback_loop="predictive"
        )
        
        # Convert to dict
        signature_dict = self.processor.to_dict(original_signature)
        
        assert "sensory_modality" in signature_dict
        assert "timestamp" in signature_dict
        assert signature_dict["action_confidence"] == 0.8
        
        # Convert back to signature (without timestamp)
        signature_dict.pop("timestamp", None)
        reconstructed_signature = self.processor.from_dict(signature_dict)
        
        assert reconstructed_signature.sensory_modality == original_signature.sensory_modality
        assert reconstructed_signature.motor_command == original_signature.motor_command
        assert reconstructed_signature.action_confidence == original_signature.action_confidence
        assert reconstructed_signature.embodiment_state == original_signature.embodiment_state
    
    def test_error_handling_invalid_tensor_dimensions(self):
        """Test error handling for invalid tensor dimensions"""
        # Wrong attention vector size
        wrong_attention = np.random.random(100)  # Should be 327
        embodiment_tensor = np.random.random(37)
        
        with pytest.raises(ValueError, match="Attention vector must be"):
            self.processor.integrate_with_attention_tensor(wrong_attention, embodiment_tensor)
        
        # Wrong embodiment tensor size
        correct_attention = np.random.random(327)
        wrong_embodiment = np.random.random(20)  # Should be 37
        
        with pytest.raises(ValueError, match="Embodiment tensor must be"):
            self.processor.integrate_with_attention_tensor(correct_attention, wrong_embodiment)

class TestEmbodimentTensorDimensions:
    """Test specific embodiment tensor dimension mappings"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.processor = EmbodimentTensorProcessor()
    
    def test_motor_actions_mapping(self):
        """Test motor actions dimension mapping (6D)"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        tensor = self.processor.create_embodiment_tensor(signature)
        
        # Motor actions: dimensions 0-5
        motor_actions = tensor[0:6]
        assert motor_actions[0] == 0.1  # linear x
        assert motor_actions[1] == 0.2  # linear y
        assert motor_actions[2] == 0.3  # linear z
        assert motor_actions[3] == 0.0  # angular x (padded)
        assert motor_actions[4] == 0.0  # angular y (padded)
        assert motor_actions[5] == 0.0  # angular z (padded)
    
    def test_sensory_modalities_mapping(self):
        """Test sensory modalities dimension mapping (8D)"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "tactile"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        tensor = self.processor.create_embodiment_tensor(signature)
        
        # Sensory modalities: dimensions 6-13
        sensory_encoding = tensor[6:14]
        assert sensory_encoding[0] == 1.0  # visual
        assert sensory_encoding[1] == 0.0  # auditory
        assert sensory_encoding[2] == 1.0  # tactile
        assert sensory_encoding[3] == 0.0  # proprioceptive
        assert sensory_encoding[4] == 0.0  # unused
        assert sensory_encoding[5] == 0.0  # unused
        assert sensory_encoding[6] == 0.0  # unused
        assert sensory_encoding[7] == 0.0  # unused
    
    def test_embodied_state_mapping(self):
        """Test embodied state dimension mapping (4D)"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.5, 2.5, 3.5, 0.75],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        tensor = self.processor.create_embodiment_tensor(signature)
        
        # Embodied state: dimensions 14-17
        embodied_state = tensor[14:18]
        assert embodied_state[0] == 1.5  # x
        assert embodied_state[1] == 2.5  # y
        assert embodied_state[2] == 3.5  # z
        assert embodied_state[3] == 0.75  # orientation
    
    def test_temporal_context_mapping(self):
        """Test temporal context dimension mapping (3D)"""
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["past", "future"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        tensor = self.processor.create_embodiment_tensor(signature)
        
        # Temporal context: dimensions 34-36
        temporal_encoding = tensor[34:37]
        assert temporal_encoding[0] == 1.0  # past
        assert temporal_encoding[1] == 0.0  # present
        assert temporal_encoding[2] == 1.0  # future

class TestRealTimeProcessing:
    """Test real-time processing capabilities"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.processor = EmbodimentTensorProcessor()
    
    def test_processing_speed(self):
        """Test processing speed for real-time requirements"""
        import time
        
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual", "auditory"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        attention_vector = np.random.random(327)
        
        # Measure processing time
        start_time = time.time()
        
        for _ in range(100):  # Process 100 tensors
            embodiment_tensor = self.processor.create_embodiment_tensor(signature)
            unified_tensor = self.processor.integrate_with_attention_tensor(
                attention_vector, embodiment_tensor
            )
            motor_response = self.processor.generate_motor_response(
                unified_tensor, {"target_position": [1.0, 2.0, 3.0]}
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 100 tensors in less than 1 second for real-time performance
        assert total_time < 1.0
        
        # Calculate processing rate
        processing_rate = 100 / total_time
        print(f"Processing rate: {processing_rate:.2f} tensors/second")
        
        # Should handle at least 100 Hz for real-time robotics
        assert processing_rate > 100
    
    def test_memory_efficiency(self):
        """Test memory efficiency for continuous processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        signature = EmbodimentTensorSignature(
            sensory_modality=["visual"],
            motor_command=[0.1, 0.2, 0.3],
            spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
            temporal_context=["present"],
            action_confidence=0.8,
            embodiment_state="virtual",
            interaction_mode="active",
            feedback_loop="closed"
        )
        
        # Process many tensors
        for i in range(1000):
            attention_vector = np.random.random(327)
            embodiment_tensor = self.processor.create_embodiment_tensor(signature)
            unified_tensor = self.processor.integrate_with_attention_tensor(
                attention_vector, embodiment_tensor
            )
            
            # Force garbage collection periodically
            if i % 100 == 0:
                import gc
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"])