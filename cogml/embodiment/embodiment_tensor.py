"""
Embodiment Tensor Implementation

Implements the 8-dimensional embodiment tensor signature for distributed
cognitive mesh processing and action-perception loop integration.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbodimentTensorSignature:
    """
    Embodiment Tensor Signature as specified in the requirements:
    
    Embodiment_Tensor[8] = {
        sensory_modality: [visual, auditory, tactile, proprioceptive],
        motor_command: [position, velocity, force],
        spatial_coordinates: [x, y, z, orientation],
        temporal_context: [past, present, future],
        action_confidence: [0.0, 1.0],
        embodiment_state: [virtual, physical, hybrid],
        interaction_mode: [passive, active, adaptive],
        feedback_loop: [open, closed, predictive]
    }
    """
    sensory_modality: List[str]
    motor_command: List[float]
    spatial_coordinates: List[float]
    temporal_context: List[str]
    action_confidence: float
    embodiment_state: str
    interaction_mode: str
    feedback_loop: str
    
    def __post_init__(self):
        """Validate tensor signature constraints"""
        # Validate sensory modalities
        valid_modalities = {"visual", "auditory", "tactile", "proprioceptive"}
        if not all(mod in valid_modalities for mod in self.sensory_modality):
            raise ValueError(f"Invalid sensory modalities. Must be subset of {valid_modalities}")
        
        # Validate motor command dimensions (position, velocity, force)
        if len(self.motor_command) != 3:
            raise ValueError("Motor command must have 3 dimensions: [position, velocity, force]")
        
        # Validate spatial coordinates (x, y, z, orientation)
        if len(self.spatial_coordinates) != 4:
            raise ValueError("Spatial coordinates must have 4 dimensions: [x, y, z, orientation]")
        
        # Validate temporal context
        valid_temporal = {"past", "present", "future"}
        if not all(ctx in valid_temporal for ctx in self.temporal_context):
            raise ValueError(f"Invalid temporal context. Must be subset of {valid_temporal}")
        
        # Validate action confidence range
        if not 0.0 <= self.action_confidence <= 1.0:
            raise ValueError("Action confidence must be in range [0.0, 1.0]")
        
        # Validate embodiment state
        valid_states = {"virtual", "physical", "hybrid"}
        if self.embodiment_state not in valid_states:
            raise ValueError(f"Embodiment state must be one of {valid_states}")
        
        # Validate interaction mode
        valid_modes = {"passive", "active", "adaptive"}
        if self.interaction_mode not in valid_modes:
            raise ValueError(f"Interaction mode must be one of {valid_modes}")
        
        # Validate feedback loop
        valid_loops = {"open", "closed", "predictive"}
        if self.feedback_loop not in valid_loops:
            raise ValueError(f"Feedback loop must be one of {valid_loops}")

class EmbodimentTensorProcessor:
    """
    Processes embodiment tensors for distributed cognitive mesh integration.
    Extends the existing attention tensor (327D) with embodiment dimensions (37D).
    """
    
    def __init__(self):
        self.attention_tensor_dims = 327  # Existing attention tensor dimensions
        self.embodiment_tensor_dims = 37  # New embodiment dimensions
        self.total_dims = self.attention_tensor_dims + self.embodiment_tensor_dims
        
        logger.info(f"Initialized EmbodimentTensorProcessor: {self.total_dims}D total")
        
    def create_embodiment_tensor(self, signature: EmbodimentTensorSignature) -> np.ndarray:
        """
        Create a 37-dimensional embodiment tensor from the signature.
        
        Dimension mapping:
        - Motor Actions: 6D (linear: x,y,z + angular: roll,pitch,yaw)
        - Sensory Modalities: 8D (vision, audio, touch, proprioception, etc.)
        - Embodied State: 4D (position, orientation, velocity, acceleration)
        - Action Affordances: 16D (possible actions in current context)
        - Temporal Context: 3D (past, present, future weights)
        """
        tensor = np.zeros(self.embodiment_tensor_dims)
        
        # Motor Actions (6D) - extend motor_command to 6D
        motor_actions = np.array(signature.motor_command + [0.0, 0.0, 0.0])[:6]
        tensor[0:6] = motor_actions
        
        # Sensory Modalities (8D) - encode active modalities
        sensory_encoding = np.zeros(8)
        modality_map = {"visual": 0, "auditory": 1, "tactile": 2, "proprioceptive": 3}
        for modality in signature.sensory_modality:
            if modality in modality_map:
                sensory_encoding[modality_map[modality]] = 1.0
        tensor[6:14] = sensory_encoding
        
        # Embodied State (4D) - spatial coordinates
        embodied_state = np.array(signature.spatial_coordinates)
        tensor[14:18] = embodied_state
        
        # Action Affordances (16D) - context-dependent action possibilities
        # For now, encode based on interaction mode and confidence
        affordances = np.zeros(16)
        mode_map = {"passive": 0.2, "active": 0.8, "adaptive": 0.6}
        base_affordance = mode_map.get(signature.interaction_mode, 0.5)
        affordances[:8] = base_affordance * signature.action_confidence
        tensor[18:34] = affordances
        
        # Temporal Context (3D)
        temporal_encoding = np.zeros(3)
        temporal_map = {"past": 0, "present": 1, "future": 2}
        for context in signature.temporal_context:
            if context in temporal_map:
                temporal_encoding[temporal_map[context]] = 1.0
        tensor[34:37] = temporal_encoding
        
        return tensor
    
    def integrate_with_attention_tensor(self, attention_vector: np.ndarray, 
                                      embodiment_tensor: np.ndarray) -> np.ndarray:
        """
        Integrate embodiment tensor with existing attention tensor.
        Creates a 364-dimensional unified cognitive-embodiment tensor.
        """
        if attention_vector.shape[0] != self.attention_tensor_dims:
            raise ValueError(f"Attention vector must be {self.attention_tensor_dims}D")
        
        if embodiment_tensor.shape[0] != self.embodiment_tensor_dims:
            raise ValueError(f"Embodiment tensor must be {self.embodiment_tensor_dims}D")
        
        # Concatenate attention and embodiment tensors
        unified_tensor = np.concatenate([attention_vector, embodiment_tensor])
        
        logger.debug(f"Created unified tensor: {unified_tensor.shape[0]}D")
        return unified_tensor
    
    def process_sensory_input(self, sensory_data: Dict[str, Any]) -> EmbodimentTensorSignature:
        """
        Process raw sensory input into embodiment tensor signature.
        """
        # Extract sensory modalities from input
        sensory_modality = []
        if sensory_data.get("visual_frames"):
            sensory_modality.append("visual")
        if sensory_data.get("audio_data"):
            sensory_modality.append("auditory")
        if sensory_data.get("tactile_data"):
            sensory_modality.append("tactile")
        if sensory_data.get("proprioceptive_data"):
            sensory_modality.append("proprioceptive")
        
        # Extract spatial coordinates
        spatial_coords = sensory_data.get("spatial_coords", [0.0, 0.0, 0.0, 0.0])
        
        # Determine embodiment state
        embodiment_state = sensory_data.get("embodiment_state", "virtual")
        
        # Create tensor signature
        signature = EmbodimentTensorSignature(
            sensory_modality=sensory_modality,
            motor_command=sensory_data.get("motor_command", [0.0, 0.0, 0.0]),
            spatial_coordinates=spatial_coords,
            temporal_context=sensory_data.get("temporal_context", ["present"]),
            action_confidence=sensory_data.get("action_confidence", 0.5),
            embodiment_state=embodiment_state,
            interaction_mode=sensory_data.get("interaction_mode", "adaptive"),
            feedback_loop=sensory_data.get("feedback_loop", "closed")
        )
        
        return signature
    
    def generate_motor_response(self, unified_tensor: np.ndarray, 
                              goal_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate motor response from unified cognitive-embodiment tensor.
        """
        # Extract embodiment portion of unified tensor
        embodiment_portion = unified_tensor[self.attention_tensor_dims:]
        
        # Extract motor actions (first 6 dimensions)
        motor_actions = embodiment_portion[0:6]
        
        # Extract spatial state (dimensions 14-18)
        spatial_state = embodiment_portion[14:18]
        
        # Generate motor response based on goal and current state
        motor_response = {
            "linear_velocity": motor_actions[0:3].tolist(),
            "angular_velocity": motor_actions[3:6].tolist(),
            "target_position": goal_state.get("target_position", [0.0, 0.0, 0.0]),
            "current_position": spatial_state[0:3].tolist(),
            "orientation": spatial_state[3],
            "confidence": float(np.mean(embodiment_portion[18:34])),  # affordances
            "timestamp": datetime.now().isoformat()
        }
        
        return motor_response
    
    def validate_embodiment_dataflow(self) -> bool:
        """
        Validate the embodiment tensor processing dataflow.
        """
        try:
            # Create test signature
            test_signature = EmbodimentTensorSignature(
                sensory_modality=["visual", "auditory"],
                motor_command=[0.1, 0.2, 0.3],
                spatial_coordinates=[1.0, 2.0, 3.0, 0.5],
                temporal_context=["present"],
                action_confidence=0.8,
                embodiment_state="virtual",
                interaction_mode="active",
                feedback_loop="closed"
            )
            
            # Create embodiment tensor
            embodiment_tensor = self.create_embodiment_tensor(test_signature)
            
            # Create mock attention tensor
            attention_tensor = np.random.random(self.attention_tensor_dims)
            
            # Integrate tensors
            unified_tensor = self.integrate_with_attention_tensor(
                attention_tensor, embodiment_tensor
            )
            
            # Generate motor response
            motor_response = self.generate_motor_response(
                unified_tensor, {"target_position": [5.0, 6.0, 7.0]}
            )
            
            # Validate dimensions and structure
            assert embodiment_tensor.shape[0] == self.embodiment_tensor_dims
            assert unified_tensor.shape[0] == self.total_dims
            assert "linear_velocity" in motor_response
            assert "confidence" in motor_response
            
            logger.info("Embodiment dataflow validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Embodiment dataflow validation failed: {e}")
            return False
    
    def to_dict(self, signature: EmbodimentTensorSignature) -> Dict[str, Any]:
        """Convert embodiment tensor signature to dictionary"""
        return {
            "sensory_modality": signature.sensory_modality,
            "motor_command": signature.motor_command,
            "spatial_coordinates": signature.spatial_coordinates,
            "temporal_context": signature.temporal_context,
            "action_confidence": signature.action_confidence,
            "embodiment_state": signature.embodiment_state,
            "interaction_mode": signature.interaction_mode,
            "feedback_loop": signature.feedback_loop,
            "timestamp": datetime.now().isoformat()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> EmbodimentTensorSignature:
        """Create embodiment tensor signature from dictionary"""
        return EmbodimentTensorSignature(
            sensory_modality=data["sensory_modality"],
            motor_command=data["motor_command"],
            spatial_coordinates=data["spatial_coordinates"],
            temporal_context=data["temporal_context"],
            action_confidence=data["action_confidence"],
            embodiment_state=data["embodiment_state"],
            interaction_mode=data["interaction_mode"],
            feedback_loop=data["feedback_loop"]
        )