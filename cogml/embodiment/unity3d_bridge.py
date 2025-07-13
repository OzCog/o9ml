"""
Unity3D Bridge for OpenCog Central

Provides integration interface for Unity3D embodied agents, enabling
bi-directional communication between Unity3D games and the cognitive mesh.
"""

import json
import asyncio
import websockets
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)

class Unity3DBridge:
    """
    Bridge interface for Unity3D cognitive integration.
    Enables Unity3D games to connect to the distributed cognitive mesh.
    """
    
    def __init__(self, cognitive_mesh_url: str = "ws://localhost:8000/api/v1/ws/cognitive-stream"):
        self.cognitive_mesh_url = cognitive_mesh_url
        self.unity_websocket = None
        self.cognitive_websocket = None
        self.is_running = False
        self.message_queue = queue.Queue()
        self.callbacks: Dict[str, Callable] = {}
        
    async def connect_to_cognitive_mesh(self):
        """Connect to the cognitive mesh WebSocket"""
        try:
            self.cognitive_websocket = await websockets.connect(self.cognitive_mesh_url)
            logger.info(f"Connected to cognitive mesh at {self.cognitive_mesh_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to cognitive mesh: {e}")
            return False
    
    async def start_unity_server(self, port: int = 8001):
        """Start WebSocket server for Unity3D connections"""
        async def handle_unity_client(websocket, path):
            self.unity_websocket = websocket
            logger.info("Unity3D client connected")
            
            try:
                async for message in websocket:
                    await self.handle_unity_message(json.loads(message))
            except websockets.exceptions.ConnectionClosed:
                logger.info("Unity3D client disconnected")
            except Exception as e:
                logger.error(f"Error handling Unity3D message: {e}")
            finally:
                self.unity_websocket = None
        
        start_server = websockets.serve(handle_unity_client, "localhost", port)
        logger.info(f"Unity3D bridge server started on port {port}")
        await start_server
    
    async def handle_unity_message(self, message: Dict[str, Any]):
        """Handle incoming messages from Unity3D"""
        message_type = message.get("type")
        
        if message_type == "cognitive_update":
            # Forward cognitive update to mesh
            await self.send_to_cognitive_mesh(message)
            
        elif message_type == "sensory_input":
            # Process sensory input from Unity3D
            await self.process_unity_sensory_input(message)
            
        elif message_type == "motor_request":
            # Handle motor command request
            await self.process_unity_motor_request(message)
            
        elif message_type == "register_callback":
            # Register callback for specific events
            callback_name = message.get("callback_name")
            self.callbacks[callback_name] = message.get("endpoint")
            
        # Trigger custom callbacks
        if message_type in self.callbacks:
            await self.trigger_callback(message_type, message)
    
    async def process_unity_sensory_input(self, message: Dict[str, Any]):
        """Process sensory input from Unity3D environment"""
        sensory_data = message.get("data", {})
        
        # Convert Unity3D format to embodiment tensor format
        embodiment_data = {
            "agent_id": f"unity3d_{message.get('agent_id', 'default')}",
            "sensory_modality": [],
            "motor_command": sensory_data.get("motor_command", [0.0, 0.0, 0.0]),
            "spatial_coordinates": sensory_data.get("position", [0.0, 0.0, 0.0]) + 
                                 [sensory_data.get("rotation", 0.0)],
            "temporal_context": ["present"],
            "action_confidence": sensory_data.get("confidence", 0.5),
            "embodiment_state": "virtual",
            "interaction_mode": sensory_data.get("interaction_mode", "active"),
            "feedback_loop": "closed"
        }
        
        # Detect active sensory modalities
        if sensory_data.get("visual_data"):
            embodiment_data["sensory_modality"].append("visual")
        if sensory_data.get("audio_data"):
            embodiment_data["sensory_modality"].append("auditory")
        if sensory_data.get("collision_data"):
            embodiment_data["sensory_modality"].append("tactile")
        
        # Send to cognitive mesh
        cognitive_message = {
            "type": "embodiment_update",
            "agent_id": embodiment_data["agent_id"],
            "data": embodiment_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_cognitive_mesh(cognitive_message)
    
    async def process_unity_motor_request(self, message: Dict[str, Any]):
        """Process motor command request from Unity3D"""
        task_data = {
            "task_id": f"unity_motor_{message.get('request_id', 'default')}",
            "task_type": "motor_control",
            "parameters": message.get("parameters", {}),
            "priority": message.get("priority", 1)
        }
        
        # Send task to cognitive mesh
        cognitive_message = {
            "type": "task_request",
            "data": task_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_cognitive_mesh(cognitive_message)
    
    async def send_to_cognitive_mesh(self, message: Dict[str, Any]):
        """Send message to cognitive mesh"""
        if self.cognitive_websocket:
            try:
                await self.cognitive_websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to cognitive mesh: {e}")
    
    async def send_to_unity(self, message: Dict[str, Any]):
        """Send message to Unity3D client"""
        if self.unity_websocket:
            try:
                await self.unity_websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to Unity3D: {e}")
    
    async def handle_cognitive_mesh_message(self, message: Dict[str, Any]):
        """Handle incoming messages from cognitive mesh"""
        message_type = message.get("type")
        
        if message_type == "motor_response":
            # Forward motor response to Unity3D
            unity_message = {
                "type": "motor_command",
                "data": message.get("data", {}),
                "timestamp": datetime.now().isoformat()
            }
            await self.send_to_unity(unity_message)
            
        elif message_type == "cognitive_state_update":
            # Forward cognitive state to Unity3D for visualization
            unity_message = {
                "type": "cognitive_update",
                "agent_id": message.get("agent_id"),
                "state": message.get("state", {}),
                "timestamp": datetime.now().isoformat()
            }
            await self.send_to_unity(unity_message)
    
    async def trigger_callback(self, callback_name: str, message: Dict[str, Any]):
        """Trigger registered callback"""
        if callback_name in self.callbacks:
            callback_data = {
                "callback": callback_name,
                "data": message,
                "timestamp": datetime.now().isoformat()
            }
            await self.send_to_unity(callback_data)
    
    async def listen_to_cognitive_mesh(self):
        """Listen for messages from cognitive mesh"""
        if not self.cognitive_websocket:
            return
        
        try:
            async for message in self.cognitive_websocket:
                data = json.loads(message)
                await self.handle_cognitive_mesh_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Cognitive mesh connection closed")
        except Exception as e:
            logger.error(f"Error listening to cognitive mesh: {e}")
    
    async def run(self, unity_port: int = 8001):
        """Run the Unity3D bridge"""
        self.is_running = True
        
        # Connect to cognitive mesh
        if not await self.connect_to_cognitive_mesh():
            logger.error("Failed to connect to cognitive mesh")
            return
        
        # Start tasks concurrently
        await asyncio.gather(
            self.start_unity_server(unity_port),
            self.listen_to_cognitive_mesh()
        )
    
    def stop(self):
        """Stop the Unity3D bridge"""
        self.is_running = False
        if self.cognitive_websocket:
            asyncio.create_task(self.cognitive_websocket.close())
        if self.unity_websocket:
            asyncio.create_task(self.unity_websocket.close())

class Unity3DGameCharacter:
    """
    Represents a Unity3D game character with cognitive integration.
    Provides high-level interface for Unity3D developers.
    """
    
    def __init__(self, character_id: str, bridge: Unity3DBridge):
        self.character_id = character_id
        self.bridge = bridge
        self.position = [0.0, 0.0, 0.0]
        self.rotation = 0.0
        self.state = {}
        
    async def update_position(self, x: float, y: float, z: float, rotation: float = None):
        """Update character position and rotation"""
        self.position = [x, y, z]
        if rotation is not None:
            self.rotation = rotation
        
        # Send sensory input to cognitive mesh
        message = {
            "type": "sensory_input",
            "agent_id": self.character_id,
            "data": {
                "position": self.position,
                "rotation": self.rotation,
                "confidence": 1.0,
                "interaction_mode": "active"
            }
        }
        
        await self.bridge.handle_unity_message(message)
    
    async def request_action(self, action_type: str, parameters: Dict[str, Any]):
        """Request cognitive action"""
        message = {
            "type": "motor_request",
            "agent_id": self.character_id,
            "request_id": f"{self.character_id}_{action_type}_{datetime.now().timestamp()}",
            "parameters": {
                "action_type": action_type,
                **parameters
            }
        }
        
        await self.bridge.handle_unity_message(message)
    
    async def send_sensory_data(self, visual_data: Any = None, audio_data: Any = None, 
                               collision_data: Any = None):
        """Send sensory data to cognitive mesh"""
        sensory_data = {
            "position": self.position,
            "rotation": self.rotation
        }
        
        if visual_data is not None:
            sensory_data["visual_data"] = visual_data
        if audio_data is not None:
            sensory_data["audio_data"] = audio_data
        if collision_data is not None:
            sensory_data["collision_data"] = collision_data
        
        message = {
            "type": "sensory_input",
            "agent_id": self.character_id,
            "data": sensory_data
        }
        
        await self.bridge.handle_unity_message(message)

# Example Unity3D integration script (C# equivalent in comments)
UNITY_INTEGRATION_EXAMPLE = '''
/*
Unity3D C# Integration Example:

using System.Collections;
using UnityEngine;
using WebSocketSharp;

public class OpenCogBridge : MonoBehaviour 
{
    private WebSocket ws;
    
    void Start() 
    {
        ws = new WebSocket("ws://localhost:8001");
        ws.OnMessage += OnMessage;
        ws.Connect();
    }
    
    void OnMessage(object sender, MessageEventArgs e) 
    {
        var message = JsonUtility.FromJson<CognitiveMessage>(e.Data);
        
        if (message.type == "motor_command") 
        {
            // Apply motor command to character
            ApplyMotorCommand(message.data);
        }
    }
    
    void Update() 
    {
        // Send sensory data every frame
        var sensoryData = new {
            type = "sensory_input",
            agent_id = "unity_character_1",
            data = new {
                position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                rotation = transform.rotation.eulerAngles.y,
                visual_data = GetVisualData(),
                audio_data = GetAudioData()
            }
        };
        
        ws.Send(JsonUtility.ToJson(sensoryData));
    }
}
*/
'''

if __name__ == "__main__":
    # Example usage
    async def main():
        bridge = Unity3DBridge()
        await bridge.run()
    
    asyncio.run(main())