"""
WebSocket Real-time Cognitive Streams

Provides real-time bi-directional communication streams for embodied agents,
enabling low-latency cognitive state synchronization and feedback loops.
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class StreamMessage:
    """Standard message format for cognitive streams"""
    message_id: str
    message_type: str
    timestamp: datetime
    agent_id: str
    data: Dict[str, Any]
    priority: int = 1
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        message_dict = asdict(self)
        message_dict['timestamp'] = self.timestamp.isoformat()
        return json.dumps(message_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class CognitiveStreamHandler:
    """Handles real-time cognitive data streams"""
    
    def __init__(self):
        self.active_streams: Dict[str, 'CognitiveStream'] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.latency_monitor = LatencyMonitor()
        self.is_running = False
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def handle_message(self, stream_id: str, message: StreamMessage):
        """Handle incoming stream message"""
        # Record latency
        self.latency_monitor.record_message(message)
        
        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(stream_id, message)
            except Exception as e:
                logger.error(f"Error handling message {message.message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    def create_stream(self, stream_id: str, websocket) -> 'CognitiveStream':
        """Create new cognitive stream"""
        stream = CognitiveStream(stream_id, websocket, self)
        self.active_streams[stream_id] = stream
        logger.info(f"Created cognitive stream: {stream_id}")
        return stream
    
    def remove_stream(self, stream_id: str):
        """Remove cognitive stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            logger.info(f"Removed cognitive stream: {stream_id}")
    
    async def broadcast_to_all(self, message: StreamMessage, exclude_stream: str = None):
        """Broadcast message to all active streams"""
        tasks = []
        for stream_id, stream in self.active_streams.items():
            if stream_id != exclude_stream:
                tasks.append(stream.send_message(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_to_agents(self, agent_ids: List[str], message: StreamMessage):
        """Broadcast message to specific agents"""
        tasks = []
        for stream in self.active_streams.values():
            if stream.agent_id in agent_ids:
                tasks.append(stream.send_message(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about active streams"""
        return {
            "active_streams": len(self.active_streams),
            "stream_ids": list(self.active_streams.keys()),
            "latency_stats": self.latency_monitor.get_stats(),
            "message_types": list(self.message_handlers.keys())
        }

class CognitiveStream:
    """Individual cognitive stream for an agent"""
    
    def __init__(self, stream_id: str, websocket, handler: CognitiveStreamHandler):
        self.stream_id = stream_id
        self.websocket = websocket
        self.handler = handler
        self.agent_id: Optional[str] = None
        self.is_active = True
        self.message_queue = asyncio.Queue()
        self.send_lock = asyncio.Lock()
        self.last_ping = time.time()
        self.metrics = StreamMetrics()
        
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while self.is_active:
                try:
                    # Set timeout for receiving messages
                    message_data = await asyncio.wait_for(
                        self.websocket.recv(), timeout=30.0
                    )
                    
                    # Parse message
                    message = StreamMessage.from_json(message_data)
                    
                    # Set agent_id if not set
                    if not self.agent_id:
                        self.agent_id = message.agent_id
                    
                    # Update metrics
                    self.metrics.record_received_message(message)
                    
                    # Handle special message types
                    if message.message_type == "ping":
                        await self.send_pong()
                    elif message.message_type == "disconnect":
                        self.is_active = False
                        break
                    else:
                        # Forward to handler
                        await self.handler.handle_message(self.stream_id, message)
                    
                except asyncio.TimeoutError:
                    # Send ping to check connection
                    await self.send_ping()
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Stream {self.stream_id} connection closed")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in stream {self.stream_id}: {e}")
                    break
                    
        finally:
            self.is_active = False
            self.handler.remove_stream(self.stream_id)
    
    async def send_message(self, message: StreamMessage):
        """Send message to client"""
        if not self.is_active:
            return False
        
        async with self.send_lock:
            try:
                await self.websocket.send(message.to_json())
                self.metrics.record_sent_message(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to {self.stream_id}: {e}")
                self.is_active = False
                return False
    
    async def send_ping(self):
        """Send ping message"""
        ping_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="ping",
            timestamp=datetime.now(),
            agent_id="system",
            data={"timestamp": time.time()}
        )
        await self.send_message(ping_message)
        self.last_ping = time.time()
    
    async def send_pong(self):
        """Send pong response"""
        pong_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="pong",
            timestamp=datetime.now(),
            agent_id="system",
            data={"timestamp": time.time()}
        )
        await self.send_message(pong_message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics"""
        return {
            "stream_id": self.stream_id,
            "agent_id": self.agent_id,
            "is_active": self.is_active,
            "last_ping": self.last_ping,
            **self.metrics.get_stats()
        }

class StreamMetrics:
    """Track metrics for a cognitive stream"""
    
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.start_time = time.time()
        self.last_activity = time.time()
        self.message_types_sent = {}
        self.message_types_received = {}
    
    def record_sent_message(self, message: StreamMessage):
        """Record sent message metrics"""
        self.messages_sent += 1
        self.last_activity = time.time()
        
        message_json = message.to_json()
        self.bytes_sent += len(message_json.encode('utf-8'))
        
        msg_type = message.message_type
        self.message_types_sent[msg_type] = self.message_types_sent.get(msg_type, 0) + 1
    
    def record_received_message(self, message: StreamMessage):
        """Record received message metrics"""
        self.messages_received += 1
        self.last_activity = time.time()
        
        message_json = message.to_json()
        self.bytes_received += len(message_json.encode('utf-8'))
        
        msg_type = message.message_type
        self.message_types_received[msg_type] = self.message_types_received.get(msg_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics statistics"""
        uptime = time.time() - self.start_time
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "uptime_seconds": uptime,
            "last_activity": self.last_activity,
            "message_types_sent": self.message_types_sent,
            "message_types_received": self.message_types_received
        }

class LatencyMonitor:
    """Monitor message latency for performance analysis"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.latencies = []
        self.message_timestamps = {}
        
    def record_message(self, message: StreamMessage):
        """Record message for latency tracking"""
        current_time = time.time()
        
        # Calculate latency if this is a response
        if hasattr(message, 'request_id') and message.request_id in self.message_timestamps:
            request_time = self.message_timestamps[message.request_id]
            latency = current_time - request_time
            self.latencies.append(latency)
            
            # Keep only recent samples
            if len(self.latencies) > self.max_samples:
                self.latencies = self.latencies[-self.max_samples:]
            
            del self.message_timestamps[message.request_id]
        
        # Store timestamp for request messages
        if message.message_type.endswith('_request'):
            self.message_timestamps[message.message_id] = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get latency statistics"""
        if not self.latencies:
            return {
                "sample_count": 0,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0
            }
        
        import numpy as np
        latencies_ms = [l * 1000 for l in self.latencies]  # Convert to milliseconds
        
        return {
            "sample_count": len(latencies_ms),
            "avg_latency_ms": np.mean(latencies_ms),
            "min_latency_ms": np.min(latencies_ms),
            "max_latency_ms": np.max(latencies_ms),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99)
        }

class RealTimeEmbodimentStream:
    """Specialized stream for real-time embodiment data"""
    
    def __init__(self, handler: CognitiveStreamHandler):
        self.handler = handler
        self.setup_embodiment_handlers()
        
    def setup_embodiment_handlers(self):
        """Setup handlers for embodiment-specific messages"""
        self.handler.register_message_handler("sensory_input", self.handle_sensory_input)
        self.handler.register_message_handler("motor_command", self.handle_motor_command)
        self.handler.register_message_handler("embodiment_state", self.handle_embodiment_state)
        self.handler.register_message_handler("feedback_loop", self.handle_feedback_loop)
        
    async def handle_sensory_input(self, stream_id: str, message: StreamMessage):
        """Handle sensory input from embodied agent"""
        sensory_data = message.data
        
        # Process sensory data
        processed_data = {
            "agent_id": message.agent_id,
            "sensory_type": sensory_data.get("type", "unknown"),
            "spatial_coords": sensory_data.get("position", [0.0, 0.0, 0.0]),
            "confidence": sensory_data.get("confidence", 0.5),
            "timestamp": message.timestamp.isoformat()
        }
        
        # Broadcast to other agents
        response_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="sensory_broadcast",
            timestamp=datetime.now(),
            agent_id="system",
            data=processed_data
        )
        
        await self.handler.broadcast_to_all(response_message, exclude_stream=stream_id)
        
        logger.debug(f"Processed sensory input from {message.agent_id}")
    
    async def handle_motor_command(self, stream_id: str, message: StreamMessage):
        """Handle motor command request"""
        motor_data = message.data
        
        # Generate motor response
        motor_response = {
            "linear_velocity": motor_data.get("linear_velocity", [0.0, 0.0, 0.0]),
            "angular_velocity": motor_data.get("angular_velocity", [0.0, 0.0, 0.0]),
            "confidence": motor_data.get("confidence", 0.8),
            "execution_time": 0.1  # Simulated execution time
        }
        
        # Send response back to requesting agent
        response_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="motor_response",
            timestamp=datetime.now(),
            agent_id="system",
            data=motor_response
        )
        
        stream = self.handler.active_streams.get(stream_id)
        if stream:
            await stream.send_message(response_message)
        
        logger.debug(f"Processed motor command from {message.agent_id}")
    
    async def handle_embodiment_state(self, stream_id: str, message: StreamMessage):
        """Handle embodiment state update"""
        state_data = message.data
        
        # Validate and process embodiment state
        processed_state = {
            "agent_id": message.agent_id,
            "position": state_data.get("position", [0.0, 0.0, 0.0]),
            "orientation": state_data.get("orientation", 0.0),
            "embodiment_type": state_data.get("embodiment_type", "virtual"),
            "interaction_mode": state_data.get("interaction_mode", "passive"),
            "timestamp": message.timestamp.isoformat()
        }
        
        # Broadcast state update
        state_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="embodiment_update",
            timestamp=datetime.now(),
            agent_id=message.agent_id,
            data=processed_state
        )
        
        await self.handler.broadcast_to_all(state_message, exclude_stream=stream_id)
        
        logger.debug(f"Processed embodiment state from {message.agent_id}")
    
    async def handle_feedback_loop(self, stream_id: str, message: StreamMessage):
        """Handle feedback loop processing"""
        feedback_data = message.data
        
        # Process feedback loop
        feedback_response = {
            "loop_type": feedback_data.get("loop_type", "closed"),
            "error_signal": feedback_data.get("error", 0.0),
            "correction": feedback_data.get("correction", [0.0, 0.0, 0.0]),
            "adaptation_rate": feedback_data.get("adaptation_rate", 0.1),
            "timestamp": message.timestamp.isoformat()
        }
        
        # Send feedback response
        response_message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="feedback_response",
            timestamp=datetime.now(),
            agent_id="system",
            data=feedback_response
        )
        
        stream = self.handler.active_streams.get(stream_id)
        if stream:
            await stream.send_message(response_message)
        
        logger.debug(f"Processed feedback loop from {message.agent_id}")

class WebSocketServer:
    """WebSocket server for cognitive streams"""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.handler = CognitiveStreamHandler()
        self.embodiment_stream = RealTimeEmbodimentStream(self.handler)
        self.server = None
        
    async def handle_client(self, websocket, path):
        """Handle new client connection"""
        stream_id = str(uuid.uuid4())
        logger.info(f"New client connected: {stream_id}")
        
        stream = self.handler.create_stream(stream_id, websocket)
        
        try:
            await stream.listen()
        except Exception as e:
            logger.error(f"Error handling client {stream_id}: {e}")
        finally:
            logger.info(f"Client disconnected: {stream_id}")
    
    async def start(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"Cognitive stream server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await self.server.wait_closed()
    
    def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            "server_address": f"ws://{self.host}:{self.port}",
            "server_running": self.server is not None,
            **self.handler.get_stream_stats()
        }

# Example client for testing
class CognitiveStreamClient:
    """Client for connecting to cognitive streams"""
    
    def __init__(self, server_url: str, agent_id: str):
        self.server_url = server_url
        self.agent_id = agent_id
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to cognitive stream server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"Connected to cognitive stream: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def send_sensory_input(self, sensory_data: Dict[str, Any]):
        """Send sensory input to server"""
        if not self.is_connected:
            return False
        
        message = StreamMessage(
            message_id=str(uuid.uuid4()),
            message_type="sensory_input",
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            data=sensory_data
        )
        
        try:
            await self.websocket.send(message.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send sensory input: {e}")
            return False
    
    async def listen_for_responses(self):
        """Listen for responses from server"""
        if not self.is_connected:
            return
        
        try:
            async for message_data in self.websocket:
                message = StreamMessage.from_json(message_data)
                await self.handle_response(message)
        except Exception as e:
            logger.error(f"Error listening for responses: {e}")
    
    async def handle_response(self, message: StreamMessage):
        """Handle response from server"""
        logger.info(f"Received {message.message_type} from {message.agent_id}")
        # Override in subclasses for specific handling
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False

if __name__ == "__main__":
    # Example usage
    async def main():
        server = WebSocketServer()
        await server.start()
    
    asyncio.run(main())