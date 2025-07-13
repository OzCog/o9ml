"""
Distributed Cognitive Mesh API

Main API server providing REST endpoints for cognitive state propagation,
task orchestration, and real-time synchronization.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import json
import asyncio
import time
from datetime import datetime
import logging
from pydantic import BaseModel
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

app = FastAPI(
    title="OpenCog Central - Cognitive Mesh API",
    description="Distributed cognitive state propagation and task orchestration API",
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc"
)

# Enable CORS for web-based embodied agents
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.cognitive_state: Dict[str, Any] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models for API requests/responses
class CognitiveState(BaseModel):
    """Cognitive state representation"""
    agent_id: str
    timestamp: datetime
    attention_vector: List[float]
    embodiment_tensor: Dict[str, Any]
    processing_status: str
    confidence: float
    
class TaskRequest(BaseModel):
    """Task orchestration request"""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: Optional[int] = 30
    
class TaskResponse(BaseModel):
    """Task orchestration response"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    
class EmbodimentTensor(BaseModel):
    """Embodiment tensor following the specified signature"""
    sensory_modality: List[str]  # [visual, auditory, tactile, proprioceptive]
    motor_command: List[float]   # [position, velocity, force]
    spatial_coordinates: List[float]  # [x, y, z, orientation]
    temporal_context: List[str]  # [past, present, future]
    action_confidence: float     # [0.0, 1.0]
    embodiment_state: str        # [virtual, physical, hybrid]
    interaction_mode: str        # [passive, active, adaptive]
    feedback_loop: str           # [open, closed, predictive]

# Global task registry
task_registry: Dict[str, Dict[str, Any]] = {}

@app.get(f"{API_PREFIX}/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_connections": len(manager.active_connections),
        "api_version": API_VERSION
    }

@app.get(f"{API_PREFIX}/cognitive-state")
async def get_cognitive_state():
    """Get current distributed cognitive state"""
    return {
        "cognitive_state": manager.cognitive_state,
        "active_agents": len(manager.active_connections),
        "last_update": datetime.now().isoformat()
    }

@app.post(f"{API_PREFIX}/cognitive-state")
async def update_cognitive_state(state: CognitiveState):
    """Update cognitive state from an agent"""
    state_dict = state.dict()
    manager.cognitive_state[state.agent_id] = state_dict
    
    # Broadcast state update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "cognitive_state_update",
        "agent_id": state.agent_id,
        "state": state_dict
    }))
    
    return {"status": "updated", "agent_id": state.agent_id}

@app.post(f"{API_PREFIX}/tasks")
async def create_task(task: TaskRequest) -> TaskResponse:
    """Create and execute a cognitive task"""
    start_time = time.time()
    
    try:
        # Register task
        task_registry[task.task_id] = {
            "request": task.dict(),
            "status": "processing",
            "created_at": datetime.now().isoformat()
        }
        
        # Simulate task processing (replace with actual cognitive processing)
        await asyncio.sleep(0.1)  # Minimal processing delay
        
        # Task execution logic would go here
        result = {
            "processed": True,
            "task_type": task.task_type,
            "parameters_processed": len(task.parameters)
        }
        
        execution_time = time.time() - start_time
        
        task_response = TaskResponse(
            task_id=task.task_id,
            status="completed",
            result=result,
            execution_time=execution_time
        )
        
        # Update task registry
        task_registry[task.task_id]["status"] = "completed"
        task_registry[task.task_id]["result"] = result
        
        return task_response
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_response = TaskResponse(
            task_id=task.task_id,
            status="failed",
            error=str(e),
            execution_time=execution_time
        )
        
        task_registry[task.task_id]["status"] = "failed"
        task_registry[task.task_id]["error"] = str(e)
        
        return error_response

@app.get(f"{API_PREFIX}/tasks/{{task_id}}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    if task_id not in task_registry:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_registry[task_id]

@app.get(f"{API_PREFIX}/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": task_registry}

@app.post(f"{API_PREFIX}/embodiment/tensor")
async def process_embodiment_tensor(tensor: EmbodimentTensor):
    """Process embodiment tensor data"""
    tensor_dict = tensor.dict()
    
    # Broadcast embodiment update
    await manager.broadcast(json.dumps({
        "type": "embodiment_update",
        "tensor": tensor_dict,
        "timestamp": datetime.now().isoformat()
    }))
    
    return {
        "status": "processed",
        "tensor_signature": "Embodiment_Tensor[8]",
        "dimensions": len(tensor_dict),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket(f"{API_PREFIX}/ws/cognitive-stream")
async def cognitive_stream_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time cognitive state streams"""
    await manager.connect(websocket)
    
    try:
        # Send current state to newly connected client
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "cognitive_state": manager.cognitive_state,
            "timestamp": datetime.now().isoformat()
        }))
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif message.get("type") == "cognitive_update":
                # Handle cognitive update from client
                agent_id = message.get("agent_id", "unknown")
                manager.cognitive_state[agent_id] = message.get("data", {})
                
                # Broadcast to other clients
                await manager.broadcast(json.dumps({
                    "type": "cognitive_state_update",
                    "agent_id": agent_id,
                    "data": message.get("data", {}),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get(f"{API_PREFIX}/stats")
async def get_api_stats():
    """Get API usage statistics"""
    return {
        "total_tasks": len(task_registry),
        "active_connections": len(manager.active_connections),
        "cognitive_agents": len(manager.cognitive_state),
        "uptime": datetime.now().isoformat(),
        "api_version": API_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")