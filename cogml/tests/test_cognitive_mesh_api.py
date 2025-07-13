"""
Tests for Distributed Cognitive Mesh API

Comprehensive test suite for REST endpoints, WebSocket connections,
and cognitive state synchronization.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
import websockets
from fastapi.testclient import TestClient

# Import the API components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cogml.api.cognitive_mesh_api import app, manager, task_registry, EmbodimentTensor

class TestCognitiveMeshAPI:
    """Test suite for the Cognitive Mesh API"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.client = TestClient(app)
        # Clear state between tests
        manager.cognitive_state.clear()
        task_registry.clear()
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "active_connections" in data
        assert data["api_version"] == "v1"
    
    def test_get_cognitive_state_empty(self):
        """Test getting cognitive state when empty"""
        response = self.client.get("/api/v1/cognitive-state")
        assert response.status_code == 200
        
        data = response.json()
        assert data["cognitive_state"] == {}
        assert data["active_agents"] == 0
        assert "last_update" in data
    
    def test_update_cognitive_state(self):
        """Test updating cognitive state"""
        cognitive_state = {
            "agent_id": "test_agent_1",
            "timestamp": datetime.now().isoformat(),
            "attention_vector": [0.1, 0.2, 0.3],
            "embodiment_tensor": {
                "sensory_modality": ["visual"],
                "motor_command": [0.1, 0.2, 0.3],
                "spatial_coordinates": [1.0, 2.0, 3.0, 0.5],
                "temporal_context": ["present"],
                "action_confidence": 0.8,
                "embodiment_state": "virtual",
                "interaction_mode": "active",
                "feedback_loop": "closed"
            },
            "processing_status": "active",
            "confidence": 0.85
        }
        
        response = self.client.post("/api/v1/cognitive-state", json=cognitive_state)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "updated"
        assert data["agent_id"] == "test_agent_1"
        
        # Verify state was stored
        response = self.client.get("/api/v1/cognitive-state")
        data = response.json()
        assert "test_agent_1" in data["cognitive_state"]
    
    def test_create_task(self):
        """Test task creation and execution"""
        task_request = {
            "task_id": "test_task_1",
            "task_type": "navigation",
            "parameters": {
                "target_position": [5.0, 6.0, 7.0],
                "speed": 0.5
            },
            "priority": 1,
            "timeout": 30
        }
        
        response = self.client.post("/api/v1/tasks", json=task_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["task_id"] == "test_task_1"
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["execution_time"] > 0
    
    def test_get_task_status(self):
        """Test getting task status"""
        # First create a task
        task_request = {
            "task_id": "test_task_2",
            "task_type": "perception",
            "parameters": {"sensor_type": "camera"}
        }
        
        self.client.post("/api/v1/tasks", json=task_request)
        
        # Then get its status
        response = self.client.get("/api/v1/tasks/test_task_2")
        assert response.status_code == 200
        
        data = response.json()
        assert "request" in data
        assert "status" in data
        assert data["status"] == "completed"
    
    def test_get_nonexistent_task(self):
        """Test getting non-existent task"""
        response = self.client.get("/api/v1/tasks/nonexistent_task")
        assert response.status_code == 404
    
    def test_list_tasks(self):
        """Test listing all tasks"""
        # Create a few tasks
        for i in range(3):
            task_request = {
                "task_id": f"test_task_{i}",
                "task_type": "test",
                "parameters": {"index": i}
            }
            self.client.post("/api/v1/tasks", json=task_request)
        
        response = self.client.get("/api/v1/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3
    
    def test_process_embodiment_tensor(self):
        """Test embodiment tensor processing"""
        tensor_data = {
            "sensory_modality": ["visual", "auditory"],
            "motor_command": [0.1, 0.2, 0.3],
            "spatial_coordinates": [1.0, 2.0, 3.0, 0.5],
            "temporal_context": ["present"],
            "action_confidence": 0.8,
            "embodiment_state": "virtual",
            "interaction_mode": "active",
            "feedback_loop": "closed"
        }
        
        response = self.client.post("/api/v1/embodiment/tensor", json=tensor_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "processed"
        assert data["tensor_signature"] == "Embodiment_Tensor[8]"
        assert data["dimensions"] == 8
        assert "timestamp" in data
    
    def test_api_stats(self):
        """Test API statistics endpoint"""
        # Create some tasks first
        for i in range(2):
            task_request = {
                "task_id": f"stats_test_{i}",
                "task_type": "test",
                "parameters": {}
            }
            self.client.post("/api/v1/tasks", json=task_request)
        
        response = self.client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_tasks"] == 2
        assert data["active_connections"] == 0  # No WebSocket connections in tests
        assert data["cognitive_agents"] == 0
        assert "uptime" in data
        assert data["api_version"] == "v1"
    
    def test_embodiment_tensor_validation(self):
        """Test embodiment tensor validation"""
        # Test with invalid sensory modality
        invalid_tensor = {
            "sensory_modality": ["invalid_modality"],
            "motor_command": [0.1, 0.2, 0.3],
            "spatial_coordinates": [1.0, 2.0, 3.0, 0.5],
            "temporal_context": ["present"],
            "action_confidence": 0.8,
            "embodiment_state": "virtual",
            "interaction_mode": "active",
            "feedback_loop": "closed"
        }
        
        # This should not fail at API level (validation is in tensor processor)
        response = self.client.post("/api/v1/embodiment/tensor", json=invalid_tensor)
        assert response.status_code == 200
    
    def test_concurrent_cognitive_state_updates(self):
        """Test concurrent cognitive state updates"""
        import threading
        import time
        
        results = []
        
        def update_state(agent_id):
            cognitive_state = {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "attention_vector": [0.1, 0.2, 0.3],
                "embodiment_tensor": {},
                "processing_status": "active",
                "confidence": 0.85
            }
            
            response = self.client.post("/api/v1/cognitive-state", json=cognitive_state)
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_state, args=[f"agent_{i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check all updates succeeded
        assert all(status == 200 for status in results)
        
        # Verify all agents are in state
        response = self.client.get("/api/v1/cognitive-state")
        data = response.json()
        assert len(data["cognitive_state"]) == 5

@pytest.mark.asyncio
class TestWebSocketConnections:
    """Test WebSocket functionality"""
    
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        # This would require running the actual server
        # For now, test the connection manager logic
        from cogml.api.cognitive_mesh_api import ConnectionManager
        
        manager = ConnectionManager()
        
        # Mock WebSocket
        class MockWebSocket:
            def __init__(self):
                self.closed = False
                
            async def accept(self):
                pass
                
            async def send_text(self, message):
                pass
        
        mock_ws = MockWebSocket()
        
        # Test connection
        await manager.connect(mock_ws)
        assert len(manager.active_connections) == 1
        
        # Test disconnect
        manager.disconnect(mock_ws)
        assert len(manager.active_connections) == 0
    
    async def test_broadcast_message(self):
        """Test message broadcasting"""
        from cogml.api.cognitive_mesh_api import ConnectionManager
        
        manager = ConnectionManager()
        
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                
            async def accept(self):
                pass
                
            async def send_text(self, message):
                self.messages.append(message)
        
        # Add multiple mock connections
        mock_ws1 = MockWebSocket()
        mock_ws2 = MockWebSocket()
        
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        # Broadcast message
        test_message = "test broadcast"
        await manager.broadcast(test_message)
        
        # Verify both connections received the message
        assert test_message in mock_ws1.messages
        assert test_message in mock_ws2.messages

class TestPerformanceAndLoad:
    """Performance and load testing"""
    
    def test_api_response_time(self):
        """Test API response times"""
        client = TestClient(app)
        
        # Measure health check response time
        start_time = time.time()
        response = client.get("/api/v1/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should respond within 100ms
    
    def test_bulk_task_creation(self):
        """Test creating many tasks"""
        client = TestClient(app)
        
        start_time = time.time()
        
        # Create 100 tasks
        for i in range(100):
            task_request = {
                "task_id": f"bulk_task_{i}",
                "task_type": "test",
                "parameters": {"index": i}
            }
            response = client.post("/api/v1/tasks", json=task_request)
            assert response.status_code == 200
        
        execution_time = time.time() - start_time
        
        # Should handle 100 tasks in reasonable time
        assert execution_time < 10.0  # 10 seconds max
        
        # Verify all tasks were created
        response = client.get("/api/v1/tasks")
        data = response.json()
        assert len(data["tasks"]) == 100
    
    def test_memory_usage(self):
        """Test memory usage with many cognitive states"""
        import psutil
        import os
        
        client = TestClient(app)
        
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many cognitive states
        for i in range(1000):
            cognitive_state = {
                "agent_id": f"memory_test_agent_{i}",
                "timestamp": datetime.now().isoformat(),
                "attention_vector": [0.1] * 100,  # Large vector
                "embodiment_tensor": {
                    "sensory_modality": ["visual", "auditory"],
                    "motor_command": [0.1, 0.2, 0.3],
                    "spatial_coordinates": [1.0, 2.0, 3.0, 0.5],
                    "temporal_context": ["present"],
                    "action_confidence": 0.8,
                    "embodiment_state": "virtual",
                    "interaction_mode": "active",
                    "feedback_loop": "closed"
                },
                "processing_status": "active",
                "confidence": 0.85
            }
            client.post("/api/v1/cognitive-state", json=cognitive_state)
        
        # Measure final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"])