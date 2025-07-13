"""
Real-time Streams Integration Tests

Tests for WebSocket cognitive streams, latency analysis,
and real-time embodiment processing.
"""

import pytest
import asyncio
import json
import time
import websockets
from datetime import datetime
from typing import Dict, Any
import uuid

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cogml.api.websocket_streams import (
    WebSocketServer, CognitiveStreamClient, StreamMessage,
    CognitiveStreamHandler, RealTimeEmbodimentStream
)

@pytest.mark.asyncio
class TestWebSocketStreams:
    """Test WebSocket streaming functionality"""
    
    async def test_stream_message_serialization(self):
        """Test StreamMessage serialization/deserialization"""
        original_message = StreamMessage(
            message_id="test_123",
            message_type="sensory_input",
            timestamp=datetime.now(),
            agent_id="test_agent",
            data={"position": [1.0, 2.0, 3.0]},
            priority=1
        )
        
        # Serialize to JSON
        json_str = original_message.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        reconstructed_message = StreamMessage.from_json(json_str)
        
        assert reconstructed_message.message_id == original_message.message_id
        assert reconstructed_message.message_type == original_message.message_type
        assert reconstructed_message.agent_id == original_message.agent_id
        assert reconstructed_message.data == original_message.data
        assert reconstructed_message.priority == original_message.priority
    
    async def test_cognitive_stream_handler(self):
        """Test cognitive stream handler functionality"""
        handler = CognitiveStreamHandler()
        
        # Test handler registration
        handled_messages = []
        
        async def test_handler(stream_id: str, message: StreamMessage):
            handled_messages.append((stream_id, message))
        
        handler.register_message_handler("test_type", test_handler)
        
        # Create test message
        test_message = StreamMessage(
            message_id="test_msg",
            message_type="test_type",
            timestamp=datetime.now(),
            agent_id="test_agent",
            data={"test": "data"}
        )
        
        # Handle message
        await handler.handle_message("stream_1", test_message)
        
        # Verify handling
        assert len(handled_messages) == 1
        assert handled_messages[0][0] == "stream_1"
        assert handled_messages[0][1].message_type == "test_type"
    
    async def test_embodiment_stream_handlers(self):
        """Test embodiment-specific stream handlers"""
        handler = CognitiveStreamHandler()
        embodiment_stream = RealTimeEmbodimentStream(handler)
        
        # Test sensory input handling
        sensory_message = StreamMessage(
            message_id="sensory_1",
            message_type="sensory_input",
            timestamp=datetime.now(),
            agent_id="robot_1",
            data={
                "type": "visual",
                "position": [1.0, 2.0, 3.0],
                "confidence": 0.8
            }
        )
        
        # This should not raise an exception
        await handler.handle_message("stream_1", sensory_message)
        
        # Test motor command handling
        motor_message = StreamMessage(
            message_id="motor_1",
            message_type="motor_command",
            timestamp=datetime.now(),
            agent_id="robot_1",
            data={
                "linear_velocity": [0.1, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.1],
                "confidence": 0.9
            }
        )
        
        await handler.handle_message("stream_1", motor_message)
        
        # Verify handlers are registered
        assert "sensory_input" in handler.message_handlers
        assert "motor_command" in handler.message_handlers
        assert "embodiment_state" in handler.message_handlers
        assert "feedback_loop" in handler.message_handlers

@pytest.mark.asyncio
class TestLatencyAndPerformance:
    """Test real-time performance and latency"""
    
    async def test_message_latency(self):
        """Test message processing latency"""
        handler = CognitiveStreamHandler()
        
        # Track processing times
        processing_times = []
        
        async def latency_handler(stream_id: str, message: StreamMessage):
            start_time = time.time()
            # Simulate processing
            await asyncio.sleep(0.001)  # 1ms processing time
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        handler.register_message_handler("latency_test", latency_handler)
        
        # Send multiple messages
        for i in range(10):
            message = StreamMessage(
                message_id=f"latency_msg_{i}",
                message_type="latency_test",
                timestamp=datetime.now(),
                agent_id="test_agent",
                data={"index": i}
            )
            
            await handler.handle_message("test_stream", message)
        
        # Verify latency
        assert len(processing_times) == 10
        avg_latency = sum(processing_times) / len(processing_times)
        
        # Should process messages quickly (under 10ms average)
        assert avg_latency < 0.01
    
    async def test_concurrent_message_processing(self):
        """Test handling multiple concurrent messages"""
        handler = CognitiveStreamHandler()
        
        processed_messages = []
        
        async def concurrent_handler(stream_id: str, message: StreamMessage):
            await asyncio.sleep(0.01)  # Simulate processing delay
            processed_messages.append(message.message_id)
        
        handler.register_message_handler("concurrent_test", concurrent_handler)
        
        # Send messages concurrently
        tasks = []
        for i in range(20):
            message = StreamMessage(
                message_id=f"concurrent_msg_{i}",
                message_type="concurrent_test",
                timestamp=datetime.now(),
                agent_id=f"agent_{i % 5}",  # 5 different agents
                data={"index": i}
            )
            
            task = handler.handle_message(f"stream_{i % 3}", message)  # 3 streams
            tasks.append(task)
        
        # Wait for all messages to be processed
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all messages processed
        assert len(processed_messages) == 20
        
        # Should handle concurrent processing efficiently
        # (should not take 20 * 0.01 = 0.2s due to concurrency)
        total_time = end_time - start_time
        assert total_time < 0.1  # Should complete in under 100ms
    
    async def test_stream_throughput(self):
        """Test message throughput capabilities"""
        handler = CognitiveStreamHandler()
        
        message_count = 0
        
        async def throughput_handler(stream_id: str, message: StreamMessage):
            nonlocal message_count
            message_count += 1
        
        handler.register_message_handler("throughput_test", throughput_handler)
        
        # Send many messages quickly
        start_time = time.time()
        
        tasks = []
        for i in range(1000):  # 1000 messages
            message = StreamMessage(
                message_id=f"throughput_msg_{i}",
                message_type="throughput_test",
                timestamp=datetime.now(),
                agent_id="throughput_agent",
                data={"index": i}
            )
            
            task = handler.handle_message("throughput_stream", message)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Calculate throughput
        total_time = end_time - start_time
        throughput = message_count / total_time
        
        # Should handle high message throughput
        assert message_count == 1000
        assert throughput > 1000  # At least 1000 messages/second

@pytest.mark.asyncio
class TestRealTimeIntegration:
    """Test real-time integration scenarios"""
    
    async def test_sensory_motor_feedback_loop(self):
        """Test complete sensory-motor feedback loop"""
        handler = CognitiveStreamHandler()
        embodiment_stream = RealTimeEmbodimentStream(handler)
        
        # Track the feedback loop
        feedback_sequence = []
        
        # Mock stream for testing
        class MockStream:
            def __init__(self, stream_id):
                self.stream_id = stream_id
                self.agent_id = "test_robot"
                
            async def send_message(self, message: StreamMessage):
                feedback_sequence.append({
                    "type": message.message_type,
                    "data": message.data,
                    "timestamp": message.timestamp
                })
        
        # Add mock stream to handler
        mock_stream = MockStream("robot_stream")
        handler.active_streams["robot_stream"] = mock_stream
        
        # Step 1: Send sensory input
        sensory_message = StreamMessage(
            message_id="sensory_feedback_1",
            message_type="sensory_input",
            timestamp=datetime.now(),
            agent_id="test_robot",
            data={
                "type": "visual",
                "position": [1.0, 2.0, 3.0],
                "confidence": 0.9
            }
        )
        
        await handler.handle_message("robot_stream", sensory_message)
        
        # Step 2: Send motor command
        motor_message = StreamMessage(
            message_id="motor_feedback_1",
            message_type="motor_command",
            timestamp=datetime.now(),
            agent_id="test_robot",
            data={
                "linear_velocity": [0.1, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.1]
            }
        )
        
        await handler.handle_message("robot_stream", motor_message)
        
        # Step 3: Send feedback
        feedback_message = StreamMessage(
            message_id="feedback_1",
            message_type="feedback_loop",
            timestamp=datetime.now(),
            agent_id="test_robot",
            data={
                "loop_type": "closed",
                "error": 0.1,
                "correction": [0.01, 0.0, 0.0]
            }
        )
        
        await handler.handle_message("robot_stream", feedback_message)
        
        # Verify feedback loop completion
        assert len(feedback_sequence) >= 2  # At least motor response + feedback response
        
        # Check for motor response
        motor_responses = [msg for msg in feedback_sequence if msg["type"] == "motor_response"]
        assert len(motor_responses) == 1
        
        # Check for feedback response
        feedback_responses = [msg for msg in feedback_sequence if msg["type"] == "feedback_response"]
        assert len(feedback_responses) == 1
    
    async def test_multi_agent_coordination(self):
        """Test coordination between multiple agents"""
        handler = CognitiveStreamHandler()
        embodiment_stream = RealTimeEmbodimentStream(handler)
        
        # Mock multiple streams
        class MockMultiStream:
            def __init__(self, stream_id, agent_id):
                self.stream_id = stream_id
                self.agent_id = agent_id
                self.received_messages = []
                
            async def send_message(self, message: StreamMessage):
                self.received_messages.append(message)
        
        # Create multiple agent streams
        agents = ["robot_1", "robot_2", "unity_agent_1"]
        streams = {}
        
        for i, agent_id in enumerate(agents):
            stream = MockMultiStream(f"stream_{i}", agent_id)
            streams[agent_id] = stream
            handler.active_streams[f"stream_{i}"] = stream
        
        # Send state update from one agent
        state_message = StreamMessage(
            message_id="multi_agent_state",
            message_type="embodiment_state",
            timestamp=datetime.now(),
            agent_id="robot_1",
            data={
                "position": [5.0, 6.0, 7.0],
                "orientation": 1.5,
                "embodiment_type": "physical"
            }
        )
        
        await handler.handle_message("stream_0", state_message)
        
        # Verify other agents received the update
        for agent_id, stream in streams.items():
            if agent_id != "robot_1":  # Exclude sender
                embodiment_updates = [
                    msg for msg in stream.received_messages 
                    if msg.message_type == "embodiment_update"
                ]
                assert len(embodiment_updates) >= 1
                
                # Verify message content
                update = embodiment_updates[0]
                assert update.agent_id == "robot_1"
                assert update.data["position"] == [5.0, 6.0, 7.0]
    
    async def test_real_time_constraints(self):
        """Test system meets real-time constraints"""
        handler = CognitiveStreamHandler()
        embodiment_stream = RealTimeEmbodimentStream(handler)
        
        # Test rapid message sequence with timing constraints
        response_times = []
        
        class TimingStream:
            def __init__(self):
                self.agent_id = "timing_test"
                
            async def send_message(self, message: StreamMessage):
                # Record response time
                current_time = time.time()
                response_times.append(current_time)
        
        timing_stream = TimingStream()
        handler.active_streams["timing_stream"] = timing_stream
        
        # Send rapid sequence of motor commands
        start_time = time.time()
        
        for i in range(50):  # 50 commands
            motor_message = StreamMessage(
                message_id=f"timing_motor_{i}",
                message_type="motor_command",
                timestamp=datetime.now(),
                agent_id="timing_test",
                data={
                    "linear_velocity": [0.1 * i, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.1 * i]
                }
            )
            
            await handler.handle_message("timing_stream", motor_message)
            
            # Real-time constraint: 10ms between commands
            await asyncio.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify timing constraints
        assert len(response_times) == 50
        assert total_time < 1.0  # Should complete in under 1 second
        
        # Check individual response times
        if len(response_times) >= 2:
            intervals = [
                response_times[i] - response_times[i-1] 
                for i in range(1, len(response_times))
            ]
            avg_interval = sum(intervals) / len(intervals)
            
            # Average response interval should be close to 10ms
            assert 0.005 < avg_interval < 0.02  # 5ms to 20ms tolerance

@pytest.mark.asyncio  
class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios"""
    
    async def test_invalid_message_handling(self):
        """Test handling of invalid messages"""
        handler = CognitiveStreamHandler()
        
        # Test with invalid JSON
        with pytest.raises(json.JSONDecodeError):
            StreamMessage.from_json("invalid json")
        
        # Test with missing required fields
        incomplete_json = '{"message_id": "test", "message_type": "test"}'
        
        with pytest.raises((KeyError, TypeError)):
            StreamMessage.from_json(incomplete_json)
    
    async def test_handler_exception_recovery(self):
        """Test recovery from handler exceptions"""
        handler = CognitiveStreamHandler()
        
        # Handler that raises an exception
        async def failing_handler(stream_id: str, message: StreamMessage):
            raise Exception("Handler failure")
        
        handler.register_message_handler("failing_type", failing_handler)
        
        # Message should not crash the system
        failing_message = StreamMessage(
            message_id="failing_msg",
            message_type="failing_type",
            timestamp=datetime.now(),
            agent_id="test_agent",
            data={}
        )
        
        # This should not raise an exception (should be caught internally)
        await handler.handle_message("test_stream", failing_message)
        
        # System should continue working
        handled = []
        
        async def working_handler(stream_id: str, message: StreamMessage):
            handled.append(message.message_id)
        
        handler.register_message_handler("working_type", working_handler)
        
        working_message = StreamMessage(
            message_id="working_msg",
            message_type="working_type",
            timestamp=datetime.now(),
            agent_id="test_agent",
            data={}
        )
        
        await handler.handle_message("test_stream", working_message)
        
        # Verify system still works
        assert len(handled) == 1
        assert handled[0] == "working_msg"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])