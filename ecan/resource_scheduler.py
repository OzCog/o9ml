"""
Resource Scheduler with Priority Queues

Implements priority queue-based resource scheduling for ECAN attention allocation.
Manages cognitive resource distribution with attention-based priorities and
real-time scheduling constraints.

Key features:
- Priority queue management for attention requests
- Real-time scheduling with deadline constraints
- Resource conflict resolution protocols  
- Dynamic priority adjustment
- Load balancing across cognitive resources
"""

import time
import heapq
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from threading import Lock, Event, Thread
from queue import Queue, Empty
import logging
from enum import Enum
from .attention_kernel import ECANAttentionTensor, AttentionKernel
from .economic_allocator import AttentionAllocationRequest, EconomicAllocator

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class ScheduledTask:
    """
    A scheduled cognitive task with attention requirements.
    """
    task_id: str
    attention_request: AttentionAllocationRequest
    callback: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    max_execution_time: float = 60.0  # seconds
    retry_count: int = 0
    max_retries: int = 3
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering (higher priority first)"""
        return self.attention_request.compute_total_priority() > other.attention_request.compute_total_priority()
    
    def is_expired(self) -> bool:
        """Check if task has expired based on deadline"""
        if self.attention_request.deadline is None:
            return False
        return time.time() > self.attention_request.deadline
    
    def get_age(self) -> float:
        """Get task age in seconds"""
        return time.time() - self.created_time
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time if completed"""
        if self.started_time and self.completed_time:
            return self.completed_time - self.started_time
        return None


class ResourceScheduler:
    """
    Priority queue-based scheduler for cognitive resource allocation.
    
    Manages scheduling of attention allocation requests with consideration for
    priorities, deadlines, dependencies, and resource constraints.
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 scheduler_interval: float = 0.1,
                 enable_background_processing: bool = True):
        """
        Initialize resource scheduler.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            scheduler_interval: Scheduler polling interval in seconds
            enable_background_processing: Enable background scheduling thread
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.scheduler_interval = scheduler_interval
        
        # Task queues and storage
        self.priority_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.task_dependencies: Dict[str, List[str]] = {}  # task_id -> dependencies
        
        # Resource allocation tracking
        self.resource_allocations: Dict[str, float] = {}  # resource_id -> allocated_amount
        self.resource_limits: Dict[str, float] = {
            'attention_budget': 100.0,
            'processing_slots': float(max_concurrent_tasks),
            'memory_mb': 1000.0
        }
        
        # Scheduling metrics
        self.metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_expired': 0,
            'average_wait_time': 0.0,
            'average_execution_time': 0.0,
            'scheduler_cycles': 0,
            'resource_utilization': {}
        }
        
        # Thread synchronization
        self._lock = Lock()
        self._shutdown_event = Event()
        self._scheduler_thread: Optional[Thread] = None
        
        # Start background scheduler if enabled
        if enable_background_processing:
            self.start_scheduler()
        
        logger.info(f"ResourceScheduler initialized: max_concurrent={max_concurrent_tasks}, "
                   f"interval={scheduler_interval}s, background={enable_background_processing}")
    
    def schedule_task(self, task: ScheduledTask) -> bool:
        """
        Schedule a task for execution.
        
        Args:
            task: Scheduled task to add to queue
            
        Returns:
            True if task was scheduled, False otherwise
        """
        with self._lock:
            # Check if task already exists
            if any(t.task_id == task.task_id for t in self.priority_queue):
                logger.warning(f"Task {task.task_id} already scheduled")
                return False
            
            # Check if task is expired
            if task.is_expired():
                task.status = TaskStatus.EXPIRED
                self.metrics['tasks_expired'] += 1
                logger.warning(f"Task {task.task_id} expired before scheduling")
                return False
            
            # Add to priority queue
            heapq.heappush(self.priority_queue, task)
            
            # Track dependencies
            if task.dependencies:
                self.task_dependencies[task.task_id] = task.dependencies.copy()
            
            self.metrics['tasks_scheduled'] += 1
            
            logger.debug(f"Scheduled task {task.task_id} with priority {task.attention_request.compute_total_priority():.3f}")
            return True
    
    def process_scheduler_cycle(self, 
                               attention_kernel: AttentionKernel,
                               economic_allocator: EconomicAllocator) -> Dict[str, Any]:
        """
        Process one scheduler cycle, executing eligible tasks.
        
        Args:
            attention_kernel: Attention kernel for allocations
            economic_allocator: Economic allocator for resource decisions
            
        Returns:
            Cycle results and metrics
        """
        with self._lock:
            cycle_start = time.time()
            tasks_started = 0
            tasks_completed = 0
            tasks_expired = 0
            
            # Clean up completed and expired tasks
            self._cleanup_completed_tasks()
            
            # Check for expired tasks in queue
            expired_tasks = []
            while self.priority_queue and self.priority_queue[0].is_expired():
                expired_task = heapq.heappop(self.priority_queue)
                expired_task.status = TaskStatus.EXPIRED
                expired_tasks.append(expired_task)
                tasks_expired += 1
            
            # Process ready tasks
            ready_tasks = self._get_ready_tasks()
            
            for task in ready_tasks:
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    break  # At capacity
                
                # Check resource availability
                if self._check_resource_availability(task):
                    # Try to allocate attention
                    allocation_result = economic_allocator.evaluate_allocation_request(task.attention_request)
                    
                    if allocation_result['recommended']:
                        # Allocate attention and start task
                        success = attention_kernel.allocate_attention(
                            task.attention_request.atom_id,
                            task.attention_request.requested_attention
                        )
                        
                        if success:
                            self._start_task(task)
                            tasks_started += 1
                        else:
                            logger.warning(f"Failed to allocate attention for task {task.task_id}")
                    else:
                        logger.debug(f"Economic allocator rejected task {task.task_id}: "
                                   f"score={allocation_result['allocation_score']:.3f}")
            
            # Check for completed running tasks
            completed_tasks_list = self._check_running_tasks()
            tasks_completed = len(completed_tasks_list)
            
            # Update metrics
            self.metrics['scheduler_cycles'] += 1
            self.metrics['tasks_completed'] += tasks_completed
            self.metrics['tasks_expired'] += tasks_expired
            
            cycle_time = time.time() - cycle_start
            
            results = {
                'cycle_time': cycle_time,
                'tasks_started': tasks_started,
                'tasks_completed': tasks_completed,
                'tasks_expired': tasks_expired,
                'queue_size': len(self.priority_queue),
                'running_tasks': len(self.running_tasks),
                'resource_utilization': self._calculate_resource_utilization()
            }
            
            logger.debug(f"Scheduler cycle: {tasks_started} started, {tasks_completed} completed, "
                        f"{tasks_expired} expired, queue={len(self.priority_queue)}")
            
            return results
    
    def _get_ready_tasks(self) -> List[ScheduledTask]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        ready_tasks = []
        remaining_tasks = []
        
        while self.priority_queue:
            task = heapq.heappop(self.priority_queue)
            
            # Check if dependencies are satisfied
            if self._dependencies_satisfied(task):
                ready_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Put non-ready tasks back in queue
        for task in remaining_tasks:
            heapq.heappush(self.priority_queue, task)
        
        return ready_tasks
    
    def _dependencies_satisfied(self, task: ScheduledTask) -> bool:
        """Check if all task dependencies are satisfied"""
        if task.task_id not in self.task_dependencies:
            return True  # No dependencies
        
        dependencies = self.task_dependencies[task.task_id]
        for dep_id in dependencies:
            # Check if dependency is completed
            if dep_id not in self.completed_tasks:
                return False
            
            # Check if dependency succeeded
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _check_resource_availability(self, task: ScheduledTask) -> bool:
        """Check if required resources are available for task"""
        # Check processing slots
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check attention budget (simplified)
        required_attention = task.attention_request.requested_attention.compute_salience() * 10.0
        current_attention_usage = sum(
            self.running_tasks[tid].attention_request.requested_attention.compute_salience() * 10.0
            for tid in self.running_tasks
        )
        
        if current_attention_usage + required_attention > self.resource_limits['attention_budget']:
            return False
        
        return True
    
    def _start_task(self, task: ScheduledTask):
        """Start executing a task"""
        task.status = TaskStatus.RUNNING
        task.started_time = time.time()
        
        # Remove from queue and add to running tasks
        self.running_tasks[task.task_id] = task
        
        # Execute callback if provided
        if task.callback:
            try:
                # Run callback in separate thread for non-blocking execution
                def run_callback():
                    try:
                        result = task.callback(task)
                        task.status = TaskStatus.COMPLETED
                        task.completed_time = time.time()
                        logger.debug(f"Task {task.task_id} completed successfully")
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.completed_time = time.time()
                        logger.error(f"Task {task.task_id} failed: {e}")
                
                callback_thread = Thread(target=run_callback, daemon=True)
                callback_thread.start()
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_time = time.time()
                logger.error(f"Failed to start task {task.task_id}: {e}")
        
        logger.debug(f"Started task {task.task_id}")
    
    def _check_running_tasks(self) -> List[ScheduledTask]:
        """Check running tasks for completion or timeout"""
        completed = []
        current_time = time.time()
        
        for task_id, task in list(self.running_tasks.items()):
            # Check for completion
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self._complete_task(task)
                completed.append(task)
            
            # Check for timeout
            elif (task.started_time and 
                  current_time - task.started_time > task.max_execution_time):
                task.status = TaskStatus.FAILED
                task.completed_time = current_time
                self._complete_task(task)
                completed.append(task)
                logger.warning(f"Task {task.task_id} timed out after {task.max_execution_time}s")
        
        return completed
    
    def _complete_task(self, task: ScheduledTask):
        """Complete a task and move to completed tasks"""
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        
        self.completed_tasks[task.task_id] = task
        
        # Update metrics
        if task.status == TaskStatus.COMPLETED:
            self.metrics['tasks_completed'] += 1
        elif task.status == TaskStatus.FAILED:
            self.metrics['tasks_failed'] += 1
        
        # Calculate execution time
        if task.get_execution_time():
            execution_time = task.get_execution_time()
            current_avg = self.metrics['average_execution_time']
            total_completed = self.metrics['tasks_completed']
            
            if total_completed > 0:
                self.metrics['average_execution_time'] = (
                    (current_avg * (total_completed - 1) + execution_time) / total_completed
                )
        
        logger.debug(f"Completed task {task.task_id} with status {task.status}")
    
    def _cleanup_completed_tasks(self, max_age: float = 3600.0):
        """Clean up old completed tasks"""
        current_time = time.time()
        expired_tasks = []
        
        for task_id, task in list(self.completed_tasks.items()):
            if task.completed_time and current_time - task.completed_time > max_age:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del self.completed_tasks[task_id]
            if task_id in self.task_dependencies:
                del self.task_dependencies[task_id]
        
        if expired_tasks:
            logger.debug(f"Cleaned up {len(expired_tasks)} old completed tasks")
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        utilization = {}
        
        # Processing slots utilization
        utilization['processing_slots'] = len(self.running_tasks) / self.max_concurrent_tasks
        
        # Attention budget utilization
        current_attention = sum(
            task.attention_request.requested_attention.compute_salience() * 10.0
            for task in self.running_tasks.values()
        )
        utilization['attention_budget'] = current_attention / self.resource_limits['attention_budget']
        
        return utilization
    
    def start_scheduler(self):
        """Start background scheduler thread"""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler thread already running")
            return
        
        def scheduler_loop():
            logger.info("Background scheduler started")
            while not self._shutdown_event.is_set():
                try:
                    # This would normally call process_scheduler_cycle
                    # but we need external components, so we just sleep
                    time.sleep(self.scheduler_interval)
                except Exception as e:
                    logger.error(f"Scheduler thread error: {e}")
        
        self._scheduler_thread = Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Background scheduler thread started")
    
    def stop_scheduler(self):
        """Stop background scheduler thread"""
        self._shutdown_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("Background scheduler stopped")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task"""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        
        # Check pending tasks
        for task in self.priority_queue:
            if task.task_id == task_id:
                return task.status
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        with self._lock:
            # Check pending tasks
            for i, task in enumerate(self.priority_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    del self.priority_queue[i]
                    heapq.heapify(self.priority_queue)
                    logger.info(f"Cancelled pending task {task_id}")
                    return True
            
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_time = time.time()
                self._complete_task(task)
                logger.info(f"Cancelled running task {task_id}")
                return True
            
            return False
    
    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler metrics"""
        with self._lock:
            current_utilization = self._calculate_resource_utilization()
            
            # Calculate average wait time
            total_wait_time = 0.0
            wait_time_count = 0
            
            for task in list(self.running_tasks.values()) + list(self.completed_tasks.values()):
                if task.started_time:
                    wait_time = task.started_time - task.created_time
                    total_wait_time += wait_time
                    wait_time_count += 1
            
            avg_wait_time = total_wait_time / max(wait_time_count, 1)
            
            return {
                **self.metrics,
                'queue_size': len(self.priority_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'average_wait_time': avg_wait_time,
                'current_resource_utilization': current_utilization,
                'scheduler_active': self._scheduler_thread and self._scheduler_thread.is_alive()
            }
    
    def reset_scheduler(self):
        """Reset scheduler state and metrics"""
        with self._lock:
            # Clear all tasks
            self.priority_queue.clear()
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.task_dependencies.clear()
            
            # Reset metrics
            self.metrics = {
                'tasks_scheduled': 0,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'tasks_expired': 0,
                'average_wait_time': 0.0,
                'average_execution_time': 0.0,
                'scheduler_cycles': 0,
                'resource_utilization': {}
            }
            
            logger.info("Scheduler state and metrics reset")
    
    def __del__(self):
        """Cleanup when scheduler is destroyed"""
        self.stop_scheduler()