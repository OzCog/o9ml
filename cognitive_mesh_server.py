#!/usr/bin/env python3
"""
OpenCog Central - Distributed Cognitive Mesh Server

Main server startup script that initializes and runs all components:
- REST API server
- WebSocket real-time streams
- Embodiment processing
- Unity3D and ROS bridges
"""

import asyncio
import logging
import signal
import sys
import os
from typing import Optional
import argparse
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from cogml.api.cognitive_mesh_api import app as fastapi_app
from cogml.api.websocket_streams import WebSocketServer
from cogml.embodiment.unity3d_bridge import Unity3DBridge
from cogml.embodiment.ros_bridge import ROSCognitiveNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cognitive_mesh.log')
    ]
)

logger = logging.getLogger(__name__)

class CognitiveMeshServer:
    """Main server orchestrator for the cognitive mesh"""
    
    def __init__(self, config: dict):
        self.config = config
        self.fastapi_server = None
        self.websocket_server = None
        self.unity_bridge = None
        self.ros_node = None
        self.running = False
        
        # Server components
        self.servers = {}
        
    async def start_fastapi_server(self):
        """Start FastAPI REST server"""
        config = uvicorn.Config(
            fastapi_app,
            host=self.config['api']['host'],
            port=self.config['api']['port'],
            log_level=self.config['api']['log_level'],
            access_log=True
        )
        
        self.fastapi_server = uvicorn.Server(config)
        logger.info(f"Starting FastAPI server on {self.config['api']['host']}:{self.config['api']['port']}")
        
        # Run in background
        asyncio.create_task(self.fastapi_server.serve())
    
    async def start_websocket_server(self):
        """Start WebSocket real-time streams server"""
        self.websocket_server = WebSocketServer(
            host=self.config['websocket']['host'],
            port=self.config['websocket']['port']
        )
        
        logger.info(f"Starting WebSocket server on {self.config['websocket']['host']}:{self.config['websocket']['port']}")
        
        # Run in background
        asyncio.create_task(self.websocket_server.start())
    
    async def start_unity_bridge(self):
        """Start Unity3D bridge if enabled"""
        if not self.config['unity']['enabled']:
            logger.info("Unity3D bridge disabled")
            return
        
        self.unity_bridge = Unity3DBridge(
            cognitive_mesh_url=f"ws://{self.config['websocket']['host']}:{self.config['websocket']['port']}"
        )
        
        logger.info(f"Starting Unity3D bridge on port {self.config['unity']['port']}")
        
        # Run in background
        asyncio.create_task(self.unity_bridge.run(self.config['unity']['port']))
    
    async def start_ros_node(self):
        """Start ROS cognitive node if enabled"""
        if not self.config['ros']['enabled']:
            logger.info("ROS cognitive node disabled")
            return
        
        self.ros_node = ROSCognitiveNode(
            node_name=self.config['ros']['node_name'],
            cognitive_mesh_url=f"ws://{self.config['websocket']['host']}:{self.config['websocket']['port']}"
        )
        
        logger.info(f"Starting ROS cognitive node: {self.config['ros']['node_name']}")
        
        # Run in background
        asyncio.create_task(self.ros_node.run())
    
    async def start_all_services(self):
        """Start all enabled services"""
        logger.info("Starting OpenCog Central Distributed Cognitive Mesh...")
        
        # Start core services
        await self.start_fastapi_server()
        await self.start_websocket_server()
        
        # Wait a moment for core services to initialize
        await asyncio.sleep(2)
        
        # Start bridge services
        await self.start_unity_bridge()
        await self.start_ros_node()
        
        self.running = True
        logger.info("All services started successfully!")
        
        # Print service information
        self.print_service_info()
    
    def print_service_info(self):
        """Print information about running services"""
        logger.info("=" * 60)
        logger.info("OpenCog Central - Distributed Cognitive Mesh")
        logger.info("=" * 60)
        logger.info(f"REST API:        http://{self.config['api']['host']}:{self.config['api']['port']}")
        logger.info(f"API Docs:        http://{self.config['api']['host']}:{self.config['api']['port']}/api/v1/docs")
        logger.info(f"WebSocket:       ws://{self.config['websocket']['host']}:{self.config['websocket']['port']}")
        
        if self.config['unity']['enabled']:
            logger.info(f"Unity3D Bridge:  ws://localhost:{self.config['unity']['port']}")
        
        if self.config['ros']['enabled']:
            logger.info(f"ROS Node:        {self.config['ros']['node_name']}")
        
        logger.info("=" * 60)
        logger.info("Service Status:")
        logger.info(f"  FastAPI Server:     {'✓ Running' if self.fastapi_server else '✗ Not started'}")
        logger.info(f"  WebSocket Server:   {'✓ Running' if self.websocket_server else '✗ Not started'}")
        logger.info(f"  Unity3D Bridge:     {'✓ Running' if self.unity_bridge else '✗ Disabled'}")
        logger.info(f"  ROS Node:           {'✓ Running' if self.ros_node else '✗ Disabled'}")
        logger.info("=" * 60)
    
    async def stop_all_services(self):
        """Stop all running services"""
        logger.info("Stopping all services...")
        
        self.running = False
        
        if self.fastapi_server:
            self.fastapi_server.should_exit = True
        
        if self.websocket_server:
            self.websocket_server.stop()
        
        if self.unity_bridge:
            self.unity_bridge.stop()
        
        if self.ros_node:
            self.ros_node.stop()
        
        logger.info("All services stopped")
    
    async def run_forever(self):
        """Run server until interrupted"""
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop_all_services()

def load_config(config_file: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults"""
    default_config = {
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'log_level': 'info'
        },
        'websocket': {
            'host': '0.0.0.0',
            'port': 8001
        },
        'unity': {
            'enabled': True,
            'port': 8002
        },
        'ros': {
            'enabled': False,  # Disabled by default as ROS may not be available
            'node_name': 'opencog_cognitive_node'
        }
    }
    
    if config_file and os.path.exists(config_file):
        import json
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                for section, values in file_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return default_config

async def health_check(config: dict):
    """Perform health check on services"""
    import aiohttp
    
    api_url = f"http://{config['api']['host']}:{config['api']['port']}/api/v1/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ API Health: {data['status']}")
                    print(f"  Version: {data['version']}")
                    print(f"  Active connections: {data['active_connections']}")
                    return True
                else:
                    print(f"✗ API Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ API Health check failed: {e}")
        return False

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='OpenCog Central - Distributed Cognitive Mesh')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--health-check', action='store_true', help='Perform health check and exit')
    parser.add_argument('--enable-ros', action='store_true', help='Enable ROS integration')
    parser.add_argument('--disable-unity', action='store_true', help='Disable Unity3D integration')
    parser.add_argument('--api-port', type=int, default=8000, help='API server port')
    parser.add_argument('--ws-port', type=int, default=8001, help='WebSocket server port')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.enable_ros:
        config['ros']['enabled'] = True
    if args.disable_unity:
        config['unity']['enabled'] = False
    if args.api_port != 8000:
        config['api']['port'] = args.api_port
    if args.ws_port != 8001:
        config['websocket']['port'] = args.ws_port
    
    # Health check mode
    if args.health_check:
        print("Performing health check...")
        success = await health_check(config)
        sys.exit(0 if success else 1)
    
    # Create and start server
    server = CognitiveMeshServer(config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(server.stop_all_services())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        await server.start_all_services()
        
        # Run forever
        await server.run_forever()
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)