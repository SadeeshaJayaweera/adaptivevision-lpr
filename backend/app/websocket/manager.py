from fastapi import WebSocket
from typing import List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_detection(self, prediction_data: dict):
        message = json.dumps({"type": "NEW_DETECTION", "data": prediction_data})
        for connection in self.active_connections:
            await connection.send_text(message)
            
    async def broadcast_status(self, status_data: dict):
        message = json.dumps({"type": "STATUS_UPDATE", "data": status_data})
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
