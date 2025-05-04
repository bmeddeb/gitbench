# Web Monitoring Example

This example demonstrates how to create a simple web application for monitoring Git repository cloning operations. It uses FastAPI to provide a real-time web dashboard that displays clone status, progress, and repository information.

## Code Example

```python
#!/usr/bin/env python3
"""
GitFleet Web Monitor Example

This example demonstrates how to create a web-based dashboard for monitoring
repository cloning operations using GitFleet and FastAPI.

Dependencies:
- fastapi: Web framework for APIs (pip install fastapi)
- uvicorn: ASGI server for FastAPI (pip install uvicorn)
- jinja2: Template engine (pip install jinja2)

Install all with: pip install "gitfleet[web]"
"""

import os
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from GitFleet import RepoManager
from GitFleet.models.repo import CloneTask


# Create the FastAPI application
app = FastAPI(title="GitFleet Web Monitor")

# Create a directory for templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create a simple CSS file
css_content = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f8f9fa;
    color: #333;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
}
h1 {
    color: #3f51b5;
    border-bottom: 2px solid #3f51b5;
    padding-bottom: 10px;
}
.dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
    margin-top: 20px;
}
.repo-card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 15px;
    transition: transform 0.2s;
}
.repo-card:hover {
    transform: translateY(-5px);
}
.repo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.repo-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #3f51b5;
}
.status {
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}
.status-queued {
    background-color: #ffd54f;
    color: #7e6514;
}
.status-cloning {
    background-color: #64b5f6;
    color: #0d47a1;
}
.status-completed {
    background-color: #a5d6a7;
    color: #1b5e20;
}
.status-failed {
    background-color: #ef9a9a;
    color: #b71c1c;
}
.progress-bar {
    height: 10px;
    background-color: #e0e0e0;
    border-radius: 5px;
    margin: 10px 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background-color: #4caf50;
    width: 0%;
    transition: width 0.3s ease;
}
.details {
    font-size: 0.9rem;
    margin-top: 10px;
    color: #666;
}
.error {
    color: #d32f2f;
    margin-top: 10px;
    font-size: 0.9rem;
    word-break: break-word;
}
.summary {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 15px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
}
.summary-item {
    text-align: center;
}
.summary-value {
    font-size: 2rem;
    font-weight: bold;
}
.summary-label {
    font-size: 0.9rem;
    color: #666;
}
.queue { color: #ffa000; }
.active { color: #1976d2; }
.completed { color: #388e3c; }
.failed { color: #d32f2f; }
"""

# Write CSS file
with open("static/style.css", "w") as f:
    f.write(css_content)

# Create HTML template
template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>GitFleet Monitor</title>
    <link rel="stylesheet" href="/static/style.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <div class="container">
        <h1>GitFleet Repository Clone Monitor</h1>
        
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value" id="total-repos">{{ tasks|length }}</div>
                <div class="summary-label">Total Repositories</div>
            </div>
            <div class="summary-item">
                <div class="summary-value queue" id="queued-repos">{{ queued }}</div>
                <div class="summary-label">Queued</div>
            </div>
            <div class="summary-item">
                <div class="summary-value active" id="active-repos">{{ active }}</div>
                <div class="summary-label">Active</div>
            </div>
            <div class="summary-item">
                <div class="summary-value completed" id="completed-repos">{{ completed }}</div>
                <div class="summary-label">Completed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value failed" id="failed-repos">{{ failed }}</div>
                <div class="summary-label">Failed</div>
            </div>
        </div>
        
        <div class="dashboard" id="dashboard">
            {% for url, task in tasks.items() %}
            <div class="repo-card" id="repo-{{ loop.index }}">
                <div class="repo-header">
                    <div class="repo-title">{{ url.split('/')[-1].replace('.git', '') }}</div>
                    <div class="status status-{{ task.status.status_type }}">{{ task.status.status_type|upper }}</div>
                </div>
                <div>{{ url }}</div>
                
                {% if task.status.status_type == "cloning" and task.status.progress is not none %}
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ task.status.progress }}%"></div>
                </div>
                <div>Progress: {{ task.status.progress }}%</div>
                {% endif %}
                
                {% if task.status.status_type == "failed" and task.status.error %}
                <div class="error">Error: {{ task.status.error }}</div>
                {% endif %}
                
                {% if task.temp_dir %}
                <div class="details">Directory: {{ task.temp_dir }}</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // Update summary counts
            document.getElementById('total-repos').textContent = Object.keys(data.tasks).length;
            document.getElementById('queued-repos').textContent = data.summary.queued;
            document.getElementById('active-repos').textContent = data.summary.active;
            document.getElementById('completed-repos').textContent = data.summary.completed;
            document.getElementById('failed-repos').textContent = data.summary.failed;
            
            // Clear existing repos
            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = '';
            
            // Add repo cards
            let index = 1;
            for (const [url, task] of Object.entries(data.tasks)) {
                const repoName = url.split('/').pop().replace('.git', '');
                const statusType = task.status.status_type;
                
                const card = document.createElement('div');
                card.className = 'repo-card';
                card.id = `repo-${index}`;
                
                let cardContent = `
                    <div class="repo-header">
                        <div class="repo-title">${repoName}</div>
                        <div class="status status-${statusType}">${statusType.toUpperCase()}</div>
                    </div>
                    <div>${url}</div>
                `;
                
                if (statusType === 'cloning' && task.status.progress !== null) {
                    cardContent += `
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${task.status.progress}%"></div>
                        </div>
                        <div>Progress: ${task.status.progress}%</div>
                    `;
                }
                
                if (statusType === 'failed' && task.status.error) {
                    cardContent += `
                        <div class="error">Error: ${task.status.error}</div>
                    `;
                }
                
                if (task.temp_dir) {
                    cardContent += `
                        <div class="details">Directory: ${task.temp_dir}</div>
                    `;
                }
                
                card.innerHTML = cardContent;
                dashboard.appendChild(card);
                index++;
            }
        };
    </script>
</body>
</html>
"""

# Write template file
with open("templates/index.html", "w") as f:
    f.write(template_content)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store global state
class AppState:
    def __init__(self):
        self.repo_manager: Optional[RepoManager] = None
        self.clone_future = None
        self.active_connections: List[WebSocket] = []
        self.monitor_task = None
        self.refresh_rate = 1.0  # Default refresh rate

state = AppState()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Will be removed on next iteration
                pass

manager = ConnectionManager()

# Root page
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # Get current tasks
    if state.repo_manager:
        tasks = await state.repo_manager.fetch_clone_tasks()
        
        # Count by status
        queued = sum(1 for task in tasks.values() if task.status.status_type == "queued")
        active = sum(1 for task in tasks.values() if task.status.status_type == "cloning")
        completed = sum(1 for task in tasks.values() if task.status.status_type == "completed")
        failed = sum(1 for task in tasks.values() if task.status.status_type == "failed")
    else:
        tasks = {}
        queued, active, completed, failed = 0, 0, 0, 0
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "tasks": tasks,
        "queued": queued,
        "active": active,
        "completed": completed,
        "failed": failed
    })

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive, we'll broadcast updates
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Monitor task that periodically updates all clients
async def clone_monitor():
    previous_tasks = {}
    
    while True:
        if state.repo_manager:
            tasks = await state.repo_manager.fetch_clone_tasks()
            
            # Check if tasks have changed
            if tasks != previous_tasks:
                # Count by status
                queued = sum(1 for task in tasks.values() if task.status.status_type == "queued")
                active = sum(1 for task in tasks.values() if task.status.status_type == "cloning")
                completed = sum(1 for task in tasks.values() if task.status.status_type == "completed")
                failed = sum(1 for task in tasks.values() if task.status.status_type == "failed")
                
                # Create update message
                message = {
                    "tasks": tasks,
                    "summary": {
                        "queued": queued,
                        "active": active,
                        "completed": completed,
                        "failed": failed
                    }
                }
                
                # Broadcast to all clients
                import json
                await manager.broadcast(json.dumps(message, default=lambda o: o.__dict__))
                
                # Update previous tasks
                previous_tasks = tasks
                
            # Check if we're done
            if state.clone_future and state.clone_future.done():
                # Get final status once more
                tasks = await state.repo_manager.fetch_clone_tasks()
                message = {
                    "tasks": tasks,
                    "summary": {
                        "queued": 0,
                        "active": 0,
                        "completed": sum(1 for task in tasks.values() if task.status.status_type == "completed"),
                        "failed": sum(1 for task in tasks.values() if task.status.status_type == "failed")
                    },
                    "complete": True
                }
                import json
                await manager.broadcast(json.dumps(message, default=lambda o: o.__dict__))
        
        # Wait before checking again
        await asyncio.sleep(state.refresh_rate)

# API to start cloning
@app.post("/clone")
async def start_clone(repos: List[str], username: Optional[str] = "", token: Optional[str] = ""):
    """Start cloning the specified repositories."""
    if state.clone_future and not state.clone_future.done():
        return {"status": "error", "message": "Clone operation already in progress"}
    
    # Create repo manager
    state.repo_manager = RepoManager(
        urls=repos,
        github_username=username,
        github_token=token
    )
    
    # Start cloning
    state.clone_future = state.repo_manager.clone_all()
    
    # Start monitor if not already running
    if state.monitor_task is None or state.monitor_task.done():
        state.monitor_task = asyncio.create_task(clone_monitor())
    
    return {"status": "success", "message": f"Started cloning {len(repos)} repositories"}

# API to get clone status
@app.get("/status")
async def get_status():
    """Get the current status of all clone operations."""
    if not state.repo_manager:
        return {"status": "error", "message": "No clone operation in progress"}
    
    tasks = await state.repo_manager.fetch_clone_tasks()
    
    # Count by status
    queued = sum(1 for task in tasks.values() if task.status.status_type == "queued")
    active = sum(1 for task in tasks.values() if task.status.status_type == "cloning")
    completed = sum(1 for task in tasks.values() if task.status.status_type == "completed")
    failed = sum(1 for task in tasks.values() if task.status.status_type == "failed")
    
    return {
        "status": "success",
        "summary": {
            "total": len(tasks),
            "queued": queued,
            "active": active,
            "completed": completed,
            "failed": failed
        },
        "tasks": tasks
    }

# Set refresh rate
@app.post("/refresh-rate")
async def set_refresh_rate(rate: float):
    """Set the refresh rate for status updates (in seconds)."""
    if rate < 0.1:
        return {"status": "error", "message": "Refresh rate must be at least 0.1 seconds"}
    
    state.refresh_rate = rate
    return {"status": "success", "message": f"Refresh rate set to {rate} seconds"}

# Cleanup
@app.post("/cleanup")
async def cleanup():
    """Clean up clone directories."""
    if not state.repo_manager:
        return {"status": "error", "message": "No clone operation has been started"}
    
    cleanup_results = state.repo_manager.cleanup()
    return {"status": "success", "results": cleanup_results}

# Main function to start the server
def main():
    parser = argparse.ArgumentParser(description="GitFleet Web Monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Additional arguments for automatic cloning
    parser.add_argument("--repos", nargs="+", help="Repositories to clone automatically on startup")
    parser.add_argument("--github-username", help="GitHub username for authentication")
    parser.add_argument("--github-token", help="GitHub token for authentication")
    
    args = parser.parse_args()
    
    # If repos are provided, start cloning automatically
    if args.repos:
        @app.on_event("startup")
        async def startup_event():
            await start_clone(
                repos=args.repos,
                username=args.github_username or "",
                token=args.github_token or ""
            )
    
    # Start server
    uvicorn.run(
        "web_monitor:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
```

## Key Features Demonstrated

This example demonstrates several key features of GitFleet combined with web technologies:

1. **Web Dashboard**: Creating a real-time web interface for clone monitoring
2. **WebSockets**: Using WebSockets for live updates without page refreshing
3. **RESTful API**: Providing HTTP endpoints for controlling clone operations
4. **FastAPI Integration**: Combining GitFleet with the FastAPI web framework
5. **Automatic Status Updates**: Broadcasting clone status changes to connected clients
6. **Interactive UI**: Visual representation of clone progress and status

## Web Monitor Architecture

The web monitor application consists of several components:

1. **FastAPI Server**: Provides the web interface and API endpoints
2. **WebSocket Server**: Handles real-time communication with web clients
3. **GitFleet Backend**: Manages repository cloning and monitoring
4. **HTML/CSS Frontend**: Provides a user-friendly interface
5. **JavaScript Client**: Updates the UI in real-time

## Running the Example

To run this example:

1. Install GitFleet with web dependencies:
   ```bash
   pip install "gitfleet[web]"
   ```

2. Save the example code to `web_monitor.py`

3. Run the server:
   ```bash
   python web_monitor.py
   ```

4. Or run with automatic clone startup:
   ```bash
   python web_monitor.py --repos https://github.com/octocat/Hello-World.git
   ```

5. Open a web browser and navigate to [http://localhost:8000](http://localhost:8000)

## API Endpoints

The web monitor provides several API endpoints:

### `POST /clone`

Start cloning repositories.

**Parameters:**
- `repos`: List of repository URLs to clone
- `username` (optional): GitHub username for authentication
- `token` (optional): GitHub token for authentication

**Example:**
```bash
curl -X POST "http://localhost:8000/clone" \
  -H "Content-Type: application/json" \
  -d '{"repos": ["https://github.com/octocat/Hello-World.git"], "username": "", "token": ""}'
```

### `GET /status`

Get the current status of clone operations.

**Example:**
```bash
curl "http://localhost:8000/status"
```

### `POST /refresh-rate`

Set the refresh rate for status updates.

**Parameters:**
- `rate`: Refresh rate in seconds (minimum 0.1)

**Example:**
```bash
curl -X POST "http://localhost:8000/refresh-rate" \
  -H "Content-Type: application/json" \
  -d '{"rate": 2.0}'
```

### `POST /cleanup`

Clean up clone directories.

**Example:**
```bash
curl -X POST "http://localhost:8000/cleanup"
```

## WebSocket Communication

The web interface uses WebSockets for real-time updates. The WebSocket endpoint is `/ws` and provides JSON-formatted status updates:

```javascript
const ws = new WebSocket(`ws://${window.location.host}/ws`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    // Update UI with data
};
```

## UI Features

The web interface includes several UI features:

1. **Summary Dashboard**: Shows counts of repositories by status
2. **Repository Cards**: Individual cards for each repository
3. **Progress Bars**: Visual representation of clone progress
4. **Status Indicators**: Color-coded status indicators
5. **Error Display**: Detailed error information for failed clones
6. **Real-time Updates**: Live updates without page refreshing

## Extending the Example

This example can be extended in several ways:

1. **Authentication**: Add user authentication for secure access
2. **Repository Management**: Add UI for adding/removing repositories
3. **Advanced Monitoring**: Add more detailed monitoring information
4. **Git Operations**: Add interfaces for other Git operations
5. **Database Integration**: Store historical clone information
6. **Notifications**: Add email or webhook notifications for completed clones

## Related Examples

- [Clone Monitoring](clone-monitoring.md): Command-line monitoring of clone operations
- [GitHub Client](github-client.md): Working with the GitHub API client