# Clone Monitoring

GitFleet provides a powerful system for asynchronously cloning Git repositories and monitoring the clone progress. This page explains how to monitor clone operations effectively.

## Overview

When cloning repositories with GitFleet, you can:

1. Monitor the progress of clone operations in real-time
2. Get detailed status information for each clone task
3. Handle clone failures gracefully
4. Visualize clone progress in various ways

## Clone Status Types

Each clone task can have one of the following status types:

| Status Type | Description |
|-------------|-------------|
| `queued` | The clone task is queued but not yet started |
| `cloning` | The clone operation is in progress |
| `completed` | The clone operation completed successfully |
| `failed` | The clone operation failed |

For `cloning` status, a progress percentage is also available.
For `failed` status, an error message is provided.

## Basic Clone Monitoring

The simplest way to monitor clone operations is to check the status after cloning:

```python
import asyncio
from GitFleet import RepoManager

async def main():
    # Initialize repository manager
    repo_manager = RepoManager(
        urls=["https://github.com/user/repo"],
        github_username="username",
        github_token="token"
    )
    
    # Start cloning all repositories
    await repo_manager.clone_all()
    
    # Get the status of all clone tasks
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Check each task's status
    for url, task in clone_tasks.items():
        status_type = task.status.status_type
        
        if status_type == "completed":
            print(f"✅ {url} - Cloned successfully to {task.temp_dir}")
        elif status_type == "failed":
            print(f"❌ {url} - Failed: {task.status.error}")
        else:
            print(f"⏳ {url} - Status: {status_type}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Real-Time Progress Monitoring

For a better user experience, you can monitor clone progress in real-time:

```python
import asyncio
import os
from GitFleet import RepoManager

async def main():
    # Initialize repository manager
    repo_manager = RepoManager(
        urls=["https://github.com/user/repo1", "https://github.com/user/repo2"],
        github_username="username",
        github_token="token"
    )
    
    # Start cloning all repositories (returns a future)
    clone_future = repo_manager.clone_all()
    
    # Monitor progress until all clones complete
    try:
        while not clone_future.done():
            # Get current status of all clone tasks
            clone_tasks = await repo_manager.fetch_clone_tasks()
            
            # Clear terminal (for a cleaner display)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("Repository Clone Status\n")
            
            # Flag to check if all tasks are complete
            all_done = True
            
            for url, task in clone_tasks.items():
                status = task.status.status_type
                progress = task.status.progress
                
                # Pretty status indicators
                status_icon = {
                    "queued": "⌛",
                    "cloning": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }.get(status, "❓")
                
                print(f"{status_icon} {url}")
                
                # Show progress bar for cloning status
                if status == "cloning" and progress is not None:
                    bar_length = 30
                    filled_length = int(bar_length * progress / 100)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    print(f"  Progress: [{bar}] {progress}%")
                
                # Show error if failed
                if status == "failed" and task.status.error:
                    print(f"  Error: {task.status.error}")
                
                # Show clone directory if available
                if task.temp_dir:
                    print(f"  Directory: {task.temp_dir}")
                
                print()
                
                # Check if we need to continue monitoring
                if status not in ["completed", "failed"]:
                    all_done = False
            
            # Exit the loop if all done
            if all_done:
                break
            
            # Wait before refreshing
            await asyncio.sleep(1)
        
        # Make sure the clone_all task completes
        await clone_future
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted. Clone operations may continue in the background.")
    
    print("All clone operations completed or failed.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Handling Timeouts and Cancellation

For long-running clone operations, you may want to implement timeouts:

```python
import asyncio
from GitFleet import RepoManager

async def clone_with_timeout(repo_manager, timeout=300):  # 5 minutes timeout
    # Start cloning
    clone_future = repo_manager.clone_all()
    
    try:
        # Wait for cloning to complete with timeout
        await asyncio.wait_for(clone_future, timeout=timeout)
        print("All clones completed successfully")
    except asyncio.TimeoutError:
        print(f"Clone operation timed out after {timeout} seconds")
        
        # Check status of timed-out clones
        clone_tasks = await repo_manager.fetch_clone_tasks()
        
        for url, task in clone_tasks.items():
            if task.status.status_type != "completed":
                print(f"Incomplete: {url} - Status: {task.status.status_type}")
    
    # Get final status
    return await repo_manager.fetch_clone_tasks()
```

## Visualizing Clone Progress in a Web Interface

GitFleet can be integrated with web interfaces to provide a better visualization of clone progress. Here's a simple example using Flask:

```python
from flask import Flask, jsonify
import asyncio
from GitFleet import RepoManager

app = Flask(__name__)

# Shared repo manager for the app
repo_manager = None
clone_tasks = {}

@app.route('/start_clone', methods=['POST'])
def start_clone():
    global repo_manager
    
    # Initialize repo manager with URLs from request
    repo_manager = RepoManager(
        urls=["https://github.com/user/repo1", "https://github.com/user/repo2"],
        github_username="username",
        github_token="token"
    )
    
    # Start clone in background task
    asyncio.create_task(clone_and_monitor())
    
    return jsonify({"status": "started"})

@app.route('/status', methods=['GET'])
def get_status():
    # Return current status
    return jsonify(clone_tasks)

async def clone_and_monitor():
    global clone_tasks
    
    # Start cloning
    clone_future = repo_manager.clone_all()
    
    # Monitor until complete
    while not clone_future.done():
        # Update status
        tasks = await repo_manager.fetch_clone_tasks()
        
        # Convert to serializable format
        clone_tasks = {}
        for url, task in tasks.items():
            clone_tasks[url] = {
                "status": task.status.status_type,
                "progress": task.status.progress,
                "error": task.status.error,
                "directory": task.temp_dir
            }
        
        # Wait before checking again
        await asyncio.sleep(1)
    
    # One final update
    tasks = await repo_manager.fetch_clone_tasks()
    
    # Convert to serializable format
    clone_tasks = {}
    for url, task in tasks.items():
        clone_tasks[url] = {
            "status": task.status.status_type,
            "progress": task.status.progress,
            "error": task.status.error,
            "directory": task.temp_dir
        }

# Sample HTML/JS client:
"""
<!DOCTYPE html>
<html>
<head>
    <title>Clone Monitor</title>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    statusDiv.innerHTML = '';
                    
                    for (const [url, task] of Object.entries(data)) {
                        const taskDiv = document.createElement('div');
                        taskDiv.className = 'task';
                        
                        // Add task info
                        taskDiv.innerHTML = `<h3>${url}</h3>
                                            <p>Status: ${task.status}</p>`;
                        
                        // Add progress bar if cloning
                        if (task.status === 'cloning' && task.progress !== null) {
                            taskDiv.innerHTML += `
                                <div class="progress">
                                    <div class="progress-bar" style="width: ${task.progress}%">
                                        ${task.progress}%
                                    </div>
                                </div>`;
                        }
                        
                        // Add error if failed
                        if (task.status === 'failed' && task.error) {
                            taskDiv.innerHTML += `<p class="error">Error: ${task.error}</p>`;
                        }
                        
                        // Add directory if available
                        if (task.directory) {
                            taskDiv.innerHTML += `<p>Directory: ${task.directory}</p>`;
                        }
                        
                        statusDiv.appendChild(taskDiv);
                    }
                });
        }
        
        // Update status every second
        setInterval(updateStatus, 1000);
        
        // Initial update
        document.addEventListener('DOMContentLoaded', updateStatus);
    </script>
    <style>
        .task {
            margin: 10px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .progress {
            height: 20px;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background-color: #4CAF50;
            text-align: center;
            line-height: 20px;
            color: white;
            border-radius: 5px;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
    <h1>Repository Clone Monitor</h1>
    <div id="status">Loading...</div>
</body>
</html>
"""
```

## The `CloneStatus` and `CloneTask` Classes

GitFleet provides two main classes for tracking clone operations:

### CloneStatus

```python
class CloneStatus:
    status_type: str  # "queued", "cloning", "completed", or "failed"
    progress: Optional[int]  # Percentage of completion (0-100) for "cloning" status
    error: Optional[str]  # Error message for "failed" status
```

### CloneTask

```python
class CloneTask:
    url: str  # Repository URL
    status: CloneStatus  # Current status
    temp_dir: Optional[str]  # Path to cloned repository (if completed)
```

## Clone Notifications

You can implement notifications for clone events:

```python
import asyncio
import smtplib
from email.message import EmailMessage
from GitFleet import RepoManager

async def clone_with_notifications(repo_manager, email):
    # Start cloning
    clone_future = repo_manager.clone_all()
    
    # Previous status to track changes
    previous_status = {}
    
    # Monitor until complete
    while not clone_future.done():
        # Get current status
        clone_tasks = await repo_manager.fetch_clone_tasks()
        
        # Check for status changes
        for url, task in clone_tasks.items():
            current_status = task.status.status_type
            
            # If we haven't seen this task before or status changed
            if url not in previous_status or previous_status[url] != current_status:
                # Update previous status
                previous_status[url] = current_status
                
                # Send notification for completed or failed
                if current_status in ["completed", "failed"]:
                    send_notification(
                        email,
                        f"Repository Clone {current_status.capitalize()}",
                        f"The clone of {url} has {current_status}.\n" +
                        (f"Error: {task.status.error}" if current_status == "failed" else "") +
                        (f"Directory: {task.temp_dir}" if task.temp_dir else "")
                    )
        
        # Wait before checking again
        await asyncio.sleep(5)

def send_notification(email, subject, message):
    # Simple email notification
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = 'noreply@example.com'
    msg['To'] = email
    
    # Send email
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('username', 'password')
        server.send_message(msg)
```

## Related Documentation

- [CloneStatus](../CloneStatus.md): Detailed documentation of the CloneStatus class
- [CloneTask](../CloneTask.md): Detailed documentation of the CloneTask class
- [RepoManager](../RepoManager.md): Main interface for repository operations
- [Clone Monitoring Example](../examples/clone-monitoring.md): Complete example with clone monitoring