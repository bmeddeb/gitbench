#!/usr/bin/env python3
"""
Flask-based Repository Analysis Web Application

This example demonstrates how to use gitbench with Flask to create a web-based repository
analysis tool that provides:
1. Web-based repository cloning with real-time progress monitoring
2. Commit history extraction and visualization
3. Blame analysis with performance metrics
4. Interactive tables and charts for the analysis results

Prerequisites:
- Flask: For web application (pip install flask)
- pandas: For data analysis (pip install pandas)
- gitbench: With all extensions (pip install "gitbench[all]")
"""

import os
import sys
import time
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the Python path to import gitbench directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import gitbench components
from gitbench import RepoManager
from gitbench import (
    to_pydantic_task, to_pydantic_status, convert_clone_tasks,
    PydanticCloneStatus, PydanticCloneTask, CloneStatusType
)

# Import Flask and related libraries
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

# Thread pool for running async operations
thread_pool = ThreadPoolExecutor(max_workers=10)

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas is not installed. Install with: pip install pandas")
    print("Some visualizations will be limited.")

# Create Flask application
app = Flask(__name__)

# Global variables to store state
repo_managers = {}  # Dictionary to store RepoManager instances for each session
status_updates = {}  # Dictionary to store status updates for each repository
analysis_results = {}  # Dictionary to store analysis results

# Directories to analyze with blame
TARGET_DIRS = ["plots", "api"]

# Default GitHub credentials (should be set via environment variables in production)
DEFAULT_GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "")
DEFAULT_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def generate_session_id():
    """Generate a unique session ID for each analysis run."""
    import uuid
    return str(uuid.uuid4())

def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = seconds % 60
        return f"{minutes} min {int(remaining_seconds)} sec"
    else:
        hours = int(seconds / 3600)
        remaining_minutes = int((seconds % 3600) / 60)
        return f"{hours} hr {remaining_minutes} min"


# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/start_clone', methods=['POST'])
def start_clone():
    """Start cloning a repository and return a session ID"""
    data = request.json
    repo_url = data.get('repo_url')
    github_username = data.get('github_username', DEFAULT_GITHUB_USERNAME)
    github_token = data.get('github_token', DEFAULT_GITHUB_TOKEN)
    
    if not repo_url:
        return jsonify({'error': 'Repository URL is required'}), 400
    
    # Generate a unique session ID
    session_id = generate_session_id()
    
    # Initialize a RepoManager
    manager = RepoManager(
        urls=[repo_url],
        github_username=github_username,
        github_token=github_token
    )
    
    # Store the manager in global dict
    repo_managers[session_id] = {
        'manager': manager,
        'repo_url': repo_url,
        'start_time': time.time(),
        'status': 'initializing'
    }
    
    # Initialize status updates for this repo
    status_updates[session_id] = []
    
    # Start the background task to clone and monitor in a separate thread
    def run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(clone_and_monitor(session_id, repo_url))
        loop.close()
    
    thread = threading.Thread(target=run_async_task)
    thread.daemon = True
    thread.start()
    
    # Return the session ID to the client
    return jsonify({
        'session_id': session_id,
        'message': 'Cloning started',
        'repo_url': repo_url
    })


@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get the current status of a repository cloning operation"""
    if session_id not in repo_managers:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    # Get the latest status
    repo_data = repo_managers[session_id]
    status_history = status_updates[session_id]
    
    # Return the full status with history
    return jsonify({
        'session_id': session_id,
        'repo_url': repo_data['repo_url'],
        'status': repo_data['status'],
        'elapsed_time': format_duration(time.time() - repo_data['start_time']),
        'history': status_history,
        'temp_dir': repo_data.get('temp_dir')
    })


@app.route('/api/extract_commits/<session_id>', methods=['POST'])
def extract_commits(session_id):
    """Extract commit history from a cloned repository"""
    if session_id not in repo_managers:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    repo_data = repo_managers[session_id]
    
    # Check if repository is cloned
    if repo_data['status'] != 'completed' or not repo_data.get('temp_dir'):
        return jsonify({'error': 'Repository not yet cloned'}), 400
    
    # Start the background task to extract commits in a separate thread
    def run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(extract_commit_history(session_id))
        loop.close()
    
    thread = threading.Thread(target=run_async_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Commit extraction started',
        'session_id': session_id
    })


@app.route('/api/list_directories/<session_id>', methods=['GET'])
def list_directories(session_id):
    """List all available directories in the cloned repository"""
    if session_id not in repo_managers:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    repo_data = repo_managers[session_id]
    
    # Check if repository is cloned
    if repo_data['status'] != 'completed' or not repo_data.get('temp_dir'):
        return jsonify({'error': 'Repository not yet cloned'}), 400
    
    temp_dir = repo_data['temp_dir']
    
    # Get all directories in the repository
    directories = []
    for root, dirs, _ in os.walk(temp_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        # Add all directories relative to the repository root
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Convert to relative path
            rel_path = os.path.relpath(dir_path, temp_dir)
            
            # Skip hidden directories and deeply nested ones for clarity
            if not rel_path.startswith('.') and rel_path.count('/') < 3:
                directories.append(rel_path)
    
    # Add the root directory as an option
    directories.insert(0, '/')
    
    # Sort directories for better display
    directories.sort()
    
    return jsonify({
        'directories': directories
    })


@app.route('/api/run_blame/<session_id>', methods=['POST'])
def run_blame(session_id):
    """Run blame analysis on specified directories"""
    if session_id not in repo_managers:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    repo_data = repo_managers[session_id]
    
    # Check if repository is cloned
    if repo_data['status'] != 'completed' or not repo_data.get('temp_dir'):
        return jsonify({'error': 'Repository not yet cloned'}), 400
    
    # Get directories to analyze
    data = request.json
    target_dirs = data.get('directories', [])
    
    # If no directories provided, return error
    if not target_dirs:
        return jsonify({'error': 'No directories specified for analysis'}), 400
    
    # Start the background task to run blame analysis in a separate thread
    def run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_blame_analysis(session_id, target_dirs))
        loop.close()
    
    thread = threading.Thread(target=run_async_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Blame analysis started',
        'session_id': session_id,
        'directories': target_dirs
    })


@app.route('/api/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Get analysis results for a session"""
    if session_id not in analysis_results:
        return jsonify({'error': 'No results found for this session'}), 404
    
    return jsonify(analysis_results[session_id])


@app.route('/templates/index.html')
def serve_template():
    """Serve the main HTML template"""
    return render_template('index.html')


# Background tasks
async def clone_and_monitor(session_id, repo_url):
    """Clone a repository and monitor its progress."""
    repo_data = repo_managers[session_id]
    manager = repo_data['manager']
    start_time = repo_data['start_time']
    
    # Start the clone operation
    clone_future = manager.clone_all()
    
    try:
        # Monitor until complete
        while not clone_future.done():
            # Get current status
            tasks = await manager.fetch_clone_tasks()
            task = tasks.get(repo_url)
            
            if not task:
                # No task found
                update = {
                    'timestamp': time.time(),
                    'elapsed': time.time() - start_time,
                    'status': 'error',
                    'message': 'No task found for repository'
                }
                status_updates[session_id].append(update)
                repo_data['status'] = 'error'
                break
            
            # Convert to Pydantic for better handling
            pydantic_task = to_pydantic_task(task)
            status_type = pydantic_task.status.status_type
            progress = pydantic_task.status.progress
            
            # Create status update
            update = {
                'timestamp': time.time(),
                'elapsed': time.time() - start_time,
                'status': status_type,
                'progress': progress,
                'error': pydantic_task.status.error,
                'temp_dir': pydantic_task.temp_dir,
                'json': json.loads(pydantic_task.model_dump_json())
            }
            
            # Add to status updates
            status_updates[session_id].append(update)
            
            # Update main repo status
            repo_data['status'] = status_type
            if pydantic_task.temp_dir:
                repo_data['temp_dir'] = pydantic_task.temp_dir
            
            # Exit if complete or failed
            if status_type in [CloneStatusType.COMPLETED, CloneStatusType.FAILED]:
                break
            
            # Wait before checking again
            await asyncio.sleep(1)
        
        # Ensure clone is complete
        await clone_future
        
        # Final check for temp_dir
        if 'temp_dir' not in repo_data or not repo_data['temp_dir']:
            tasks = await manager.fetch_clone_tasks()
            task = tasks.get(repo_url)
            if task and task.temp_dir:
                repo_data['temp_dir'] = task.temp_dir
        
        # Report final status
        if repo_data.get('temp_dir'):
            repo_data['status'] = 'completed'
        else:
            repo_data['status'] = 'failed'
        
    except Exception as e:
        # Handle any exceptions
        update = {
            'timestamp': time.time(),
            'elapsed': time.time() - start_time,
            'status': 'error',
            'message': str(e)
        }
        status_updates[session_id].append(update)
        repo_data['status'] = 'error'


async def extract_commit_history(session_id):
    """Extract commit history from a cloned repository."""
    repo_data = repo_managers[session_id]
    manager = repo_data['manager']
    temp_dir = repo_data['temp_dir']
    
    # Initialize results if not present
    if session_id not in analysis_results:
        analysis_results[session_id] = {}
    
    # Update status
    repo_data['commit_status'] = 'extracting'
    
    try:
        # Extract commits
        start_time = time.time()
        commits = await manager.extract_commits(temp_dir)
        duration = time.time() - start_time
        
        # Process results
        if isinstance(commits, list):
            # Convert timestamps if pandas is available
            if HAS_PANDAS:
                df = pd.DataFrame(commits)
                
                # Convert Unix timestamps to datetime objects
                df['author_timestamp'] = pd.to_datetime(df['author_timestamp'], unit='s')
                df['committer_timestamp'] = pd.to_datetime(df['committer_timestamp'], unit='s')
                
                # Add derived fields
                df['date'] = df['author_timestamp'].dt.date
                df['month'] = df['author_timestamp'].dt.to_period('M')
                df['year'] = df['author_timestamp'].dt.year
                
                # Generate statistics
                # Convert time periods to strings for JSON serialization
                monthly_commits = df.groupby('month').size().tail(6)
                monthly_commits_dict = {str(k): int(v) for k, v in monthly_commits.items()}
                
                # Handle empty dataframes or missing date column correctly
                min_date = None
                max_date = None
                
                if 'date' in df.columns and not df['date'].empty:
                    min_date = df['date'].min().isoformat()
                    max_date = df['date'].max().isoformat()
                
                # Calculate totals safely
                total_additions = 0
                total_deletions = 0
                
                if 'additions' in df.columns and not df['additions'].empty:
                    # Handle non-numeric values by converting to numeric and ignoring errors
                    additions = pd.to_numeric(df['additions'], errors='coerce')
                    total_additions = int(additions.sum())
                
                if 'deletions' in df.columns and not df['deletions'].empty:
                    # Handle non-numeric values by converting to numeric and ignoring errors
                    deletions = pd.to_numeric(df['deletions'], errors='coerce')
                    total_deletions = int(deletions.sum())
                
                net_change = total_additions - total_deletions
                
                commit_stats = {
                    'count': len(df),
                    'date_range': {
                        'min': min_date,
                        'max': max_date
                    },
                    'authors': df['author_name'].value_counts().head(10).to_dict(),
                    'total_additions': total_additions,
                    'total_deletions': total_deletions,
                    'net_change': net_change,
                    'commits_by_month': monthly_commits_dict,
                    'performance': {
                        'duration_seconds': duration,
                        'duration_formatted': format_duration(duration),
                        'commits_per_second': len(df) / duration if duration > 0 else 0
                    }
                }
            else:
                # Calculate totals manually if pandas isn't available
                total_additions = 0
                total_deletions = 0
                
                for commit in commits:
                    try:
                        if 'additions' in commit and commit['additions'] is not None:
                            # Convert to integer if it's not already
                            additions = int(commit['additions'])
                            total_additions += additions
                    except (ValueError, TypeError):
                        # Skip if value cannot be converted to int
                        pass
                        
                    try:
                        if 'deletions' in commit and commit['deletions'] is not None:
                            # Convert to integer if it's not already
                            deletions = int(commit['deletions'])
                            total_deletions += deletions
                    except (ValueError, TypeError):
                        # Skip if value cannot be converted to int
                        pass
                
                # Basic stats without pandas
                commit_stats = {
                    'count': len(commits),
                    'recent_commits': commits[:10],
                    'total_additions': total_additions,
                    'total_deletions': total_deletions,
                    'net_change': total_additions - total_deletions,
                    'performance': {
                        'duration_seconds': duration,
                        'duration_formatted': format_duration(duration)
                    }
                }
            
            # Store results
            analysis_results[session_id]['commits'] = commit_stats
            repo_data['commit_status'] = 'completed'
        else:
            # Handle error
            repo_data['commit_status'] = 'failed'
            analysis_results[session_id]['commits'] = {
                'error': str(commits),
                'status': 'failed'
            }
    
    except Exception as e:
        # Handle exceptions
        repo_data['commit_status'] = 'error'
        analysis_results[session_id]['commits'] = {
            'error': str(e),
            'status': 'error'
        }


async def run_blame_analysis(session_id, target_dirs):
    """Run blame analysis on specified directories."""
    repo_data = repo_managers[session_id]
    manager = repo_data['manager']
    temp_dir = repo_data['temp_dir']
    
    # Initialize results if not present
    if session_id not in analysis_results:
        analysis_results[session_id] = {}
    
    # Update status
    repo_data['blame_status'] = 'analyzing'
    
    try:
        # Determine directories to analyze
        dirs_to_analyze = []
        for target_dir in target_dirs:
            dir_path = os.path.join(temp_dir, target_dir)
            if os.path.isdir(dir_path):
                dirs_to_analyze.append((target_dir, dir_path))
            else:
                continue  # Skip invalid directories
        
        if not dirs_to_analyze:
            # No valid directories found
            repo_data['blame_status'] = 'failed'
            analysis_results[session_id]['blame'] = {
                'error': 'No valid directories found',
                'status': 'failed'
            }
            return
        
        # Find files to analyze
        file_paths = []
        for _, dir_path in dirs_to_analyze:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith((".py", ".js", ".html", ".css", ".md")):
                        rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                        file_paths.append(rel_path)
        
        if not file_paths:
            # No suitable files found
            repo_data['blame_status'] = 'failed'
            analysis_results[session_id]['blame'] = {
                'error': 'No suitable files found',
                'status': 'failed'
            }
            return
        
        # Run blame analysis with benchmarking
        start_time = time.time()
        blame_start_time = time.time()
        blame_results = await manager.bulk_blame(temp_dir, file_paths)
        blame_duration = time.time() - blame_start_time
        total_duration = time.time() - start_time
        
        # Calculate metrics
        success_count = sum(1 for result in blame_results.values() if isinstance(result, list))
        error_count = len(blame_results) - success_count
        
        # Calculate total lines analyzed
        total_lines = 0
        for result in blame_results.values():
            if isinstance(result, list):
                total_lines += len(result)
        
        # Calculate performance metrics
        lines_per_second = total_lines / blame_duration if blame_duration > 0 else 0
        files_per_second = success_count / blame_duration if blame_duration > 0 else 0
        
        # Prepare author statistics
        author_stats = {}
        if HAS_PANDAS:
            # Convert blame data for pandas
            all_blame_data = []
            for file_path, blame_lines in blame_results.items():
                if not isinstance(blame_lines, list):
                    continue
                
                for line in blame_lines:
                    blame_row = {
                        'file': file_path,
                        'line_number': line.get('final_line_no'),
                        'author': line.get('author_name'),
                        'email': line.get('author_email'),
                        'commit': line.get('commit_id', '')[:8],
                    }
                    all_blame_data.append(blame_row)
            
            # Create DataFrame
            if all_blame_data:
                blame_df = pd.DataFrame(all_blame_data)
                
                # Generate statistics
                author_stats = {
                    'by_author': blame_df.groupby('author').size().sort_values(ascending=False).head(10).to_dict(),
                    'by_file': blame_df.groupby('file').size().sort_values(ascending=False).head(10).to_dict(),
                    'top_file_authors': {}
                }
                
                # Get author distribution for top files
                for file in blame_df['file'].value_counts().head(5).index:
                    file_df = blame_df[blame_df['file'] == file]
                    file_authors = file_df.groupby('author').size().sort_values(ascending=False).head(3)
                    author_stats['top_file_authors'][file] = {
                        'total_lines': len(file_df),
                        'authors': file_authors.to_dict()
                    }
        else:
            # Basic analysis without pandas
            all_authors = {}
            file_stats = {}
            
            for file_path, blame_lines in blame_results.items():
                if not isinstance(blame_lines, list):
                    continue
                
                file_stats[file_path] = len(blame_lines)
                
                file_authors = {}
                for line in blame_lines:
                    author = line.get('author_name')
                    
                    # Update global authors count
                    if author in all_authors:
                        all_authors[author] += 1
                    else:
                        all_authors[author] = 1
                    
                    # Update file authors count
                    if author in file_authors:
                        file_authors[author] += 1
                    else:
                        file_authors[author] = 1
            
            # Sort authors by line count
            sorted_authors = dict(sorted(all_authors.items(), key=lambda x: x[1], reverse=True)[:10])
            sorted_files = dict(sorted(file_stats.items(), key=lambda x: x[1], reverse=True)[:10])
            
            author_stats = {
                'by_author': sorted_authors,
                'by_file': sorted_files
            }
        
        # Store analysis results
        blame_analysis = {
            'status': 'completed',
            'performance': {
                'total_duration_seconds': total_duration,
                'blame_duration_seconds': blame_duration,
                'total_duration_formatted': format_duration(total_duration),
                'blame_duration_formatted': format_duration(blame_duration),
                'files_processed': success_count,
                'files_failed': error_count,
                'total_lines': total_lines,
                'lines_per_second': lines_per_second,
                'files_per_second': files_per_second
            },
            'statistics': author_stats,
            'directories_analyzed': [name for name, _ in dirs_to_analyze]
        }
        
        analysis_results[session_id]['blame'] = blame_analysis
        repo_data['blame_status'] = 'completed'
        
    except Exception as e:
        # Handle exceptions
        repo_data['blame_status'] = 'error'
        analysis_results[session_id]['blame'] = {
            'error': str(e),
            'status': 'error'
        }


# Create templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(templates_dir, exist_ok=True)

# Write HTML template
with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>gitbench Repository Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .progress {
            height: 25px;
            margin-bottom: 10px;
        }
        .card {
            margin-bottom: 20px;
        }
        .status-badge {
            font-size: 1rem;
        }
        #cloneStatus, #analysisResults {
            display: none;
        }
        .chart-container {
            height: 300px;
            margin-bottom: 20px;
        }
        .status-queued {
            background-color: #ffc107;
        }
        .status-cloning {
            background-color: #0d6efd;
        }
        .status-completed {
            background-color: #198754;
        }
        .status-failed, .status-error {
            background-color: #dc3545;
        }
        pre {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
            max-height: 400px;
            overflow-y: auto;
        }
        table {
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col">
                <h1 class="display-4">gitbench Repository Analysis</h1>
                <p class="lead">Web-based repository analysis tool powered by gitbench and Flask</p>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <!-- Repository Form -->
                <div class="card" id="repoForm">
                    <div class="card-header">
                        <h5 class="card-title">Repository Information</h5>
                    </div>
                    <div class="card-body">
                        <form id="repositoryForm">
                            <div class="mb-3">
                                <label for="repositoryUrl" class="form-label">Repository URL</label>
                                <input type="text" class="form-control" id="repositoryUrl" required
                                    placeholder="https://github.com/username/repository.git">
                            </div>
                            <div class="mb-3">
                                <label for="githubUsername" class="form-label">GitHub Username (optional)</label>
                                <input type="text" class="form-control" id="githubUsername" 
                                    placeholder="Your GitHub username">
                            </div>
                            <div class="mb-3">
                                <label for="githubToken" class="form-label">GitHub Token (optional)</label>
                                <input type="password" class="form-control" id="githubToken" 
                                    placeholder="Your GitHub personal access token">
                            </div>
                            <button type="submit" class="btn btn-primary">Clone Repository</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <!-- Clone Status -->
                <div class="card" id="cloneStatus">
                    <div class="card-header">
                        <h5 class="card-title">Clone Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <strong>Repository:</strong> <span id="statusRepoUrl"></span>
                        </div>
                        <div class="mb-3">
                            <strong>Status:</strong> 
                            <span class="badge status-badge" id="statusBadge">Initializing</span>
                        </div>
                        <div class="mb-3">
                            <strong>Elapsed Time:</strong> <span id="statusElapsedTime">0s</span>
                        </div>
                        <div class="progress-container" id="progressContainer">
                            <label>Progress:</label>
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                    id="cloneProgressBar" role="progressbar" 
                                    aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" 
                                    style="width: 0%">0%</div>
                            </div>
                        </div>
                        <div class="mb-3 mt-3" id="repoDirectoryContainer" style="display: none;">
                            <strong>Repository Directory:</strong> <span id="repoDirectory"></span>
                        </div>
                        <div class="mt-3" id="analysisButtons" style="display: none;">
                            <button class="btn btn-success" id="extractCommitsBtn">Extract Commits</button>
                            <div class="mt-3" id="blameOptions">
                                <h6>Select Directories for Blame Analysis:</h6>
                                <div class="mb-2" id="directoryOptions">
                                    <div class="spinner-border spinner-border-sm" role="status">
                                        <span class="visually-hidden">Loading directories...</span>
                                    </div>
                                    <small>Loading directories...</small>
                                </div>
                                <button class="btn btn-info" id="runBlameBtn" disabled>Run Blame Analysis</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Analysis Results -->
        <div class="row mt-4" id="analysisResults">
            <div class="col-12">
                <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="commits-tab" data-bs-toggle="tab" 
                            data-bs-target="#commits" type="button" role="tab" 
                            aria-controls="commits" aria-selected="true">Commit History</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="blame-tab" data-bs-toggle="tab" 
                            data-bs-target="#blame" type="button" role="tab" 
                            aria-controls="blame" aria-selected="false">Blame Analysis</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="performance-tab" data-bs-toggle="tab" 
                            data-bs-target="#performance" type="button" role="tab" 
                            aria-controls="performance" aria-selected="false">Performance</button>
                    </li>
                </ul>
                <div class="tab-content p-3 border border-top-0 rounded-bottom" id="analysisTabContent">
                    <!-- Commits Tab -->
                    <div class="tab-pane fade show active" id="commits" role="tabpanel" aria-labelledby="commits-tab">
                        <div id="commitsLoading">
                            <div class="d-flex justify-content-center">
                                <div class="spinner-border" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                            <p class="text-center">Extracting commit history...</p>
                        </div>
                        <div id="commitsContent" style="display: none;">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Commit Summary</h5>
                                        </div>
                                        <div class="card-body">
                                            <p><strong>Total Commits:</strong> <span id="totalCommits">0</span></p>
                                            <p><strong>Date Range:</strong> <span id="commitDateRange">N/A</span></p>
                                            <p><strong>Lines Added:</strong> <span id="linesAdded">0</span></p>
                                            <p><strong>Lines Deleted:</strong> <span id="linesDeleted">0</span></p>
                                            <p><strong>Net Change:</strong> <span id="netChange">0</span></p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Top Contributors</h5>
                                        </div>
                                        <div class="card-body">
                                            <table class="table table-sm">
                                                <thead>
                                                    <tr>
                                                        <th>Author</th>
                                                        <th>Commits</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="topContributorsTable">
                                                    <!-- Will be filled by JavaScript -->
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Commits by Month</h5>
                                        </div>
                                        <div class="card-body">
                                            <div class="chart-container">
                                                <canvas id="commitsByMonthChart"></canvas>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="commitsError" style="display: none;">
                            <div class="alert alert-danger" role="alert">
                                <h4 class="alert-heading">Error Extracting Commits</h4>
                                <p id="commitsErrorMessage">An error occurred while extracting commit history.</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Blame Tab -->
                    <div class="tab-pane fade" id="blame" role="tabpanel" aria-labelledby="blame-tab">
                        <div id="blameLoading">
                            <div class="d-flex justify-content-center">
                                <div class="spinner-border" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                            <p class="text-center">Running blame analysis...</p>
                        </div>
                        <div id="blameContent" style="display: none;">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Authors by Line Count</h5>
                                        </div>
                                        <div class="card-body">
                                            <div class="chart-container">
                                                <canvas id="authorLinesChart"></canvas>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Files by Line Count</h5>
                                        </div>
                                        <div class="card-body">
                                            <table class="table table-sm">
                                                <thead>
                                                    <tr>
                                                        <th>File</th>
                                                        <th>Lines</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="topFilesTable">
                                                    <!-- Will be filled by JavaScript -->
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5 class="card-title">Author Distribution for Top Files</h5>
                                        </div>
                                        <div class="card-body" id="fileAuthorDistribution">
                                            <!-- Will be filled by JavaScript -->
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="blameError" style="display: none;">
                            <div class="alert alert-danger" role="alert">
                                <h4 class="alert-heading">Error Running Blame Analysis</h4>
                                <p id="blameErrorMessage">An error occurred while running blame analysis.</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Performance Tab -->
                    <div class="tab-pane fade" id="performance" role="tabpanel" aria-labelledby="performance-tab">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Clone Performance</h5>
                                    </div>
                                    <div class="card-body">
                                        <p><strong>Clone Duration:</strong> <span id="cloneDuration">N/A</span></p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Commit Extraction Performance</h5>
                                    </div>
                                    <div class="card-body" id="commitPerformance">
                                        <p><strong>Duration:</strong> <span id="commitDuration">N/A</span></p>
                                        <p><strong>Commits Processed:</strong> <span id="commitsProcessed">0</span></p>
                                        <p><strong>Commits/Second:</strong> <span id="commitsPerSecond">0</span></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Blame Analysis Performance</h5>
                                    </div>
                                    <div class="card-body" id="blamePerformance">
                                        <p><strong>Total Duration:</strong> <span id="blameTotalDuration">N/A</span></p>
                                        <p><strong>Core Blame Duration:</strong> <span id="blameCoreDuration">N/A</span></p>
                                        <p><strong>Files Processed:</strong> <span id="filesProcessed">0</span></p>
                                        <p><strong>Lines Analyzed:</strong> <span id="linesAnalyzed">0</span></p>
                                        <p><strong>Files/Second:</strong> <span id="filesPerSecond">0</span></p>
                                        <p><strong>Lines/Second:</strong> <span id="linesPerSecond">0</span></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Global variables
        let sessionId = null;
        let statusInterval = null;
        let currentStatus = null;

        // DOM Elements
        const repositoryForm = document.getElementById('repositoryForm');
        const cloneStatusDiv = document.getElementById('cloneStatus');
        const analysisResultsDiv = document.getElementById('analysisResults');
        const analysisButtonsDiv = document.getElementById('analysisButtons');
        const extractCommitsBtn = document.getElementById('extractCommitsBtn');
        const runBlameBtn = document.getElementById('runBlameBtn');

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Handle form submission
            repositoryForm.addEventListener('submit', function(e) {
                e.preventDefault();
                startClone();
            });

            // Handle analysis buttons
            extractCommitsBtn.addEventListener('click', function() {
                startCommitExtraction();
            });

            runBlameBtn.addEventListener('click', function() {
                startBlameAnalysis();
            });
        });

        // Start clone operation
        function startClone() {
            const repoUrl = document.getElementById('repositoryUrl').value;
            const githubUsername = document.getElementById('githubUsername').value;
            const githubToken = document.getElementById('githubToken').value;

            // Show clone status div
            cloneStatusDiv.style.display = 'block';
            document.getElementById('statusRepoUrl').textContent = repoUrl;
            document.getElementById('statusBadge').textContent = 'Initializing';
            document.getElementById('statusBadge').className = 'badge status-badge';

            // Reset progress
            document.getElementById('cloneProgressBar').style.width = '0%';
            document.getElementById('cloneProgressBar').textContent = '0%';
            document.getElementById('repoDirectoryContainer').style.display = 'none';
            analysisButtonsDiv.style.display = 'none';
            analysisResultsDiv.style.display = 'none';

            // Send request to start clone
            fetch('/api/start_clone', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    repo_url: repoUrl,
                    github_username: githubUsername,
                    github_token: githubToken
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }

                // Store session ID and start monitoring
                sessionId = data.session_id;
                startStatusMonitoring();
            })
            .catch(error => {
                console.error('Error starting clone:', error);
                alert('Failed to start cloning. See console for details.');
            });
        }

        // Start status monitoring
        function startStatusMonitoring() {
            if (statusInterval) {
                clearInterval(statusInterval);
            }

            // Check status every second
            statusInterval = setInterval(function() {
                if (!sessionId) return;

                fetch(`/api/status/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error(data.error);
                            clearInterval(statusInterval);
                            return;
                        }

                        updateCloneStatus(data);
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                    });
            }, 1000);
        }

        // Fetch available directories for blame analysis
        function fetchDirectories() {
            if (!sessionId) return;
            
            const directoryOptions = document.getElementById('directoryOptions');
            
            fetch(`/api/list_directories/${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        directoryOptions.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }
                    
                    // Create checkboxes for directory selection
                    directoryOptions.innerHTML = '';
                    if (data.directories && data.directories.length > 0) {
                        const dirList = document.createElement('div');
                        dirList.className = 'directory-list';
                        
                        data.directories.forEach(dir => {
                            const dirItem = document.createElement('div');
                            dirItem.className = 'form-check';
                            
                            const checkbox = document.createElement('input');
                            checkbox.type = 'checkbox';
                            checkbox.className = 'form-check-input directory-checkbox';
                            checkbox.id = `dir_${dir.replace(/[^a-zA-Z0-9]/g, '_')}`;
                            checkbox.value = dir;
                            
                            const label = document.createElement('label');
                            label.className = 'form-check-label';
                            label.htmlFor = checkbox.id;
                            label.textContent = dir;
                            
                            dirItem.appendChild(checkbox);
                            dirItem.appendChild(label);
                            dirList.appendChild(dirItem);
                        });
                        
                        directoryOptions.appendChild(dirList);
                        
                        // Enable/disable blame button based on selection
                        const checkboxes = document.querySelectorAll('.directory-checkbox');
                        checkboxes.forEach(checkbox => {
                            checkbox.addEventListener('change', function() {
                                const checkedDirs = document.querySelectorAll('.directory-checkbox:checked');
                                document.getElementById('runBlameBtn').disabled = checkedDirs.length === 0;
                            });
                        });
                        
                        // Enable run blame button now that directories are loaded
                        document.getElementById('runBlameBtn').disabled = true;
                    } else {
                        directoryOptions.innerHTML = '<div class="alert alert-warning">No directories found in repository</div>';
                        document.getElementById('runBlameBtn').disabled = true;
                    }
                })
                .catch(error => {
                    console.error('Error fetching directories:', error);
                    directoryOptions.innerHTML = '<div class="alert alert-danger">Failed to load directories</div>';
                });
        }

        // Update clone status display
        function updateCloneStatus(data) {
            currentStatus = data;
            const statusBadge = document.getElementById('statusBadge');
            const statusClass = `badge status-badge status-${data.status}`;

            // Update badge
            statusBadge.textContent = data.status.toUpperCase();
            statusBadge.className = statusClass;

            // Update elapsed time
            document.getElementById('statusElapsedTime').textContent = data.elapsed_time;

            // Get latest status update
            if (data.history && data.history.length > 0) {
                const latest = data.history[data.history.length - 1];

                // Update progress bar if cloning
                if (latest.status === 'cloning' && latest.progress !== null) {
                    const progressBar = document.getElementById('cloneProgressBar');
                    progressBar.style.width = `${latest.progress}%`;
                    progressBar.textContent = `${latest.progress}%`;
                    document.getElementById('progressContainer').style.display = 'block';
                }
            }

            // If directory is available, show it
            if (data.temp_dir) {
                document.getElementById('repoDirectoryContainer').style.display = 'block';
                document.getElementById('repoDirectory').textContent = data.temp_dir;
            }

            // If completed, show analysis buttons
            if (data.status === 'completed') {
                analysisButtonsDiv.style.display = 'block';
                clearInterval(statusInterval); // Stop polling
                
                // Update performance tab with clone duration
                document.getElementById('cloneDuration').textContent = data.elapsed_time;
                
                // Fetch available directories
                fetchDirectories();
            }
        }

        // Start commit extraction
        function startCommitExtraction() {
            if (!sessionId) return;

            // Show loading state
            document.getElementById('commitsLoading').style.display = 'block';
            document.getElementById('commitsContent').style.display = 'none';
            document.getElementById('commitsError').style.display = 'none';
            
            // Show analysis results div and select commits tab
            analysisResultsDiv.style.display = 'block';
            const commitsTab = new bootstrap.Tab(document.getElementById('commits-tab'));
            commitsTab.show();

            // Send request to extract commits
            fetch(`/api/extract_commits/${sessionId}`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showCommitError(data.error);
                    return;
                }

                // Start polling for results
                pollCommitResults();
            })
            .catch(error => {
                console.error('Error starting commit extraction:', error);
                showCommitError('Failed to start commit extraction');
            });
        }

        // Poll for commit results
        function pollCommitResults() {
            const interval = setInterval(function() {
                if (!sessionId) {
                    clearInterval(interval);
                    return;
                }

                fetch(`/api/results/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            clearInterval(interval);
                            return;
                        }

                        if (data.commits) {
                            clearInterval(interval);
                            displayCommitResults(data.commits);
                        }
                    })
                    .catch(error => {
                        console.error('Error polling commit results:', error);
                        clearInterval(interval);
                        showCommitError('Failed to fetch commit results');
                    });
            }, 1000);
        }

        // Display commit results
        function displayCommitResults(commits) {
            if (commits.error || commits.status === 'error' || commits.status === 'failed') {
                showCommitError(commits.error || 'Failed to extract commits');
                return;
            }

            // Update performance tab
            document.getElementById('commitDuration').textContent = commits.performance?.duration_formatted || 'N/A';
            document.getElementById('commitsProcessed').textContent = commits.count || 0;
            document.getElementById('commitsPerSecond').textContent = 
                commits.performance?.commits_per_second?.toFixed(2) || 0;

            // Update commit summary
            document.getElementById('totalCommits').textContent = commits.count || 0;
            document.getElementById('linesAdded').textContent = commits.total_additions || 0;
            document.getElementById('linesDeleted').textContent = commits.total_deletions || 0;
            document.getElementById('netChange').textContent = commits.net_change || 0;
            
            // Update date range with proper handling for missing values
            if (commits.date_range) {
                const minDate = commits.date_range.min || 'N/A';
                const maxDate = commits.date_range.max || 'N/A';
                const dateRangeText = (minDate === 'N/A' || maxDate === 'N/A') ? 
                    'N/A' : `${minDate} to ${maxDate}`;
                document.getElementById('commitDateRange').textContent = dateRangeText;
            } else {
                document.getElementById('commitDateRange').textContent = 'N/A';
            }
            
            // Populate top contributors table
            const contributorsTable = document.getElementById('topContributorsTable');
            contributorsTable.innerHTML = '';
            
            if (commits.authors) {
                for (const [author, count] of Object.entries(commits.authors)) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${author}</td>
                        <td>${count}</td>
                    `;
                    contributorsTable.appendChild(row);
                }
            }
            
            // Create commits by month chart
            if (commits.commits_by_month) {
                const labels = Object.keys(commits.commits_by_month);
                const data = Object.values(commits.commits_by_month);
                
                const ctx = document.getElementById('commitsByMonthChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Commits',
                            data: data,
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Number of Commits'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Month'
                                }
                            }
                        }
                    }
                });
            }
            
            // Show content
            document.getElementById('commitsLoading').style.display = 'none';
            document.getElementById('commitsContent').style.display = 'block';
        }

        // Show commit error
        function showCommitError(message) {
            document.getElementById('commitsLoading').style.display = 'none';
            document.getElementById('commitsContent').style.display = 'none';
            document.getElementById('commitsError').style.display = 'block';
            document.getElementById('commitsErrorMessage').textContent = message;
        }

        // Start blame analysis
        function startBlameAnalysis() {
            if (!sessionId) return;
            
            // Get selected directories
            const selectedDirs = [];
            document.querySelectorAll('.directory-checkbox:checked').forEach(checkbox => {
                selectedDirs.push(checkbox.value);
            });
            
            // Validate selected directories
            if (selectedDirs.length === 0) {
                alert('Please select at least one directory to analyze');
                return;
            }

            // Show loading state
            document.getElementById('blameLoading').style.display = 'block';
            document.getElementById('blameContent').style.display = 'none';
            document.getElementById('blameError').style.display = 'none';
            
            // Show analysis results div and select blame tab
            analysisResultsDiv.style.display = 'block';
            const blameTab = new bootstrap.Tab(document.getElementById('blame-tab'));
            blameTab.show();

            // Send request to run blame analysis
            fetch(`/api/run_blame/${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    directories: selectedDirs
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showBlameError(data.error);
                    return;
                }

                // Start polling for results
                pollBlameResults();
            })
            .catch(error => {
                console.error('Error starting blame analysis:', error);
                showBlameError('Failed to start blame analysis');
            });
        }

        // Poll for blame results
        function pollBlameResults() {
            const interval = setInterval(function() {
                if (!sessionId) {
                    clearInterval(interval);
                    return;
                }

                fetch(`/api/results/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            clearInterval(interval);
                            return;
                        }

                        if (data.blame) {
                            clearInterval(interval);
                            displayBlameResults(data.blame);
                        }
                    })
                    .catch(error => {
                        console.error('Error polling blame results:', error);
                        clearInterval(interval);
                        showBlameError('Failed to fetch blame results');
                    });
            }, 1000);
        }

        // Display blame results
        function displayBlameResults(blame) {
            if (blame.error || blame.status === 'error' || blame.status === 'failed') {
                showBlameError(blame.error || 'Failed to run blame analysis');
                return;
            }

            // Update performance tab
            document.getElementById('blameTotalDuration').textContent = 
                blame.performance?.total_duration_formatted || 'N/A';
            document.getElementById('blameCoreDuration').textContent = 
                blame.performance?.blame_duration_formatted || 'N/A';
            document.getElementById('filesProcessed').textContent = 
                blame.performance?.files_processed || 0;
            document.getElementById('linesAnalyzed').textContent = 
                blame.performance?.total_lines || 0;
            document.getElementById('filesPerSecond').textContent = 
                blame.performance?.files_per_second?.toFixed(2) || 0;
            document.getElementById('linesPerSecond').textContent = 
                blame.performance?.lines_per_second?.toFixed(2) || 0;

            // Populate top files table
            const filesTable = document.getElementById('topFilesTable');
            filesTable.innerHTML = '';
            
            if (blame.statistics?.by_file) {
                for (const [file, count] of Object.entries(blame.statistics.by_file)) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${file}</td>
                        <td>${count}</td>
                    `;
                    filesTable.appendChild(row);
                }
            }
            
            // Create author lines chart
            if (blame.statistics?.by_author) {
                const labels = Object.keys(blame.statistics.by_author);
                const data = Object.values(blame.statistics.by_author);
                
                const ctx = document.getElementById('authorLinesChart').getContext('2d');
                new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.6)',
                                'rgba(54, 162, 235, 0.6)',
                                'rgba(255, 206, 86, 0.6)',
                                'rgba(75, 192, 192, 0.6)',
                                'rgba(153, 102, 255, 0.6)',
                                'rgba(255, 159, 64, 0.6)',
                                'rgba(199, 199, 199, 0.6)',
                                'rgba(83, 102, 255, 0.6)',
                                'rgba(40, 159, 64, 0.6)',
                                'rgba(210, 199, 199, 0.6)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)',
                                'rgba(199, 199, 199, 1)',
                                'rgba(83, 102, 255, 1)',
                                'rgba(40, 159, 64, 1)',
                                'rgba(210, 199, 199, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right',
                            },
                            title: {
                                display: true,
                                text: 'Lines of Code by Author'
                            }
                        }
                    }
                });
            }
            
            // Display author distribution for top files
            const fileAuthorDistribution = document.getElementById('fileAuthorDistribution');
            fileAuthorDistribution.innerHTML = '';
            
            if (blame.statistics?.top_file_authors) {
                for (const [file, fileData] of Object.entries(blame.statistics.top_file_authors)) {
                    const fileDiv = document.createElement('div');
                    fileDiv.className = 'mb-4';
                    
                    // File header
                    const fileHeader = document.createElement('h5');
                    fileHeader.textContent = file;
                    fileDiv.appendChild(fileHeader);
                    
                    // File total lines
                    const totalLines = document.createElement('p');
                    totalLines.className = 'small text-muted';
                    totalLines.textContent = `Total lines: ${fileData.total_lines}`;
                    fileDiv.appendChild(totalLines);
                    
                    // Author table
                    const authorTable = document.createElement('table');
                    authorTable.className = 'table table-sm';
                    authorTable.innerHTML = `
                        <thead>
                            <tr>
                                <th>Author</th>
                                <th>Lines</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    `;
                    
                    const tbody = authorTable.querySelector('tbody');
                    for (const [author, lines] of Object.entries(fileData.authors)) {
                        const percentage = ((lines / fileData.total_lines) * 100).toFixed(1);
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${author}</td>
                            <td>${lines}</td>
                            <td>${percentage}%</td>
                        `;
                        tbody.appendChild(row);
                    }
                    
                    fileDiv.appendChild(authorTable);
                    fileAuthorDistribution.appendChild(fileDiv);
                }
            }
            
            // Show content
            document.getElementById('blameLoading').style.display = 'none';
            document.getElementById('blameContent').style.display = 'block';
        }

        // Show blame error
        function showBlameError(message) {
            document.getElementById('blameLoading').style.display = 'none';
            document.getElementById('blameContent').style.display = 'none';
            document.getElementById('blameError').style.display = 'block';
            document.getElementById('blameErrorMessage').textContent = message;
        }
    </script>
</body>
</html>
""")

if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)