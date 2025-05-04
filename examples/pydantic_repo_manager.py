#!/usr/bin/env python3
"""
Pydantic Integration Example

This example demonstrates how to use the Pydantic models with the Rust-based RepoManager:
1. Use the Rust RepoManager to perform operations
2. Convert Rust objects to Pydantic models for validation and serialization
3. Show Pydantic model features like JSON serialization and validation

Optional dependencies:
- pydantic: Required for Pydantic models (pip install pydantic)
  Install with: pip install "gitbench[pydantic]"
"""

import os
import sys
import json
import asyncio
from pprint import pprint
from datetime import datetime

# Add the parent directory to the Python path so we can import gitbench modules directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the RepoManager and Pydantic conversion utilities
from gitbench import RepoManager
from gitbench import (
    PydanticCloneStatus, PydanticCloneTask, CloneStatusType,
    to_pydantic_status, to_pydantic_task, convert_clone_tasks
)


async def main():
    # Get GitHub token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN", "")
    
    if not github_token:
        print("⚠️ GitHub token not found in environment variables.")
        print("Set the GITHUB_TOKEN environment variable to use this example.")
        print("You can create a token at: https://github.com/settings/tokens")
        return

    try:
        # Create a repository URL to clone (public repository for example)
        repo_url = "https://github.com/octocat/Hello-World.git"
        urls = [repo_url]

        # Create a RepoManager
        print("\n🔧 Creating RepoManager with Pydantic support")
        manager = RepoManager(urls, "username", github_token)
        
        # Start cloning the repository
        print("\n🔄 Cloning the repository")
        await manager.clone_all()
        
        # Get the clone tasks (returns Rust objects)
        print("\n📋 Checking clone tasks")
        rust_tasks = await manager.fetch_clone_tasks()
        
        # Convert to Pydantic models
        tasks = convert_clone_tasks(rust_tasks)
        
        # Use Pydantic serialization features
        for url, task in tasks.items():
            print(f"\n📦 Repository: {url}")
            
            # Show the task status
            status = task.status
            print(f"  Status: {status.status_type}")
            if status.status_type == CloneStatusType.CLONING:
                print(f"  Progress: {status.progress}%")
            elif status.status_type == CloneStatusType.FAILED:
                print(f"  Error: {status.error}")
            elif status.status_type == CloneStatusType.COMPLETED:
                print(f"  Temp directory: {task.temp_dir}")
            
            # Demonstrate Pydantic model serialization
            print("\n  Pydantic Model Features:")
            
            # Convert to dict
            task_dict = task.model_dump()
            print(f"  ↳ model_dump(): {list(task_dict.keys())}")
            
            # Convert to JSON
            task_json = task.model_dump_json(indent=2)
            print(f"  ↳ model_dump_json():")
            pretty_json = json.dumps(json.loads(task_json), indent=4)
            for line in pretty_json.split('\n')[:7]:
                print(f"    {line}")
            print("    ...")
            
            # Validate from Python dict
            print("\n  Validation Features:")
            try:
                # Create a PydanticCloneStatus from basic Python types
                new_status = PydanticCloneStatus(
                    status_type=CloneStatusType.COMPLETED,
                    progress=None,
                    error=None
                )
                print(f"  ↳ Created new status: {new_status.status_type}")
                
                # Create a PydanticCloneTask with validation
                new_task = PydanticCloneTask(
                    url="https://github.com/example/repo.git",
                    status=new_status,
                    temp_dir="/tmp/example"
                )
                print(f"  ↳ Created new task: {new_task.url}")
                
                # Show validation error handling (intentionally creating an error)
                print("\n  Validation Error Handling (Expected Error):")
                try:
                    # This will intentionally fail validation to demonstrate error handling
                    print("  ↳ Attempting to create a PydanticCloneStatus with invalid status_type (will fail)...")
                    invalid_status = PydanticCloneStatus(
                        status_type="invalid_status",
                        progress=None,
                        error=None
                    )
                except Exception as e:
                    print(f"  ↳ ✓ Successfully caught validation error: {str(e)[:100]}...")
                
            except Exception as e:
                print(f"  ↳ Error: {str(e)}")
        
        # Cleanup temp directories
        print("\n🧹 Cleaning up temporary directories")
        cleanup_results = manager.cleanup()
        for url, result in cleanup_results.items():
            print(f"  {url}: {'Success' if result is True else result}")
            
    except ImportError as e:
        print(f"\n❌ Error: {str(e)}")
        print("This example requires the Rust extension to be built.")
        print("Build it with: maturin develop")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())