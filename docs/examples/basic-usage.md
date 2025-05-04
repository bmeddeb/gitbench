# Basic Usage Example

This example demonstrates the fundamental operations of gitbench, including initializing a repository manager, cloning repositories, monitoring clone status, and analyzing repository data.

## Code Example

```python
#!/usr/bin/env python3
"""
gitbench Basic Usage Example

This example demonstrates how to use the gitbench library to:
1. Initialize a repository manager
2. Clone repositories
3. Check clone statuses
4. Analyze blame information
5. Analyze commit history
"""

import asyncio
import os

from gitbench import RepoManager

# GitHub credentials - replace with your own or use environment variables
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "your-username")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "your-personal-access-token")

# Repository URLs to analyze
REPO_URLS = [
    # Example - you can change to your repositories
    "https://github.com/bmeddeb/gradelib",
]


async def main():
    # Initialize the repository manager
    print("Initializing repository manager...")
    repo_manager = RepoManager(
        urls=REPO_URLS, github_username=GITHUB_USERNAME, github_token=GITHUB_TOKEN
    )

    # Start cloning repositories asynchronously
    print(f"Starting clone of {len(REPO_URLS)} repositories...")
    # PyO3 already returns futures, not coroutines
    clone_future = repo_manager.clone_all()

    # Monitor cloning progress with a more detailed display
    try:
        # Continue monitoring until all repos are done (completed or failed)
        previous_status = {}
        while not clone_future.done():
            # Get current status of all clone tasks
            clone_tasks = await repo_manager.fetch_clone_tasks()

            # Check if there are any active clones
            all_done = True

            # Clear terminal if supported (not on Windows)
            if os.name != "nt":
                os.system("clear")
            else:
                os.system("cls")

            print("\n===== REPOSITORY CLONE STATUS =====\n")

            for url, task in clone_tasks.items():
                status = task.status.status_type
                progress = task.status.progress

                # Status indicator
                status_indicator = {
                    "queued": "⌛ QUEUED",
                    "cloning": "🔄 CLONING",
                    "completed": "✅ COMPLETED",
                    "failed": "❌ FAILED",
                }.get(status, status.upper())

                # Print repository info
                print(f"Repository: {url}")
                print(f"Status: {status_indicator}")

                # Show progress bar for cloning status
                if status == "cloning" and progress is not None:
                    bar_length = 30
                    filled_length = int(bar_length * progress / 100)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    print(f"Progress: [{bar}] {progress}%")

                    # Print a message when progress changes significantly
                    prev_progress = 0
                    if (
                        url in previous_status
                        and previous_status[url]["status"] == "cloning"
                    ):
                        prev_progress = previous_status[url]["progress"] or 0

                    if progress - prev_progress >= 10:
                        print(f"  ↑ Progress increased by {progress - prev_progress}%")

                # Show error if failed
                if status == "failed" and task.status.error:
                    print(f"Error: {task.status.error}")

                # Show clone directory if available
                if task.temp_dir:
                    print(f"Directory: {task.temp_dir}")

                # Update status tracking
                previous_status[url] = {"status": status, "progress": progress}

                # Check if we need to continue monitoring
                if status not in ["completed", "failed"]:
                    all_done = False

                print("-" * 50)

            # Exit loop if all done
            if all_done:
                break

            # Wait before refreshing
            await asyncio.sleep(1)

        # Make sure the clone_all task completes
        await clone_future

    except KeyboardInterrupt:
        print(
            "\nMonitoring interrupted. Clone operations may continue in the background."
        )

    print("\nAll clone operations completed or failed.")

    # Find a successfully cloned repository to analyze
    repo_path = None
    clone_tasks = await repo_manager.fetch_clone_tasks()
    for _, task in clone_tasks.items():
        if task.status.status_type == "completed" and task.temp_dir:
            repo_path = task.temp_dir
            break

    if not repo_path:
        print("No repositories were cloned successfully.")
        return

    # Analyze blame for a few files in the repository
    print(f"\nAnalyzing blame information for files in {repo_path}...")
    # Find Python files in the repository (adjust for your specific case)
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_paths.append(
                    os.path.join(root, file).replace(repo_path + os.sep, "")
                )
                if len(file_paths) >= 3:  # Limit to 3 files for this example
                    break
        if len(file_paths) >= 3:
            break

    if file_paths:
        print(f"Analyzing blame for {len(file_paths)} files...")
        blame_results = await repo_manager.bulk_blame(repo_path, file_paths)

        for file_path, blame_info in blame_results.items():
            if isinstance(blame_info, list):  # Success case
                print(f"\nBlame summary for {file_path}:")
                authors = {}
                for line in blame_info:
                    author = line["author_name"]
                    if author in authors:
                        authors[author] += 1
                    else:
                        authors[author] = 1

                print("Top contributors:")
                for author, count in sorted(
                    authors.items(), key=lambda x: x[1], reverse=True
                )[:3]:
                    print(f"  {author}: {count} lines")
            else:  # Error case
                print(f"Error analyzing {file_path}: {blame_info}")

    # Analyze commit history
    print("\nAnalyzing commit history...")
    commits = await repo_manager.extract_commits(repo_path)

    if isinstance(commits, list):  # Success case
        print(f"Found {len(commits)} commits")

        # Show the 5 most recent commits
        print("\nRecent commits:")
        for commit in commits[:5]:
            # Convert timestamp to a human-readable date
            from datetime import datetime
            timestamp = commit['author_timestamp']
            date = datetime.fromtimestamp(timestamp)
            
            print(f"Commit: {commit['sha'][:7]}")
            print(f"Author: {commit['author_name']}")
            print(f"Date: {date.strftime('%Y-%m-%d %H:%M:%S')}")
            message_summary = commit["message"].split("\n")[0][:50]
            print(f"Message: {message_summary}...")
            print(f"Changes: +{commit['additions']} -{commit['deletions']}")
            print("")
    else:  # Error case
        print(f"Error analyzing commits: {commits}")


if __name__ == "__main__":
    # Set up the event loop and run the main async function
    asyncio.run(main())
```

## Key Features Demonstrated

This example demonstrates several key features of gitbench:

1. **Repository Management**: Initializing and configuring a `RepoManager` instance
2. **Asynchronous Cloning**: Non-blocking repository cloning with a future-based API
3. **Clone Progress Monitoring**: Tracking the progress of clone operations with detailed status information
4. **Blame Analysis**: Extracting and summarizing blame information for files
5. **Commit History Analysis**: Extracting and analyzing commit history information

## Running the Example

To run this example:

1. Install gitbench using pip:
   ```bash
   pip install gitbench
   ```

2. Set your GitHub credentials as environment variables:
   ```bash
   export GITHUB_USERNAME=your-username
   export GITHUB_TOKEN=your-personal-access-token
   ```

3. Run the example:
   ```bash
   python examples/basic_usage.py
   ```

## Related Examples

- [Clone Monitoring](clone-monitoring.md): More advanced techniques for monitoring clone operations
- [GitHub Client](github-client.md): Working with the GitHub API client