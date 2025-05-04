#!/usr/bin/env python3
"""
gitbench Clone Monitoring Example

This example demonstrates how to use the gitbench library to:
1. Initialize a repository manager
2. Clone repositories
3. Monitor cloning progress with detailed status updates

The example shows how to implement a progress bar and real-time monitoring
of clone operations, with rich formatting.
"""

import argparse
import asyncio
import os
from datetime import datetime

from gitbench import RepoManager

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_green": "\033[42m",
    "bg_red": "\033[41m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
}


def colorize(text, color):
    """Add color to terminal text if supported"""
    if os.name == "nt":  # Windows terminals might not support ANSI
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


async def monitor_clones(repo_manager, clone_future, refresh_rate=1):
    """Monitor clone operations with a fancy display"""
    previous_status = {}
    start_time = datetime.now()

    while True:
        # Get current status of all clone tasks
        clone_tasks = await repo_manager.fetch_clone_tasks()

        # Check if all clones are done
        active_clones = 0
        completed_clones = 0
        failed_clones = 0
        queued_clones = 0

        # Clear terminal
        if os.name != "nt":
            os.system("clear")
        else:
            os.system("cls")

        # Print header
        elapsed = datetime.now() - start_time
        elapsed_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"

        print(colorize(f"\n⏱️  CLONE MONITOR - Running for {elapsed_str}", "bold"))
        print(colorize("=" * 60, "bold"))

        # For each repository, show its status
        for i, (url, task) in enumerate(clone_tasks.items(), 1):
            status = task.status.status_type
            progress = task.status.progress

            # Count by status
            if status == "cloning":
                active_clones += 1
            elif status == "completed":
                completed_clones += 1
            elif status == "failed":
                failed_clones += 1
            elif status == "queued":
                queued_clones += 1

            # Get repo name from URL
            repo_name = url.split("/")[-1]

            # Status formatting
            status_color = {
                "queued": "yellow",
                "cloning": "blue",
                "completed": "green",
                "failed": "red",
            }.get(status, "white")

            status_text = {
                "queued": "⌛ QUEUED",
                "cloning": "🔄 CLONING",
                "completed": "✅ COMPLETED",
                "failed": "❌ FAILED",
            }.get(status, status.upper())

            # Get previous progress for comparison
            prev_progress = 0
            if url in previous_status and previous_status[url]["status"] == "cloning":
                prev_progress = previous_status[url]["progress"] or 0

            # Print repository header
            print(
                f"\n{i}. {colorize(repo_name, 'bold')} [{colorize(status_text, status_color)}]"
            )
            print(f"   URL: {url}")

            # Show progress bar for active clones
            if status == "cloning" and progress is not None:
                bar_length = 40
                filled_length = int(bar_length * progress / 100)

                # Format progress bar
                bar_content = colorize("█" * filled_length, "bg_green") + "░" * (
                    bar_length - filled_length
                )

                # Show progress change indicators
                progress_change = ""
                if progress > prev_progress:
                    progress_change = colorize(
                        f" ↑{progress - prev_progress}%", "green"
                    )

                # Print progress information
                print(f"   Progress: [{bar_content}] {progress}%{progress_change}")

                # Estimated time (very rough)
                if progress > 0:
                    elapsed_seconds = elapsed.total_seconds()
                    estimated_total = elapsed_seconds * 100 / progress
                    remaining = estimated_total - elapsed_seconds
                    if remaining > 0:
                        remaining_min = int(remaining // 60)
                        remaining_sec = int(remaining % 60)
                        print(
                            f"   Estimated time remaining: ~{remaining_min}m {remaining_sec}s"
                        )

            # Show error details for failed clones
            if status == "failed" and task.status.error:
                print(f"   Error: {colorize(task.status.error, 'red')}")

            # Show clone directory if completed
            if task.temp_dir:
                print(f"   Directory: {task.temp_dir}")

            # Store current status for the next iteration
            previous_status[url] = {"status": status, "progress": progress}

        # Print summary
        print(colorize("\n" + "=" * 60, "bold"))
        print(colorize("SUMMARY:", "bold"))
        print(f"Total repositories: {len(clone_tasks)}")
        print(f"Active clones: {colorize(str(active_clones), 'blue')}")
        print(f"Completed: {colorize(str(completed_clones), 'green')}")
        print(f"Failed: {colorize(str(failed_clones), 'red')}")
        print(f"Queued: {colorize(str(queued_clones), 'yellow')}")

        # Exit condition: no active or queued clones
        if (active_clones == 0 and queued_clones == 0) or clone_future.done():
            print(colorize("\nAll clone operations have completed!", "green"))
            break

        # Wait before refreshing
        await asyncio.sleep(refresh_rate)

    return clone_tasks


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Clone and monitor Git repositories")
    parser.add_argument(
        "--repos",
        "-r",
        nargs="+",
        default=[
            "https://github.com/bmeddeb/gradelib",
            "https://github.com/bmeddeb/SER402-Team3",
            "https://github.com/bmeddeb/gitbench",
        ],
        help="List of repository URLs to clone",
    )
    parser.add_argument(
        "--username",
        "-u",
        default=os.environ.get("GITHUB_USERNAME", ""),
        help="GitHub username (or set GITHUB_USERNAME env var)",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--refresh",
        "-f",
        type=float,
        default=1.0,
        help="Refresh rate in seconds for the monitoring display",
    )

    args = parser.parse_args()

    # Initialize the repository manager
    print(colorize("Initializing repository manager...", "bold"))

    if not args.username or not args.token:
        print(
            colorize(
                "Warning: GitHub credentials not provided. Anonymous access will be used.",
                "yellow",
            )
        )
        print(
            "For better rate limits, provide credentials with --username and --token options"
        )
        print("or set GITHUB_USERNAME and GITHUB_TOKEN environment variables.\n")

    repo_manager = RepoManager(
        urls=args.repos, github_username=args.username, github_token=args.token
    )

    # Start cloning repositories
    print(colorize(f"Starting clone of {len(args.repos)} repositories...", "bold"))
    # PyO3 already returns futures, not coroutines
    clone_future = repo_manager.clone_all()

    try:
        # Monitor cloning progress
        final_status = await monitor_clones(repo_manager, clone_future, args.refresh)

        # Make sure clone_all completes
        await clone_future

        # Print final status
        print(colorize("\nFinal Status:", "bold"))
        for url, task in final_status.items():
            status_color = "green" if task.status.status_type == "completed" else "red"
            print(f"• {url}: {colorize(task.status.status_type.upper(), status_color)}")

    except KeyboardInterrupt:
        print(
            colorize(
                "\nMonitoring interrupted. Clone operations may continue in the background.",
                "yellow",
            )
        )
        print("You can run this script again to resume monitoring.")


if __name__ == "__main__":
    # Set up the event loop and run the main async function
    asyncio.run(main())
