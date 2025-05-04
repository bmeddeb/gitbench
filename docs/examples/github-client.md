# GitHub API Client Example

This example demonstrates how to use gitbench's GitHub API client to interact with GitHub repositories, users, and other resources. It covers authentication, fetching repository data, working with contributors, branches, and using Pydantic integration for data processing.

## Code Example

```python
#!/usr/bin/env python3
"""
GitHub API Client Example

This example demonstrates how to use the gitbench GitHub API client to:
1. Create a GitHub client and validate credentials
2. Fetch repositories for a user
3. Get repository details
4. List contributors and branches
5. Convert the results to pandas DataFrames
6. Check rate limits
7. Use Pydantic model features for serialization

Optional dependencies:
- pandas: Required for DataFrame conversion (pip install pandas)
  Install with: pip install "gitbench[pandas]"
- pydantic: Required for model validation (pip install pydantic)
  Install with: pip install "gitbench[pydantic]"
"""

import os
import sys
import asyncio
import json
from pprint import pprint
from datetime import datetime

# Add the parent directory to the Python path so we can import gitbench modules directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct imports from the project modules
from gitbench.providers.github import GitHubClient
from gitbench.utils.converters import to_dataframe, to_json


async def main():
    # Get GitHub token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN", "")
    
    if not github_token:
        print("⚠️ GitHub token not found in environment variables.")
        print("Set the GITHUB_TOKEN environment variable to use this example.")
        print("You can create a token at: https://github.com/settings/tokens")
        return

    # Create a GitHub client
    client = GitHubClient(token=github_token)

    # Validate credentials
    is_valid = await client.validate_credentials()
    if not is_valid:
        print("❌ Invalid GitHub token")
        return

    # Get authenticated user info
    print("\n🧑‍💻 Authenticated User Information:")
    user = await client.fetch_user_info()
    print(f"Authenticated as: {user.login} ({user.name})")

    # Check rate limits
    print("\n📊 API Rate Limits:")
    rate_limit = await client.get_rate_limit()
    print(f"Rate Limit: {rate_limit.remaining}/{rate_limit.limit} remaining")
    print(f"Reset time: {rate_limit.reset_time}")
    
    # Get repositories for a user
    username = "octocat"  # Example GitHub user
    print(f"\n📚 Repositories for {username}:")
    repos = await client.fetch_repositories(username)

    # Print repository information
    print(f"Found {len(repos)} repositories:")
    for i, repo in enumerate(repos[:5], 1):  # Print just the first 5
        print(f"  {i}. {repo.full_name} - {repo.description or 'No description'}")
        print(f"     ↳ {repo.stargazers_count} stars, {repo.forks_count} forks, {repo.language or 'No language'}")
    
    if len(repos) > 5:
        print(f"  ... and {len(repos) - 5} more repositories")

    if repos:
        # Get detailed information about the first repository
        repo = repos[0]
        print(f"\n📖 Details for {repo.full_name}:")
        repo_details = await client.fetch_repository_details(username, repo.name)

        # Print detailed information
        print(f"  Description: {repo_details.description or 'None'}")
        print(f"  Language: {repo_details.language or 'None'}")
        print(f"  Stars: {repo_details.stargazers_count}")
        print(f"  Forks: {repo_details.forks_count}")
        print(f"  Default Branch: {repo_details.default_branch}")
        
        # Get contributors
        print(f"\n👥 Top Contributors:")
        contributors = await client.fetch_contributors(username, repo.name)
        for i, contributor in enumerate(contributors[:5], 1):  # Show top 5
            print(f"  {i}. {contributor.login} - {contributor.contributions} contributions")
        
        if len(contributors) > 5:
            print(f"  ... and {len(contributors) - 5} more contributors")
        
        # Get branches
        print(f"\n🌿 Branches:")
        branches = await client.fetch_branches(username, repo.name)
        for i, branch in enumerate(branches[:5], 1):  # Show first 5
            protected = "🔒 Protected" if branch.protected else "🔓 Not protected"
            print(f"  {i}. {branch.name} - {protected}")
        
        if len(branches) > 5:
            print(f"  ... and {len(branches) - 5} more branches")

        # Convert to pandas DataFrame (if pandas is installed)
        try:
            print("\n📊 Converting repositories to pandas DataFrame:")
            df = to_dataframe(repos)
            print(f"  DataFrame shape: {df.shape}")
            print("  Columns:", ", ".join(list(df.columns)[:10]) + "...")
            print("\n  Sample data:")
            print(df[["name", "full_name", "stargazers_count"]].head())
        except ImportError:
            print("  pandas not installed. Install with: pip install pandas")
            
        # Demonstrate Pydantic model features
        print("\n🔄 Pydantic Model Features:")
        if repos:
            repo = repos[0]
            
            # Use model_dump() to convert to dict
            print("  Model to dict:")
            repo_dict = repo.model_dump()
            print(f"  ↳ {list(repo_dict.keys())[:5]}...")
            
            # Use model_dump_json() to convert directly to JSON
            print("\n  Model to JSON:")
            repo_json = repo.model_dump_json(indent=2)
            print(f"  ↳ First 100 chars: {repo_json[:100]}...")
            
            # Use datetime conversion methods
            if repo.created_at:
                print("\n  Date/time helpers:")
                created_dt = repo.created_datetime()
                if created_dt:
                    print(f"  ↳ Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    days_since = (datetime.now() - created_dt).days
                    print(f"  ↳ Age: {days_since} days")
            
            # Show serialization with utility functions
            print("\n  Using to_json utility:")
            json_str = to_json(repo, indent=None)
            print(f"  ↳ {json_str[:100]}...")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
```

## Key Features Demonstrated

This example demonstrates the following key features of gitbench's GitHub client:

1. **Authentication**: Setting up a GitHub client with a token and validating credentials
2. **Repository Data**: Fetching repositories, repository details, contributors, and branches
3. **Rate Limit Management**: Checking and managing GitHub API rate limits
4. **Pydantic Integration**: Utilizing Pydantic models for data serialization and validation
5. **Data Conversion**: Converting API responses to pandas DataFrames and JSON
6. **Error Handling**: Proper error checking and handling of API responses

## Running the Example

To run this example:

1. Install gitbench with optional dependencies:
   ```bash
   pip install "gitbench[pandas,pydantic]"
   ```

2. Set your GitHub token as an environment variable:
   ```bash
   export GITHUB_TOKEN=your-personal-access-token
   ```

3. Run the example:
   ```bash
   python examples/github_client.py
   ```

## GitHub API Features

The gitbench GitHub client provides access to many GitHub API endpoints, including:

- **Users**: Get user information, repositories, and organizations
- **Repositories**: Get repository details, contents, and statistics
- **Branches and Commits**: List branches, get commit history
- **Contributors**: Get repository contributors and their statistics
- **Issues and Pull Requests**: Work with issues and pull requests
- **Rate Limits**: Check and manage API rate limits

## Data Models

All GitHub API responses are converted to Pydantic models, providing:

- **Type Validation**: Ensure data types are correct
- **Data Access**: Convenient attribute access to API data
- **Serialization**: Easy conversion to JSON, dictionaries, or DataFrames
- **DateTime Parsing**: Convenience methods for working with dates and times

## Related Examples

- [Token Manager](token-manager.md): Advanced token management for GitHub API
- [Basic Usage](basic-usage.md): Core gitbench usage including repository operations