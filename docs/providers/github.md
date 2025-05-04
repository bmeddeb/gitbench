# GitHub API Client

The GitFleet GitHub API client provides a convenient interface to interact with the GitHub API. This client allows you to fetch repository information, user data, and other GitHub-specific resources.

## Installation

The GitHub client is included with GitFleet. No additional installation is required.

```python
from GitFleet import GitHubClient
```

## Authentication

Initialize the client with your GitHub personal access token:

```python
github = GitHubClient(token="your-github-token")
```

You can create a token on GitHub at [https://github.com/settings/tokens](https://github.com/settings/tokens).

For GitHub Enterprise, you can specify a custom base URL:

```python
github = GitHubClient(
    token="your-github-token",
    base_url="https://github.your-company.com/api/v3"
)
```

## Basic Usage

### Fetch Repositories

Retrieve repositories for a user or organization:

```python
# Get repositories for a user (returns a List[RepoInfo])
repos = await github.fetch_repositories("octocat")

# Print repository names using dataclass attributes
for repo in repos:
    print(f"{repo.full_name} - {repo.description}")
    print(f"  Stars: {repo.stargazers_count}, Language: {repo.language}")
```

### User Information

Get information about the authenticated user:

```python
# Returns UserInfo dataclass
user = await github.fetch_user_info()
print(f"Authenticated as: {user.login}")
print(f"Name: {user.name}")
print(f"Email: {user.email}")
print(f"Avatar URL: {user.avatar_url}")
```

### Repository Details

Fetch detailed information about a specific repository:

```python
# Returns RepoDetails dataclass
repo = await github.fetch_repository_details("octocat", "hello-world")
print(f"Description: {repo.description}")
print(f"Topics: {', '.join(repo.topics)}")
print(f"License: {repo.license}")
print(f"Created: {repo.created_at}")
print(f"Updated: {repo.updated_at}")
print(f"Has Wiki: {repo.has_wiki}")
```

### Contributors and Branches

Get contributors for a repository:

```python
# Returns List[ContributorInfo]
contributors = await github.fetch_contributors("octocat", "hello-world")
for contributor in contributors:
    print(f"{contributor.login} - {contributor.contributions} contributions")
    print(f"  User ID: {contributor.id}")
```

Get branches for a repository:

```python
# Returns List[BranchInfo]
branches = await github.fetch_branches("octocat", "hello-world")
for branch in branches:
    protected = "Protected" if branch.protected else "Not protected"
    print(f"{branch.name} - {protected}")
    print(f"  Commit SHA: {branch.commit_sha}")
```

### Rate Limits

Check your current rate limit status:

```python
# Returns RateLimitInfo dataclass
rate_limit = await github.get_rate_limit()
print(f"API calls remaining: {rate_limit.remaining}/{rate_limit.limit}")
print(f"Reset time: {rate_limit.reset_time}")
print(f"Used: {rate_limit.used}")
```

### Implementation Selection

You can choose between the Rust and Python implementations:

```python
# Default: Use Rust implementation if available, fall back to Python
github = GitHubClient(token="your-token")

# Force Python implementation
github_py = GitHubClient(
    token="your-token",
    use_python_impl=True
)

# Check which implementation is being used
if hasattr(github, "_use_rust"):
    print(f"Using Rust implementation: {github._use_rust}")
```

## Error Handling

The GitHub client includes a comprehensive error hierarchy:

```python
from GitFleet.providers.base import ProviderError, AuthError, RateLimitError

try:
    repos = await github.fetch_repositories("octocat")
except AuthError as e:
    print(f"Authentication error: {e}")
    print(f"Provider type: {e.provider_type}")  # Will be ProviderType.GITHUB
except RateLimitError as e:
    print(f"Rate limit exceeded. Resets at: {e.reset_time}")
    print(f"Provider type: {e.provider_type}")  # Will be ProviderType.GITHUB
except ProviderError as e:
    print(f"Provider error: {e}")
    print(f"Provider type: {e.provider_type}")  # Will be ProviderType.GITHUB
```

You can also use provider-specific error classes:

```python
from GitFleet.providers.github import GitHubError

try:
    repos = await github.fetch_repositories("octocat")
except GitHubError as e:
    print(f"GitHub error: {e}")
```

The error hierarchy is as follows:

```
Exception
└── ProviderError (base.py)
    ├── AuthError (base.py)
    ├── RateLimitError (base.py)
    └── GitHubError (github.py)
```

## Data Analysis with Pandas

Convert API response data to pandas DataFrames for analysis:

```python
# Method 1: Using utility function (recommended)
from GitFleet import to_dataframe

repos = await github.fetch_repositories("octocat")
df = to_dataframe(repos)

# Method 2: Using client method
repos = await github.fetch_repositories("octocat")
df = await github.to_pandas(repos)

# Analyze the data
print(f"Most popular repositories (by stars):")
popular_repos = df.sort_values("stargazers_count", ascending=False)
print(popular_repos[["name", "stargazers_count", "forks_count"]].head())

# Language distribution
print("\nLanguage distribution:")
print(df["language"].value_counts())

# Filter by attributes
python_repos = df[df["language"] == "Python"]
print(f"\nPython repositories: {len(python_repos)}")

# Advanced queries
active_repos = df[(df["updated_at"] > "2023-01-01") & (df["fork"] == False)]
print(f"\nActive non-fork repos since 2023: {len(active_repos)}")
```

### Working with Contributors

```python
contributors = await github.fetch_contributors("octocat", "hello-world")
contributors_df = to_dataframe(contributors)

# Find top contributors
top_contributors = contributors_df.sort_values("contributions", ascending=False)
print(top_contributors[["login", "contributions"]].head(10))
```

### Customizing DataFrame Conversion

You can customize the DataFrame conversion by accessing the raw data:

```python
repos = await github.fetch_repositories("octocat")

# Custom conversion with selected fields
import pandas as pd
custom_data = [
    {
        "repo_name": repo.name,
        "stars": repo.stargazers_count or 0,
        "is_popular": (repo.stargazers_count or 0) > 100,
        "lang": repo.language or "Unknown"
    }
    for repo in repos
]
custom_df = pd.DataFrame(custom_data)
```

## Pagination

The GitHub client automatically handles pagination. You don't need to worry about pagination limits as the client will fetch all available results.