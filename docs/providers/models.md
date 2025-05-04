# Provider Data Models

GitFleet uses standardized data models across all Git provider APIs. These models provide consistent data structures regardless of which provider (GitHub, GitLab, BitBucket) you are working with.

## Core Data Models

All data models are implemented as Python dataclasses, providing type hints, autocompletion, and better IDE integration.

### Repository Information

```python
@dataclass
class RepoInfo:
    """Common repository information structure."""
    name: str                     # Repository name
    full_name: str                # Full name (owner/repo)
    clone_url: str                # Git clone URL
    description: Optional[str]    # Repository description
    default_branch: str           # Default branch name
    created_at: str               # Creation timestamp
    updated_at: str               # Last update timestamp
    language: Optional[str]       # Primary language
    fork: bool                    # Whether it's a fork
    forks_count: int              # Number of forks
    stargazers_count: Optional[int] # Number of stars
    provider_type: ProviderType   # Provider (GitHub, GitLab, etc.)
    visibility: str               # Visibility (public/private)
    owner: Dict[str, Any]         # Owner information
    raw_data: Optional[Dict[str, Any]] = None  # Raw provider data
```

### User Information

```python
@dataclass
class UserInfo:
    """User information structure."""
    id: str                       # User ID
    login: str                    # Username/login
    name: Optional[str]           # Full name
    email: Optional[str]          # Email address
    avatar_url: Optional[str]     # Avatar URL
    provider_type: ProviderType   # Provider (GitHub, GitLab, etc.)
    raw_data: Optional[Dict[str, Any]] = None  # Raw provider data
```

### Rate Limit Information

```python
@dataclass
class RateLimitInfo:
    """Rate limit information structure."""
    limit: int                    # Total request limit
    remaining: int                # Remaining requests
    reset_time: int               # Reset timestamp
    used: int                     # Used requests count
    provider_type: ProviderType   # Provider (GitHub, GitLab, etc.)
```

### Detailed Repository Information

```python
@dataclass
class RepoDetails(RepoInfo):
    """Detailed repository information structure."""
    topics: List[str]             # Repository topics/tags
    license: Optional[str]        # License information
    homepage: Optional[str]       # Homepage URL
    has_wiki: bool                # Whether repo has wiki
    has_issues: bool              # Whether repo has issues
    has_projects: bool            # Whether repo has projects
    archived: bool                # Whether repo is archived
    pushed_at: Optional[str]      # Last push timestamp
    size: int                     # Repository size
```

### Contributor Information

```python
@dataclass
class ContributorInfo:
    """Contributor information structure."""
    login: str                    # Username
    id: str                       # User ID
    avatar_url: Optional[str]     # Avatar URL
    contributions: int            # Number of contributions
    provider_type: ProviderType   # Provider (GitHub, GitLab, etc.)
```

### Branch Information

```python
@dataclass
class BranchInfo:
    """Branch information structure."""
    name: str                     # Branch name
    commit_sha: str               # Latest commit SHA
    protected: bool               # Whether branch is protected
    provider_type: ProviderType   # Provider (GitHub, GitLab, etc.)
```

## Working with Models

### Direct Access

Each model provides direct attribute access:

```python
# Get repositories for a user
repos = await github.fetch_repositories("octocat")

# Access model attributes directly
for repo in repos:
    print(f"Repository: {repo.full_name}")
    print(f"Description: {repo.description}")
    print(f"Stars: {repo.stargazers_count}")
```

### Working with Dates

Date fields like `created_at` and `updated_at` are stored as strings but have helper methods to convert them to datetime objects:

```python
# Get a repository
repo = repos[0]

# Convert string dates to datetime objects
if repo.created_at:
    created_dt = repo.created_datetime()
    print(f"Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Note: The datetime objects are timezone-aware
    # See the "Working with Dates and Times" guide for details
    # on handling timezone-aware datetimes
```

For complete details on working with dates, see the [Working with Dates and Times](../advanced/datetime-handling.md) guide.

### Conversion to Pandas DataFrames

All models can be converted to pandas DataFrames for data analysis:

```python
# Import pandas converter
from GitFleet import to_dataframe

# Get repositories for a user
repos = await github.fetch_repositories("octocat")

# Convert to DataFrame
df = to_dataframe(repos)

# Analyze the data
top_repos = df.sort_values("stargazers_count", ascending=False).head(10)
language_distribution = df["language"].value_counts()
```

You can also use a client-specific method:

```python
repos = await github.fetch_repositories("octocat")
df = await github.to_pandas(repos)
```

### Raw Provider Data

Each model includes the original provider data for advanced use cases:

```python
repo = await github.fetch_repository_details("octocat", "hello-world")

# Access standardized fields
print(repo.stargazers_count)

# Access raw provider data for GitHub-specific fields
if repo.raw_data and "allow_forking" in repo.raw_data:
    print(f"Allows forking: {repo.raw_data['allow_forking']}")
```

## Error Handling

When working with provider models, you may encounter provider-specific errors:

```python
from GitFleet.providers.base import ProviderError, AuthError, RateLimitError

try:
    repos = await github.fetch_repositories("octocat")
except AuthError as e:
    print(f"Authentication error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded. Resets at: {e.reset_time}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Provider Type Enumeration

The `ProviderType` enumeration is used to identify which provider a model belongs to:

```python
from GitFleet.providers import ProviderType

# Check the provider type
repos = await github.fetch_repositories("octocat")
if repos[0].provider_type == ProviderType.GITHUB:
    print("This is a GitHub repository")
```

Available provider types:
- `ProviderType.GITHUB`
- `ProviderType.GITLAB` (coming soon)
- `ProviderType.BITBUCKET` (coming soon)