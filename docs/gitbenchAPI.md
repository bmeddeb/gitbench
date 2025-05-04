# gitbench Library Public API

This document provides a comprehensive reference for all classes, methods, and functions available to users of the gitbench library.

## Repository Management (Rust-powered)

### RepoManager

Core class for managing Git repositories.

```python
RepoManager(urls: List[str], github_username: str, github_token: str)
```

**Parameters:**
- `urls`: List of repository URLs to clone
- `github_username`: GitHub username for authentication
- `github_token`: GitHub token for authentication

**Methods:**

#### `clone_all()`
```python
async def clone_all() -> None
```
Asynchronously clone all repositories configured in this manager instance.

#### `fetch_clone_tasks()`
```python
async def fetch_clone_tasks() -> Dict[str, CloneTask]
```
Fetches the current status of all cloning tasks asynchronously.

**Returns:** Dictionary mapping repository URLs to `CloneTask` objects

#### `clone()`
```python
async def clone(url: str) -> None
```
Clones a single repository specified by URL asynchronously.

**Parameters:**
- `url`: Repository URL to clone

#### `bulk_blame()`
```python
async def bulk_blame(repo_path: str, file_paths: List[str]) -> Dict[str, Any]
```
Performs 'git blame' on multiple files within a cloned repository asynchronously.

**Parameters:**
- `repo_path`: Path to local repository
- `file_paths`: List of file paths to blame

**Returns:** Dictionary mapping file paths to blame information

#### `extract_commits()`
```python
async def extract_commits(repo_path: str) -> List[Dict[str, Any]]
```
Extracts commit data from a cloned repository asynchronously.

**Parameters:**
- `repo_path`: Path to local repository

**Returns:** List of commit dictionaries

#### `cleanup()`
```python
def cleanup() -> Dict[str, Union[bool, str]]
```
Cleans up all temporary directories created for cloned repositories.

**Returns:** Dictionary with repository URLs as keys and cleanup results as values

### CloneTask

Represents a repository cloning task.

**Properties:**
- `url`: Repository URL
- `status`: `CloneStatus` object representing the current status
- `temp_dir`: Temporary directory path where the repository is cloned

**Methods:**

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Convert to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Convert to JSON string.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> CloneTask
```
Create from dictionary/object.

### CloneStatus

Represents the status of a cloning operation.

**Properties:**
- `status_type`: Current status type (from `CloneStatusType` enum)
- `progress`: Percentage progress (0-100) for cloning operations
- `error`: Error message if the cloning failed

**Methods:**

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Convert to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Convert to JSON string.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> CloneStatus
```
Create from dictionary/object.

### CloneStatusType

Enum for clone status:

- `QUEUED`: Task is waiting to start
- `CLONING`: Task is in progress
- `COMPLETED`: Task completed successfully
- `FAILED`: Task failed

## Provider Clients

### GitHubClient

Client for GitHub API.

```python
GitHubClient(token: str, base_url: Optional[str] = None, 
             token_manager: Optional[TokenManager] = None, 
             use_python_impl: bool = False)
```

**Parameters:**
- `token`: GitHub personal access token
- `base_url`: Optional custom base URL for GitHub Enterprise
- `token_manager`: Optional token manager for rate limit handling
- `use_python_impl`: Force using the Python implementation even if Rust is available

**Methods:**

#### `fetch_repositories()`
```python
async def fetch_repositories(owner: str) -> List[RepoInfo]
```
Get repositories for owner.

**Parameters:**
- `owner`: GitHub username or organization name

**Returns:** List of `RepoInfo` objects

#### `fetch_user_info()`
```python
async def fetch_user_info() -> UserInfo
```
Get authenticated user info.

**Returns:** `UserInfo` object representing the authenticated user

#### `get_rate_limit()`
```python
async def get_rate_limit() -> RateLimitInfo
```
Get API rate limit info.

**Returns:** `RateLimitInfo` object with current limit information

#### `fetch_repository_details()`
```python
async def fetch_repository_details(owner: str, repo: str) -> RepoDetails
```
Get detailed repository info.

**Parameters:**
- `owner`: Repository owner username or organization
- `repo`: Repository name

**Returns:** `RepoDetails` object with detailed repository information

#### `fetch_contributors()`
```python
async def fetch_contributors(owner: str, repo: str) -> List[ContributorInfo]
```
Get repository contributors.

**Parameters:**
- `owner`: Repository owner username or organization
- `repo`: Repository name

**Returns:** List of `ContributorInfo` objects

#### `fetch_branches()`
```python
async def fetch_branches(owner: str, repo: str) -> List[BranchInfo]
```
Get repository branches.

**Parameters:**
- `owner`: Repository owner username or organization
- `repo`: Repository name

**Returns:** List of `BranchInfo` objects

#### `validate_credentials()`
```python
async def validate_credentials() -> bool
```
Check if credentials are valid.

**Returns:** Boolean indicating if the credentials are valid

### GitProviderClient

Abstract base class for Git provider clients.

```python
GitProviderClient(provider_type: ProviderType)
```

**Parameters:**
- `provider_type`: Provider type from `ProviderType` enum

**Methods:**
- Abstract methods implemented by concrete clients (see `GitHubClient`)

#### `to_pandas()`
```python
def to_pandas(data: Union[List[Any], Any]) -> pandas.DataFrame
```
Convert data to DataFrame.

**Parameters:**
- `data`: Data to convert (list of objects or single object)

**Returns:** pandas DataFrame

### ProviderType

Enum for Git provider types:

- `GITHUB`: GitHub provider
- `GITLAB`: GitLab provider
- `BITBUCKET`: BitBucket provider

## Package Structure

The library is organized into several subpackages:
- `gitbench` - Main package
- `gitbench.models` - Data models
- `gitbench.providers` - Git provider clients
- `gitbench.utils` - Utility functions and classes

## Optional Features

The library has several optional features that can be installed:
- `pydantic` - Enhanced validation and serialization
- `pandas` - Data analysis and DataFrame support
- `crypto` - Secure credential encryption

Install with extras like: `pip install "gitbench[pydantic,pandas]"`
