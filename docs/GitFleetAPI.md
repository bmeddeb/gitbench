# GitFleet Library Public API

This document provides a comprehensive reference for all classes, methods, and functions available to users of the GitFleet library.

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

## Data Models

### UserInfo

User information from Git providers.

**Properties:**
- `id`: User ID
- `login`: Username
- `name`: Display name
- `email`: Email address
- `avatar_url`: URL to user avatar
- `provider_type`: Provider type from `ProviderType` enum
- `raw_data`: Raw API response data

**Methods:**

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Serialize to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Serialize to JSON.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> UserInfo
```
Create from dictionary/object.

### RepoInfo

Basic repository information.

**Properties:**
- `name`: Repository name
- `full_name`: Full repository name (owner/name)
- `clone_url`: URL for cloning the repository
- `description`: Repository description
- `default_branch`: Default branch name
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `language`: Primary language
- `fork`: Whether the repository is a fork
- `forks_count`: Number of forks
- `stargazers_count`: Number of stars
- `provider_type`: Provider type from `ProviderType` enum
- `visibility`: Repository visibility (public/private)
- `owner`: Repository owner information
- `raw_data`: Raw API response data

**Methods:**

#### `created_datetime()`
```python
def created_datetime() -> Optional[datetime]
```
Parse `created_at` as datetime.

**Returns:** Datetime object or None

#### `updated_datetime()`
```python
def updated_datetime() -> Optional[datetime]
```
Parse `updated_at` as datetime.

**Returns:** Datetime object or None

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Serialize to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Serialize to JSON.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> RepoInfo
```
Create from dictionary/object.

### RepoDetails

Detailed repository information (extends `RepoInfo`).

**Additional Properties:**
- `topics`: Repository topics/tags
- `license`: Repository license
- `homepage`: Repository homepage URL
- `has_wiki`: Whether the repository has a wiki
- `has_issues`: Whether the repository has issues enabled
- `has_projects`: Whether the repository has projects enabled
- `archived`: Whether the repository is archived
- `pushed_at`: Last push timestamp
- `size`: Repository size in KB

**Additional Methods:**

#### `pushed_datetime()`
```python
def pushed_datetime() -> Optional[datetime]
```
Parse `pushed_at` as datetime.

**Returns:** Datetime object or None

### RateLimitInfo

API rate limit information.

**Properties:**
- `limit`: Total rate limit
- `remaining`: Remaining API calls
- `reset_time`: Timestamp when the rate limit resets
- `used`: Number of API calls used
- `provider_type`: Provider type from `ProviderType` enum

**Methods:**

#### `seconds_until_reset()`
```python
def seconds_until_reset() -> int
```
Get seconds until rate limit resets.

**Returns:** Number of seconds

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Serialize to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Serialize to JSON.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> RateLimitInfo
```
Create from dictionary/object.

### BranchInfo

Git branch information.

**Properties:**
- `name`: Branch name
- `commit_sha`: SHA of the branch's HEAD commit
- `protected`: Whether the branch is protected
- `provider_type`: Provider type from `ProviderType` enum
- `raw_data`: Raw API response data

**Methods:**

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Serialize to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Serialize to JSON.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> BranchInfo
```
Create from dictionary/object.

### ContributorInfo

Repository contributor information.

**Properties:**
- `id`: Contributor ID
- `login`: Contributor username
- `contributions`: Number of contributions
- `avatar_url`: URL to contributor avatar
- `provider_type`: Provider type from `ProviderType` enum
- `raw_data`: Raw API response data

**Methods:**

#### `model_dump()`
```python
def model_dump() -> Dict[str, Any]
```
Serialize to dictionary.

#### `model_dump_json()`
```python
def model_dump_json(indent: Optional[int] = None) -> str
```
Serialize to JSON.

**Parameters:**
- `indent`: Optional indentation level for JSON formatting

#### `model_validate()`
```python
@classmethod
def model_validate(cls, obj: Any) -> ContributorInfo
```
Create from dictionary/object.

## Token Management

### TokenManager

Manages API tokens and rate limits.

```python
TokenManager()
```

**Methods:**

#### `add_token()`
```python
def add_token(token: str, provider_type: ProviderType) -> None
```
Add token to the manager.

**Parameters:**
- `token`: API token
- `provider_type`: Provider type from `ProviderType` enum

#### `get_next_available_token()`
```python
async def get_next_available_token(provider_type: ProviderType) -> Optional[TokenInfo]
```
Get available token for the specified provider.

**Parameters:**
- `provider_type`: Provider type from `ProviderType` enum

**Returns:** `TokenInfo` object or None if no tokens are available

#### `update_rate_limit()`
```python
async def update_rate_limit(token: str, provider_type: ProviderType, 
                           remaining: int, reset_time: int) -> None
```
Update rate limit information for a token.

**Parameters:**
- `token`: API token
- `provider_type`: Provider type from `ProviderType` enum
- `remaining`: Remaining API calls
- `reset_time`: Timestamp when the rate limit resets

#### `mark_token_invalid()`
```python
async def mark_token_invalid(token: str, provider_type: ProviderType) -> None
```
Mark a token as invalid (e.g., revoked or expired).

**Parameters:**
- `token`: API token
- `provider_type`: Provider type from `ProviderType` enum

#### `get_all_tokens()`
```python
def get_all_tokens(provider_type: Optional[ProviderType] = None) -> List[TokenInfo]
```
Get all tokens for a provider type or all tokens if no provider specified.

**Parameters:**
- `provider_type`: Optional provider type from `ProviderType` enum

**Returns:** List of `TokenInfo` objects

### TokenInfo

Information about an API token.

**Properties:**
- `token`: Plain text token
- `provider_type`: Provider type from `ProviderType` enum
- `status`: Current token status
- `secret_token`: Property that returns token as `SecretStr` for secure handling

## Authentication and Security

### CredentialManager

Manages and securely stores credentials.

```python
CredentialManager(encryption_key: Optional[str] = None, use_encryption: bool = True)
```

**Parameters:**
- `encryption_key`: Optional encryption key for securing credentials
- `use_encryption`: Whether to encrypt stored credentials

**Methods:**

#### `add_credential()`
```python
def add_credential(provider_type: ProviderType, token: str, 
                  username: Optional[str] = None, 
                  email: Optional[str] = None) -> None
```
Add a credential to the manager.

**Parameters:**
- `provider_type`: Provider type from `ProviderType` enum
- `token`: API token
- `username`: Optional username
- `email`: Optional email

#### `get_credential()`
```python
def get_credential(provider_type: ProviderType) -> Optional[CredentialEntry]
```
Get credential for the specified provider.

**Parameters:**
- `provider_type`: Provider type from `ProviderType` enum

**Returns:** `CredentialEntry` object or None if not found

#### `list_credentials()`
```python
def list_credentials() -> List[CredentialEntry]
```
List all stored credentials.

**Returns:** List of `CredentialEntry` objects

#### `remove_credential()`
```python
def remove_credential(provider_type: ProviderType) -> bool
```
Remove credential for the specified provider.

**Parameters:**
- `provider_type`: Provider type from `ProviderType` enum

**Returns:** Boolean indicating if a credential was removed

#### `save_to_file()`
```python
def save_to_file(file_path: str) -> None
```
Save credentials to a file.

**Parameters:**
- `file_path`: Path to save credentials

#### `load_from_file()`
```python
def load_from_file(file_path: str) -> None
```
Load credentials from a file.

**Parameters:**
- `file_path`: Path to load credentials from

#### `clear_all()`
```python
def clear_all() -> None
```
Clear all stored credentials.

## Utility Functions

### Data Conversion

#### `to_dataframe()`
```python
def to_dataframe(data: Union[List[Dict[str, Any]], Dict[str, Any], 
                            List[Any], BaseModel, List[BaseModel]]) -> pandas.DataFrame
```
Convert various data types to a pandas DataFrame.

**Parameters:**
- `data`: Data to convert (list of objects, dictionary, or Pydantic models)

**Returns:** pandas DataFrame

#### `flatten_dataframe()`
```python
def flatten_dataframe(df: pandas.DataFrame, separator: str = "_") -> pandas.DataFrame
```
Flatten nested DataFrame columns.

**Parameters:**
- `df`: DataFrame to flatten
- `separator`: Separator to use for nested column names

**Returns:** Flattened pandas DataFrame

#### `to_json()`
```python
def to_json(obj: Any, indent: Optional[int] = None) -> str
```
Convert object to JSON string.

**Parameters:**
- `obj`: Object to convert
- `indent`: Optional indentation level for JSON formatting

**Returns:** JSON string

#### `to_dict()`
```python
def to_dict(obj: Any) -> Dict[str, Any]
```
Convert object to dictionary.

**Parameters:**
- `obj`: Object to convert

**Returns:** Dictionary representation

### Rate Limiting

#### `RateLimiter`
Class for rate limiting API calls.

```python
RateLimiter(requests_per_second: float)
```

**Parameters:**
- `requests_per_second`: Maximum requests per second

**Methods:**

##### `acquire()`
```python
async def acquire() -> None
```
Acquire a token (blocks if rate exceeded).

##### `try_acquire()`
```python
def try_acquire() -> bool
```
Try to acquire a token (non-blocking).

**Returns:** Boolean indicating if token was acquired

### Rust Type Conversion

#### `clone_status_to_pydantic()`
```python
def clone_status_to_pydantic(rust_status: RustCloneStatus) -> CloneStatus
```
Convert Rust `CloneStatus` to Pydantic model.

**Parameters:**
- `rust_status`: Rust `CloneStatus` object

**Returns:** Pydantic `CloneStatus` model

#### `clone_task_to_pydantic()`
```python
def clone_task_to_pydantic(rust_task: RustCloneTask) -> CloneTask
```
Convert Rust `CloneTask` to Pydantic model.

**Parameters:**
- `rust_task`: Rust `CloneTask` object

**Returns:** Pydantic `CloneTask` model

## Package Structure

The library is organized into several subpackages:
- `GitFleet` - Main package
- `GitFleet.models` - Data models
- `GitFleet.providers` - Git provider clients
- `GitFleet.utils` - Utility functions and classes

## Optional Features

The library has several optional features that can be installed:
- `pydantic` - Enhanced validation and serialization
- `pandas` - Data analysis and DataFrame support
- `crypto` - Secure credential encryption

Install with extras like: `pip install "gitfleet[pydantic,pandas]"`