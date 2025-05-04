# Working with Dates and Times

GitFleet provides helper methods for working with dates and times from Git provider APIs. This guide explains how to use them correctly, especially when performing date arithmetic.

## Date Fields in API Responses

Most Git provider APIs return dates and times as ISO 8601 formatted strings:

```
"created_at": "2023-05-10T17:51:29Z"
"updated_at": "2023-09-15T12:34:56Z"
```

GitFleet models store these as string fields (`created_at`, `updated_at`, `pushed_at`) to preserve the original format. However, helper methods are provided to convert these strings to Python `datetime` objects when needed.

## Converting Strings to Datetime Objects

GitFleet provides helper methods to convert date strings to `datetime` objects:

```python
# Get repository details
repo_details = await github.fetch_repository_details("octocat", "hello-world")

# Convert dates to datetime objects
created_dt = repo_details.created_datetime()
updated_dt = repo_details.updated_datetime()

if hasattr(repo_details, 'pushed_datetime'):
    pushed_dt = repo_details.pushed_datetime()
```

## Important: Timezone Awareness

The datetime objects returned by these helper methods are **timezone-aware** (with UTC timezone). This is important to understand when performing datetime arithmetic.

```python
# This will print something like: 2023-05-10 17:51:29+00:00
# Note the +00:00 indicating the UTC timezone
print(repo_details.created_datetime())
```

### Handling Timezone-Aware Datetimes

When working with these datetime objects, particularly when calculating time differences, you need to be careful about timezone awareness:

```python
# Get created_datetime (timezone-aware)
created_dt = repo_details.created_datetime()

# INCORRECT: This will raise TypeError because datetime.now() is naive
try:
    days_since = (datetime.now() - created_dt).days  # TypeError!
except TypeError as e:
    print(f"Error: {e}")  # "can't subtract offset-naive and offset-aware datetimes"
```

### Correct Approaches

#### Option 1: Remove timezone information (simplest)

```python
from datetime import datetime

# Convert timezone-aware to naive by removing tzinfo
created_dt = repo_details.created_datetime()
if created_dt:
    naive_created_dt = created_dt.replace(tzinfo=None)
    days_since = (datetime.now() - naive_created_dt).days
    print(f"Repository created {days_since} days ago")
```

#### Option 2: Make both datetimes timezone-aware

```python
from datetime import datetime, timezone

# Make both datetimes timezone-aware
created_dt = repo_details.created_datetime()
if created_dt:
    now_dt = datetime.now(tz=timezone.utc)
    days_since = (now_dt - created_dt).days
    print(f"Repository created {days_since} days ago")
```

## Date Formatting

You can format datetime objects using standard Python formatting:

```python
# Format a datetime for display
created_dt = repo_details.created_datetime()
if created_dt:
    formatted_date = created_dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Repository created on: {formatted_date}")
```

## Working with Multiple Date Fields

Many GitFleet models contain multiple date fields:

```python
# Get all available date information
if repo.created_at:
    print(f"Created: {repo.created_datetime().strftime('%Y-%m-%d')}")
    
if repo.updated_at:
    print(f"Last updated: {repo.updated_datetime().strftime('%Y-%m-%d')}")
    
if hasattr(repo, 'pushed_at') and repo.pushed_at:
    print(f"Last pushed: {repo.pushed_datetime().strftime('%Y-%m-%d')}")
```

## Rate Limit Reset Times

Rate limit information uses UNIX timestamps for reset times:

```python
rate_limit = await github.get_rate_limit()

# Get seconds until reset
seconds_until_reset = rate_limit.seconds_until_reset()
print(f"Rate limit resets in {seconds_until_reset} seconds")

# Convert to datetime (if needed)
from datetime import datetime
reset_datetime = datetime.fromtimestamp(rate_limit.reset_time)
print(f"Rate limit resets at: {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
```