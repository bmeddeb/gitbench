# Model Validation in GitFleet

This document describes how GitFleet uses Pydantic models to validate and structure data from various Git provider APIs, specifically focusing on GitHub.

## GitHub API Response Validation

GitFleet uses Pydantic models to validate and structure the responses from the GitHub API. These models ensure that the data is correctly typed and structured for use in the application.

### Models Overview

The following Pydantic models are used for GitHub API responses:

- `UserInfo`: Contains user information including ID, login, name, email, etc.
- `RepoInfo`: Basic repository information
- `RepoDetails`: Detailed repository information (extends RepoInfo)
- `RateLimitInfo`: Rate limit information
- `BranchInfo`: Branch information 
- `ContributorInfo`: Contributor information
- `CommitRef`: Reference to a commit (used in BranchInfo)

### Field Type Mapping

The models map GitHub API response fields to Pydantic field types as follows:

| GitHub API Field Type | Pydantic Field Type |
|-----------------------|---------------------|
| Integer IDs | `int` |
| Strings | `str` |
| Booleans | `bool` |
| Optional fields | `Optional[Type]` |
| Dates | `str` (ISO8601 format) |
| Nested objects | Nested Pydantic models |
| Arrays | `List[Type]` |

### Special Handling Cases

Some GitHub API responses require special handling:

1. **Rate Limit Response**: GitHub returns a `reset` field, which we map to `reset_time` in our `RateLimitInfo` model.
2. **Branch Information**: GitHub returns branch information with a nested `commit` object. We extract the SHA for backward compatibility while also storing the full commit object.
3. **Repository Details**: GitHub returns a large number of fields for repository details. We map the most commonly used fields while providing a `raw_data` field for access to all data.

## Testing Model Validation

GitFleet includes tests to verify that the Pydantic models correctly validate GitHub API responses.

### Running Tests

To run the model validation tests:

1. Set the `GITHUB_TOKEN` environment variable with a valid GitHub token.
2. Run the tests with pytest:

```bash
GITHUB_TOKEN="your_token_here" pytest tests/test_github_models.py -v
```

### Test Coverage

The tests verify:

1. **Field Types**: Tests ensure that fields have the correct types (e.g., IDs are integers, names are strings).
2. **Required Fields**: Tests verify that all required fields are present in the API responses.
3. **Optional Fields**: Tests check that optional fields can be absent without causing validation errors.
4. **Nested Objects**: Tests validate that nested objects (like owner information in repositories) are correctly parsed.
5. **Implementation Consistency**: Tests compare the Python and Rust implementations to ensure they return compatible models.

## Debugging Model Validation Issues

When debugging model validation issues:

1. **Print Raw Response**: Log the raw JSON response from the API to identify any missing or incorrectly formatted fields.
2. **Use Validation Errors**: Pydantic provides detailed validation error messages that can help identify which fields are problematic.
3. **Check Conversion Logic**: The `_convert_to_model` method in the GitHub client performs additional conversions before validation.

## Extending Models

When extending models to support new GitHub API fields:

1. Review the GitHub API documentation for the field types and requirements.
2. Add new fields to the appropriate Pydantic model with correct types and default values.
3. Update any conversion logic in the `_convert_to_model` method if needed.
4. Add tests for the new fields to verify they're correctly validated.