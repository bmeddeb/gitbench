# Repository Model Validation in GitFleet

This document describes the validation approach for the repository-related models in GitFleet, including `CloneStatus`, `CloneTask`, and `RepoManager`.

## Overview

GitFleet implements a dual architecture where core functionality is written in Rust for performance, while Python code provides a user-friendly interface. The repository management models bridge these two worlds by:

1. Defining Pydantic models that mirror the Rust-generated classes
2. Providing conversion methods between Rust objects and Pydantic models
3. Implementing validation logic to ensure data integrity

## Models

### CloneStatusType

An enumeration of possible clone status types:
- `QUEUED`: The cloning task is waiting to start
- `CLONING`: The repository is currently being cloned
- `COMPLETED`: The repository has been successfully cloned
- `FAILED`: The cloning operation failed

### CloneStatus

A model representing the current status of a repository cloning operation:
- `status_type`: The type of status (from CloneStatusType)
- `progress`: The percentage of completion (0-100) if cloning, or None
- `error`: An error message if failed, or None

### CloneTask

A model representing a repository cloning task and its current status:
- `url`: The URL of the repository being cloned
- `status`: The current status (CloneStatus)
- `temp_dir`: The path to the temporary directory where the repository was cloned, or None

### RepoManager

A wrapper class that provides the same interface as the Rust RepoManager but converts the results to Pydantic models for better serialization and validation.

## Validation Approach

### Type Validation

Pydantic handles basic type validation:
- `status_type` must be a valid CloneStatusType enum value
- `progress` must be an integer or None
- `error` must be a string or None
- `url` must be a string
- `temp_dir` must be a string or None

### Semantic Validation

The models include semantic validation through their structure:
- `progress` is only meaningful when `status_type` is `CLONING`
- `error` is only meaningful when `status_type` is `FAILED`
- `temp_dir` is only set when a repository has been successfully cloned

### Conversion from Rust

The `from_rust` class methods handle conversion from Rust objects to Pydantic models:
- `CloneStatus.from_rust()` converts a Rust CloneStatus
- `CloneTask.from_rust()` converts a Rust CloneTask

## Testing

The models are tested using both unit tests and integration tests:

### Unit Tests

- Tests for model creation and validation
- Tests for enum value correctness
- Tests for field constraints

### Integration Tests with Mock Rust Objects

Since the actual Rust objects might not be available in all test environments, we use mock classes that simulate the behavior of the Rust implementations:
- `MockRustCloneStatus`
- `MockRustCloneTask`
- `MockRustRepoManager`

These mocks allow testing the Python-Rust conversion logic without requiring the actual Rust implementation.

## Debugging Tips

When debugging issues with these models:

1. Check the `status_type` values - they should match one of the enum values
2. For models with a `CLONING` status, verify that `progress` is an integer between 0 and 100
3. For models with a `FAILED` status, check the `error` field for details
4. Verify that the Rust implementation is available by checking `RUST_AVAILABLE`

## Adding New Fields

When adding new fields to these models:

1. Add the field to the Pydantic model with appropriate type hints and defaults
2. Update the `from_rust` method to handle the new field
3. Add tests for the new field
4. Update this documentation to reflect the changes