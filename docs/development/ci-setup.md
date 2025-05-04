# CI Setup for GitFleet

This document explains how to set up the CI environment for GitFleet, particularly for running tests that require GitHub API authentication.

## GitHub Token for CI

Several GitFleet tests require a valid GitHub API token to interact with the GitHub API. This section explains how to set up a token for CI use.

### Creating a GitHub API Token

1. Log in to your GitHub account
2. Go to Settings → Developer settings → Personal access tokens → Fine-grained tokens
3. Click "Generate new token"
4. Give your token a descriptive name, e.g., "GitFleet CI Tests"
5. Set an expiration date (or select "No expiration" for CI tokens)
6. Select the repository scope to limit the token to your repository
7. Set the following permissions:
   - Repository: Read-only access
   - Metadata: Read-only access

### Adding the Token to GitHub Secrets

To make the token available to GitHub Actions:

1. Go to your repository's settings
2. Click on "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Set the name to `GH_API_TOKEN`
5. Paste the token value in the secret value field
6. Click "Add secret"

This token will now be available to GitHub Actions workflows as `${{ secrets.GH_API_TOKEN }}`.

## Model Validation Workflow

The model validation workflow (`model-validation.yml`) automatically:

1. Runs whenever code in the `GitFleet/models/` or `GitFleet/providers/` directories is changed
2. Builds the Rust extension with maturin
3. Runs the Pydantic model validation tests
4. Uses the `GH_API_TOKEN` secret for API authentication

### Local Testing

For local testing of the model validation, you can:

1. Create a `.env` file in the project root
2. Add your GitHub token: `GITHUB_TOKEN=your_token_here`
3. Run the tests: `pytest tests/test_github_models.py -v`

The test script will automatically load the token from the `.env` file if the `GITHUB_TOKEN` environment variable is not set.

## Security Considerations

- Never commit API tokens to the repository
- Use repository secrets for storing sensitive credentials
- Set appropriate token scopes to limit access (read-only permissions when possible)
- Regularly rotate tokens, especially if used for CI/CD pipelines

## Troubleshooting CI Issues

If the GitHub API token tests fail in CI:

1. Check if the token is correctly added as a repository secret
2. Verify the token has sufficient permissions
3. Check if the token has expired
4. Check the GitHub API status page for any outages
5. Review the error message for rate limit issues