# GitHub REST API - cURL Commands Reference

## Authentication

### Using Authorization Header (Recommended)

```bash
curl -L \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/starred
```

### Using Basic Authentication

```bash
curl -L \
  -u username:YOUR_TOKEN_HERE \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/starred
```

### Using Environment Variable (More Secure)

Store your token:
```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

Use it in commands:
```bash
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/starred
```

## Testing Authentication

### Check if Token Works

```bash
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user
```

### Get Your Username

```bash
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/user | jq '.login'
```

## Starred Repositories

### List Your Starred Repositories

```bash
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/starred
```

### List Starred Repositories for a Specific User (Public)

```bash
curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/users/USERNAME/starred
```

### With Pagination Parameters

```bash
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/user/starred?per_page=50&page=1"
```

### With Sorting Options

```bash
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/user/starred?sort=created&direction=desc"
```

### Sort by Recently Starred

```bash
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     "https://api.github.com/user/starred?sort=created&direction=desc" | \
  jq -r '.[].full_name'
```

## Output Formatting (with jq)

### List Repo Names and Descriptions

```bash
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/starred | \
  jq '.[] | {name: .full_name, description: .description, stars: .stargazers_count}'
```

### Get Just the Repo Names

```bash
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/starred | \
  jq -r '.[].full_name'
```

### Save to File

```bash
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/starred > starred_repos.json
```

## Security Tips

- Never commit tokens to git repositories
- Store tokens in environment variables or secure vaults
- Use tokens with minimal required scopes
- Rotate tokens periodically
- Never use `-v` (verbose) flag when sharing output with authentication headers

## Token Scopes Required

For viewing starred repositories:
- `read:user` scope (for classic tokens)
- Or appropriate permissions for fine-grained tokens

## Creating a Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Select scopes: `read:user` and `user:email`
4. Generate and copy immediately (you won't see it again)

## API Endpoints Reference

- `/user` - Get authenticated user info
- `/user/starred` - List starred repositories for authenticated user
- `/users/{username}/starred` - List starred repositories for specific user (public)

## Response Format

The API returns JSON with repository information including:
- `full_name` - repository name
- `description` - repository description
- `html_url` - GitHub URL
- `stargazers_count` - number of stars
- `language` - primary language
- `updated_at` - last update time
