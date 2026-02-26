# PyGithub Scaffolding - Starred Repositories & Contents

A comprehensive guide for using PyGithub to access starred repositories and their contents.

## Installation

```bash
pip install PyGithub
```

## Basic Authentication

```python
from github import Github, Auth

# Create authentication token
auth = Auth.Token("your_github_token")

# Initialize Github instance
g = Github(auth=auth)

# Always close connection when done
g.close()
```

## Using Context Manager (Recommended)

```python
from github import Github, Auth
import os

# Load token from environment
token = os.getenv("GITHUB_TOKEN")
auth = Auth.Token(token)

# Use context manager for automatic cleanup
with Github(auth=auth) as g:
    # Your code here
    user = g.get_user()
    print(user.login)
```

## Getting Starred Repositories

### Get Your Own Starred Repos

```python
from github import Github, Auth
import os

def get_my_starred_repos():
    """Get all repositories starred by the authenticated user"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        # Get authenticated user
        user = g.get_user()

        # Get starred repositories
        starred_repos = user.get_starred()

        print(f"You have starred {starred_repos.totalCount} repositories\n")

        for repo in starred_repos:
            print(f"Name: {repo.full_name}")
            print(f"Description: {repo.description}")
            print(f"Stars: {repo.stargazers_count}")
            print(f"Language: {repo.language}")
            print(f"URL: {repo.html_url}")
            print("-" * 50)

if __name__ == "__main__":
    get_my_starred_repos()
```

### Get Another User's Starred Repos

```python
from github import Github, Auth
import os

def get_user_starred_repos(username):
    """Get repositories starred by a specific user"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        # Get specific user
        user = g.get_user(username)

        # Get their starred repositories
        starred_repos = user.get_starred()

        print(f"{username} has starred {starred_repos.totalCount} repositories\n")

        for repo in starred_repos:
            print(f"{repo.full_name} - {repo.stargazers_count} ⭐")

if __name__ == "__main__":
    get_user_starred_repos("torvalds")
```

### Get Starred Repos with Timestamps

```python
from github import Github, Auth
import os

def get_starred_with_timestamps():
    """Get starred repositories with timestamp information"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        user = g.get_user()
        starred = user.get_starred_with_date()

        for starred_repo in starred:
            repo = starred_repo.repository
            starred_at = starred_repo.starred_at

            print(f"Repo: {repo.full_name}")
            print(f"Starred at: {starred_at}")
            print()

if __name__ == "__main__":
    get_starred_with_timestamps()
```

## Accessing Repository Contents

### Get Root Contents

```python
from github import Github, Auth
import os

def get_repo_contents(repo_name):
    """Get contents of a repository's root directory"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        # Get repository
        repo = g.get_repo(repo_name)

        # Get root contents
        contents = repo.get_contents("")

        print(f"Contents of {repo_name}:\n")

        for content in contents:
            print(f"{'[DIR]' if content.type == 'dir' else '[FILE]'} {content.path}")

if __name__ == "__main__":
    get_repo_contents("PyGithub/PyGithub")
```

### Get Contents of Specific Directory

```python
from github import Github, Auth
import os

def get_directory_contents(repo_name, path):
    """Get contents of a specific directory in a repository"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        repo = g.get_repo(repo_name)

        # Get contents of specific path
        contents = repo.get_contents(path)

        print(f"Contents of {repo_name}/{path}:\n")

        for content in contents:
            print(f"{content.type.upper()}: {content.name}")

if __name__ == "__main__":
    get_directory_contents("PyGithub/PyGithub", "github")
```

### Read File Contents

```python
from github import Github, Auth
import os

def read_file_content(repo_name, file_path):
    """Read the content of a specific file"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        repo = g.get_repo(repo_name)

        # Get file contents
        file_content = repo.get_contents(file_path)

        # Decode content
        content = file_content.decoded_content.decode('utf-8')

        print(f"Content of {file_path}:\n")
        print(content)

        # Additional file info
        print(f"\nFile size: {file_content.size} bytes")
        print(f"SHA: {file_content.sha}")

if __name__ == "__main__":
    read_file_content("PyGithub/PyGithub", "README.md")
```

### Recursively Get All Files

```python
from github import Github, Auth
import os

def get_all_files_recursive(repo_name):
    """Recursively get all files in a repository"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        repo = g.get_repo(repo_name)

        # Start with root contents
        contents = repo.get_contents("")
        all_files = []

        while contents:
            file_content = contents.pop(0)

            if file_content.type == "dir":
                # If directory, add its contents to the list
                contents.extend(repo.get_contents(file_content.path))
            else:
                # If file, add to results
                all_files.append(file_content.path)

        print(f"Found {len(all_files)} files in {repo_name}:\n")
        for file_path in all_files:
            print(file_path)

        return all_files

if __name__ == "__main__":
    get_all_files_recursive("PyGithub/PyGithub")
```

## Complete Example: Starred Repos + Contents

```python
from github import Github, Auth
import os
from typing import List

def analyze_starred_repos(max_repos: int = 5):
    """
    Get starred repositories and analyze their contents
    """
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        user = g.get_user()
        starred_repos = user.get_starred()

        print(f"Analyzing your starred repositories...\n")

        # Process first N repos
        for i, repo in enumerate(starred_repos[:max_repos]):
            print(f"\n{'='*60}")
            print(f"Repository {i+1}: {repo.full_name}")
            print(f"{'='*60}")
            print(f"Description: {repo.description}")
            print(f"Stars: {repo.stargazers_count}")
            print(f"Language: {repo.language}")
            print(f"URL: {repo.html_url}")

            # Get repository contents
            try:
                contents = repo.get_contents("")
                print(f"\nRoot files/directories ({len(contents)}):")

                for content in contents[:10]:  # Show first 10 items
                    print(f"  - {content.type}: {content.name}")

                if len(contents) > 10:
                    print(f"  ... and {len(contents) - 10} more items")

                # Check for README
                try:
                    readme = repo.get_readme()
                    print(f"\n✓ README.md found ({readme.size} bytes)")
                except:
                    print(f"\n✗ No README.md found")

            except Exception as e:
                print(f"\nCouldn't access contents: {e}")

if __name__ == "__main__":
    analyze_starred_repos(max_repos=5)
```

## Advanced: Filter Starred Repos by Language

```python
from github import Github, Auth
import os
from collections import Counter

def filter_starred_by_language(language: str = None):
    """
    Filter starred repositories by programming language
    """
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        user = g.get_user()
        starred_repos = user.get_starred()

        if language:
            # Filter by specific language
            filtered = [repo for repo in starred_repos if repo.language == language]
            print(f"Found {len(filtered)} {language} repositories:\n")

            for repo in filtered:
                print(f"- {repo.full_name}: {repo.description}")
        else:
            # Count by language
            languages = [repo.language for repo in starred_repos if repo.language]
            language_counts = Counter(languages)

            print("Starred repositories by language:\n")
            for lang, count in language_counts.most_common():
                print(f"{lang}: {count} repos")

if __name__ == "__main__":
    # Show all languages
    filter_starred_by_language()

    # Or filter by specific language
    # filter_starred_by_language("Python")
```

## Download File from Repository

```python
from github import Github, Auth
import os

def download_file(repo_name: str, file_path: str, save_path: str):
    """
    Download a specific file from a repository
    """
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        repo = g.get_repo(repo_name)

        # Get file contents
        file_content = repo.get_contents(file_path)

        # Decode and save
        with open(save_path, 'wb') as f:
            f.write(file_content.decoded_content)

        print(f"Downloaded {file_path} to {save_path}")
        print(f"Size: {file_content.size} bytes")

if __name__ == "__main__":
    download_file("PyGithub/PyGithub", "README.md", "./downloaded_readme.md")
```

## Common Repository Attributes

```python
from github import Github, Auth
import os

def show_repo_attributes(repo_name: str):
    """Display common attributes of a repository"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        repo = g.get_repo(repo_name)

        print(f"Repository: {repo.full_name}\n")
        print(f"Owner: {repo.owner.login}")
        print(f"Description: {repo.description}")
        print(f"Language: {repo.language}")
        print(f"Stars: {repo.stargazers_count}")
        print(f"Forks: {repo.forks_count}")
        print(f"Open Issues: {repo.open_issues_count}")
        print(f"Watchers: {repo.watchers_count}")
        print(f"Size: {repo.size} KB")
        print(f"Created: {repo.created_at}")
        print(f"Updated: {repo.updated_at}")
        print(f"Default Branch: {repo.default_branch}")
        print(f"License: {repo.license.name if repo.license else 'None'}")
        print(f"Topics: {', '.join(repo.get_topics())}")
        print(f"Homepage: {repo.homepage or 'None'}")
        print(f"Private: {repo.private}")
        print(f"Fork: {repo.fork}")
        print(f"Archived: {repo.archived}")

if __name__ == "__main__":
    show_repo_attributes("PyGithub/PyGithub")
```

## Error Handling

```python
from github import Github, Auth, GithubException
import os

def safe_repo_access(repo_name: str):
    """Safely access repository with error handling"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    try:
        with Github(auth=auth) as g:
            repo = g.get_repo(repo_name)
            print(f"Successfully accessed {repo.full_name}")

            # Try to get contents
            try:
                contents = repo.get_contents("")
                print(f"Root has {len(contents)} items")
            except GithubException as e:
                if e.status == 404:
                    print("Repository is empty or contents not accessible")
                else:
                    print(f"Error accessing contents: {e}")

    except GithubException as e:
        if e.status == 404:
            print(f"Repository '{repo_name}' not found")
        elif e.status == 401:
            print("Authentication failed - check your token")
        elif e.status == 403:
            print("Access forbidden - you don't have permission")
        else:
            print(f"GitHub API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    safe_repo_access("PyGithub/PyGithub")
```

## Pagination Handling

```python
from github import Github, Auth
import os

def handle_pagination(username: str, max_repos: int = None):
    """Handle pagination for starred repositories"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

    with Github(auth=auth) as g:
        user = g.get_user(username)
        starred = user.get_starred()

        print(f"Total starred: {starred.totalCount}")

        count = 0
        for repo in starred:
            print(f"{count + 1}. {repo.full_name}")
            count += 1

            # Stop at max if specified
            if max_repos and count >= max_repos:
                break

        print(f"\nProcessed {count} repositories")

if __name__ == "__main__":
    handle_pagination("torvalds", max_repos=10)
```

## Key Methods Summary

### User Methods
- `get_starred()` - Get starred repositories
- `get_starred_with_date()` - Get starred repos with timestamps
- `get_repos()` - Get user's repositories

### Repository Methods
- `get_contents(path)` - Get contents at path
- `get_readme()` - Get README file
- `get_topics()` - Get repository topics
- `get_languages()` - Get languages used

### ContentFile Methods
- `decoded_content` - Get decoded file content
- `type` - "file" or "dir"
- `path` - File/directory path
- `size` - Size in bytes
- `sha` - Git SHA hash

## Environment Setup

```bash
# Set your token
export GITHUB_TOKEN="your_token_here"

# Or use .env file
echo "GITHUB_TOKEN=your_token_here" > .env
```

```python
# Load from .env
from dotenv import load_dotenv
load_dotenv()
```

## Resources

- [PyGithub Documentation](https://pygithub.readthedocs.io/en/stable/index.html)
- [PyGithub Repository](https://github.com/PyGithub/PyGithub)
- [Repository Examples](https://pygithub.readthedocs.io/en/latest/examples/Repository.html)
- [GitHub API Documentation](https://docs.github.com/en/rest)

## Synchronous vs Asynchronous: How PyGithub Works

### PyGithub is Synchronous

**PyGithub is a synchronous library** - it does NOT use async/await. All HTTP requests are blocking.

```python
# This is SYNCHRONOUS (blocking) code
for repo in starred_repos:  # Each iteration may make an HTTP request
    print(repo.full_name)   # This blocks until the request completes
```

### How Lazy Pagination Works

When you call `get_starred()`, it returns a `PaginatedList` object:

```python
starred_repos = user.get_starred()  # No API call yet!
# Returns: <github.PaginatedList.PaginatedList object>
```

The actual API requests happen **lazily** as you iterate:

```python
for repo in starred_repos:  # API call happens HERE during iteration
    print(repo.full_name)
```

#### Behind the Scenes

1. **First iteration** - Fetches page 1 (default 30 items)
2. **Item 31** - Triggers fetch of page 2
3. **Item 61** - Triggers fetch of page 3
4. And so on...

Each page fetch is a **synchronous, blocking HTTP request**.

### Visualizing the Difference

#### Synchronous (PyGithub)

```python
# Time flows downward →
starred_repos = user.get_starred()

for repo in starred_repos:  # ← Blocks here
    print(repo.name)        # Waits for HTTP request to complete
    # Can't do anything else during the request
```

#### Asynchronous (aiohttp - alternative approach)

```python
# Can handle multiple requests concurrently
async for repo in starred_repos:  # ← Doesn't block
    print(repo.name)              # Other tasks can run during request
```

### Testing It Yourself

```python
from github import Github, Auth
import os
import time

auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

with Github(auth=auth) as g:
    user = g.get_user()
    starred_repos = user.get_starred()

    print(f"Total count: {starred_repos.totalCount}")
    print("Starting iteration...\n")

    for i, repo in enumerate(starred_repos):
        if i % 30 == 0:  # Every 30 repos (one page)
            print(f"Fetching page {i//30 + 1}...")
            time.sleep(0.1)  # You'll notice the pause here

        print(f"{i+1}. {repo.full_name}")
```

You'll notice slight pauses every 30 items - that's when it fetches the next page synchronously.

### Why This Matters

#### Synchronous (PyGithub) Advantages ✅
- **Simpler code** - No async/await complexity
- **Easier to debug** - Sequential execution
- **Good for scripts** - Most use cases don't need async

#### Synchronous Drawbacks ❌
- **Slower for multiple repos** - Must wait for each request
- **Blocks the thread** - Can't do other work during requests

#### Asynchronous Advantages ✅
- **Much faster** - Can fetch multiple repos in parallel
- **Better for large datasets** - 100s of starred repos
- **Non-blocking** - Can do other work while waiting

#### Asynchronous Drawbacks ❌
- **More complex** - async/await syntax
- **Harder to debug** - Concurrent execution

### If You Need Async with PyGithub

PyGithub doesn't support async natively, but you can:

#### Option 1: Use aiohttp directly

```python
import aiohttp
import asyncio

# Fully async - much faster for multiple requests
async def fetch_starred():
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {token}"}
        async with session.get("https://api.github.com/user/starred", headers=headers) as resp:
            return await resp.json()

starred = asyncio.run(fetch_starred())
```

#### Option 2: Run PyGithub in thread pool

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from github import Github, Auth
import os

def get_starred_sync():
    """Synchronous PyGithub function"""
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    with Github(auth=auth) as g:
        user = g.get_user()
        return list(user.get_starred())

async def get_starred_async():
    """Run synchronous code in thread pool"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        starred = await loop.run_in_executor(pool, get_starred_sync)
        return starred

# Use it
starred = asyncio.run(get_starred_async())
```

#### Option 3: Use `gidgethub` (async GitHub library)

```python
# Alternative library with native async support
import gidgethub.aiohttp
import aiohttp

async def main():
    async with aiohttp.ClientSession() as session:
        gh = gidgethub.aiohttp.GitHubAPI(session, "username")
        # ... async GitHub operations
```

### Performance Comparison

| Aspect | PyGithub | aiohttp (raw API) |
|--------|----------|-------------------|
| Execution | Synchronous | Asynchronous |
| HTTP Requests | Blocking | Non-blocking |
| Pagination | Lazy (on-demand) | Manual control |
| Speed (single repo) | Fast | Fast |
| Speed (100 repos) | Slow (sequential) | Very fast (parallel) |
| Code Complexity | Simple | More complex |
| Best For | Scripts, simple tasks | Large datasets, web apps |

### Recommendation

For most use cases, PyGithub's synchronous approach is perfectly fine. If you're fetching hundreds of repos or need high performance, consider using the aiohttp examples for truly asynchronous operations.
