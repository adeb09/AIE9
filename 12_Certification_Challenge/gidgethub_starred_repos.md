# Fetching Starred Repositories with gidgethub

A guide for using the gidgethub async library to fetch starred repositories from GitHub.

## Installation

```bash
pip install gidgethub[aiohttp]
```

This installs gidgethub with aiohttp support. Requires Python 3.8+.

## Key Concepts

- **`getitem()`** - For GET calls that return a single item
- **`getiter()`** - Returns an async iterable that automatically handles pagination
- **Sans-I/O design** - gidgethub performs no I/O itself, you provide the HTTP library (aiohttp)

## Basic Setup with Authentication

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def main():
    # Get OAuth token from environment
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        # Create GitHub API instance
        gh = GitHubAPI(session, "your-username",
                      oauth_token=oauth_token)

        # Your API calls here
        print(f"Rate limit remaining: {gh.rate_limit.remaining}")

asyncio.run(main())
```

## Fetch Your Own Starred Repositories

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def get_my_starred_repos():
    """Fetch all starred repositories for authenticated user"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Use getiter() for automatic pagination
        starred_repos = []

        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            print(f"⭐ {repo['full_name']}")
            print(f"   Description: {repo['description']}")
            print(f"   Stars: {repo['stargazers_count']}")
            print(f"   Language: {repo['language']}")
            print()

        print(f"\nTotal starred repositories: {len(starred_repos)}")
        return starred_repos

if __name__ == "__main__":
    asyncio.run(get_my_starred_repos())
```

## Fetch Another User's Starred Repositories

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def get_user_starred_repos(username: str):
    """Fetch starred repositories for a specific user"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Fetch specific user's starred repos
        endpoint = f"/users/{username}/starred"

        starred_count = 0
        async for repo in gh.getiter(endpoint):
            starred_count += 1
            print(f"{starred_count}. {repo['full_name']} ({repo['stargazers_count']} ⭐)")

        print(f"\n{username} has starred {starred_count} repositories")

if __name__ == "__main__":
    asyncio.run(get_user_starred_repos("torvalds"))
```

## Fetch Starred Repos with Timestamps

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def get_starred_with_timestamps():
    """Fetch starred repositories with star timestamps"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Use special accept header for timestamps
        async for starred_info in gh.getiter(
            "/user/starred",
            accept="application/vnd.github.star+json"
        ):
            repo = starred_info['repo']
            starred_at = starred_info['starred_at']

            print(f"Repo: {repo['full_name']}")
            print(f"Starred at: {starred_at}")
            print(f"Description: {repo['description']}")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(get_starred_with_timestamps())
```

## Fetch Single Repository Details with getitem()

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def get_repo_details(owner: str, repo: str):
    """Fetch details of a single repository using getitem()"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Use getitem() for single item
        repo_data = await gh.getitem(f"/repos/{owner}/{repo}")

        print(f"Name: {repo_data['full_name']}")
        print(f"Description: {repo_data['description']}")
        print(f"Stars: {repo_data['stargazers_count']}")
        print(f"Forks: {repo_data['forks_count']}")
        print(f"Language: {repo_data['language']}")
        print(f"Open Issues: {repo_data['open_issues_count']}")
        print(f"Created: {repo_data['created_at']}")
        print(f"License: {repo_data.get('license', {}).get('name', 'None')}")

        return repo_data

if __name__ == "__main__":
    asyncio.run(get_repo_details("gidgethub", "gidgethub"))
```

## Advanced: Filter Starred Repos by Language

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI
from collections import Counter

async def filter_starred_by_language(target_language: str = None):
    """Filter starred repositories by programming language"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        repos_by_language = []
        all_languages = []

        async for repo in gh.getiter("/user/starred"):
            language = repo.get('language')

            if language:
                all_languages.append(language)

                if target_language and language == target_language:
                    repos_by_language.append(repo)

        if target_language:
            print(f"Found {len(repos_by_language)} {target_language} repositories:\n")
            for repo in repos_by_language:
                print(f"- {repo['full_name']}")
        else:
            # Show language distribution
            language_counts = Counter(all_languages)
            print("Starred repositories by language:\n")
            for lang, count in language_counts.most_common():
                print(f"{lang}: {count} repos")

if __name__ == "__main__":
    # Show all languages
    asyncio.run(filter_starred_by_language())

    # Or filter by specific language
    # asyncio.run(filter_starred_by_language("Python"))
```

## Complete Example: Analyze Multiple Users Concurrently

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def analyze_user_starred(gh: GitHubAPI, username: str):
    """Analyze a single user's starred repositories"""
    try:
        starred_count = 0
        languages = []

        async for repo in gh.getiter(f"/users/{username}/starred"):
            starred_count += 1
            if repo.get('language'):
                languages.append(repo['language'])

        top_language = max(set(languages), key=languages.count) if languages else "None"

        return {
            'username': username,
            'total_starred': starred_count,
            'top_language': top_language
        }
    except Exception as e:
        return {
            'username': username,
            'error': str(e)
        }

async def analyze_multiple_users(usernames: list):
    """Analyze starred repos for multiple users concurrently"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Run all analyses concurrently
        tasks = [analyze_user_starred(gh, username) for username in usernames]
        results = await asyncio.gather(*tasks)

        print("Analysis Results:\n")
        for result in results:
            if 'error' in result:
                print(f"❌ {result['username']}: {result['error']}")
            else:
                print(f"✓ {result['username']}:")
                print(f"  Total starred: {result['total_starred']}")
                print(f"  Top language: {result['top_language']}")
                print()

        print(f"Rate limit remaining: {gh.rate_limit.remaining}")

if __name__ == "__main__":
    users = ["torvalds", "guido", "gvanrossum"]
    asyncio.run(analyze_multiple_users(users))
```

## Error Handling

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException

async def safe_fetch_starred(username: str):
    """Safely fetch starred repos with error handling"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    try:
        async with aiohttp.ClientSession() as session:
            gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

            endpoint = f"/users/{username}/starred"

            starred_repos = []
            async for repo in gh.getiter(endpoint):
                starred_repos.append(repo['full_name'])

            print(f"✓ Successfully fetched {len(starred_repos)} repos for {username}")
            return starred_repos

    except GitHubException as e:
        if e.status_code == 404:
            print(f"❌ User '{username}' not found")
        elif e.status_code == 401:
            print("❌ Authentication failed - check your token")
        elif e.status_code == 403:
            print("❌ Rate limit exceeded or access forbidden")
        else:
            print(f"❌ GitHub API error: {e}")
        return []

    except aiohttp.ClientError as e:
        print(f"❌ Network error: {e}")
        return []

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

if __name__ == "__main__":
    asyncio.run(safe_fetch_starred("torvalds"))
```

## Checking Rate Limits

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def check_rate_limit():
    """Check GitHub API rate limit status"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Make a simple request to initialize rate limit
        await gh.getitem("/user")

        print("Rate Limit Information:")
        print(f"  Limit: {gh.rate_limit.limit}")
        print(f"  Remaining: {gh.rate_limit.remaining}")
        print(f"  Reset time: {gh.rate_limit.reset_datetime}")

if __name__ == "__main__":
    asyncio.run(check_rate_limit())
```

## Pagination with Parameters

```python
import asyncio
import aiohttp
import os
from gidgethub.aiohttp import GitHubAPI

async def fetch_starred_with_params():
    """Fetch starred repos with query parameters"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # You can add query parameters to the endpoint
        # getiter() automatically handles pagination
        endpoint = "/user/starred"

        # Note: per_page is handled automatically by getiter()
        # It will fetch all pages regardless of per_page setting

        count = 0
        async for repo in gh.getiter(endpoint):
            count += 1
            if count <= 10:  # Show first 10
                print(f"{count}. {repo['full_name']}")

        print(f"\nTotal: {count} repositories")

if __name__ == "__main__":
    asyncio.run(fetch_starred_with_params())
```

## Key Differences from PyGithub

| Feature | gidgethub | PyGithub |
|---------|-----------|----------|
| Execution | Async (non-blocking) | Sync (blocking) |
| API Access | Direct endpoints | Abstracted methods |
| Pagination | `getiter()` auto-handles | `PaginatedList` lazy |
| HTTP Library | Bring your own (aiohttp) | Built-in |
| Performance | Very fast (concurrent) | Slower (sequential) |
| Complexity | More complex | Simpler |
| Best For | High performance, web apps | Scripts, simple tasks |

## Common Patterns

### Pattern 1: Collect all items into a list

```python
starred_repos = []
async for repo in gh.getiter("/user/starred"):
    starred_repos.append(repo)
```

### Pattern 2: Process items as they arrive

```python
async for repo in gh.getiter("/user/starred"):
    await process_repo(repo)  # Process immediately
```

### Pattern 3: Early termination

```python
count = 0
async for repo in gh.getiter("/user/starred"):
    count += 1
    if count >= 10:
        break  # Stop after 10 repos
```

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

## Fetching README Files

### Important: Not All Repos Have READMEs!

The `/repos/{owner}/{repo}/readme` endpoint is a **standard GitHub API endpoint**, but:

- ❌ Returns `404 Not Found` if no README exists
- ✅ Returns README data if present
- 🔍 GitHub searches for: README.md, README.rst, README.txt, README, etc. (case-insensitive)

**Statistics:** In practice, **70-90% of repositories have a README**, but many don't. This is why proper error handling is critical!

### IMPORTANT: `/readme` Only Returns README Files!

The `/readme` endpoint **ONLY** returns files named "README" (with various extensions). Other markdown files in the root directory are **NOT** included:

#### What `/readme` Returns:
- ✅ `README.md`
- ✅ `README.rst`
- ✅ `README.txt`
- ✅ `README`
- ✅ `Readme.md` (case-insensitive)

#### What `/readme` Does NOT Return:
- ❌ `CONTRIBUTING.md`
- ❌ `CHANGELOG.md`
- ❌ `LICENSE.md`
- ❌ `INSTALL.md`
- ❌ `custom-name.md`
- ❌ Any other markdown file that isn't named README

Even though these files are displayed on the GitHub page, you need to fetch them separately using the **contents API**: `/repos/{owner}/{repo}/contents/`

See `fetch_all_markdown_files.py` for examples of how to fetch ALL markdown files from a repository.

### Get Single Repository README

```python
import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI

async def get_repo_readme(owner: str, repo: str):
    """Fetch the README file for a repository"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        try:
            # Get README from GitHub API
            readme_data = await gh.getitem(f"/repos/{owner}/{repo}/readme")

            # Decode content (it's base64 encoded)
            content = base64.b64decode(readme_data['content']).decode('utf-8')

            print(f"README for {owner}/{repo}:")
            print(f"Name: {readme_data['name']}")
            print(f"Size: {readme_data['size']} bytes")
            print(f"Path: {readme_data['path']}")
            print(f"\nContent:\n{content[:500]}...")  # Show first 500 chars

            return content

        except Exception as e:
            print(f"No README found for {owner}/{repo}: {e}")
            return None

if __name__ == "__main__":
    asyncio.run(get_repo_readme("gidgethub", "gidgethub"))
```

### Fetch READMEs for All Starred Repos (Concurrent!)

This is where gidgethub really shines - fetching READMEs for all your starred repos **concurrently**:

```python
import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException

async def fetch_readme(gh: GitHubAPI, owner: str, repo: str):
    """Fetch README for a single repository"""
    try:
        readme_data = await gh.getitem(f"/repos/{owner}/{repo}/readme")
        content = base64.b64decode(readme_data['content']).decode('utf-8')

        return {
            'repo': f"{owner}/{repo}",
            'readme_name': readme_data['name'],
            'size': readme_data['size'],
            'content': content,
            'success': True
        }
    except GitHubException as e:
        return {
            'repo': f"{owner}/{repo}",
            'success': False,
            'error': str(e)
        }

async def fetch_all_starred_readmes():
    """
    Fetch READMEs for ALL starred repositories concurrently.
    This is MUCH faster than doing it sequentially!
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        print("Fetching starred repositories...")

        # Step 1: Get all starred repos
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)

        print(f"Found {len(starred_repos)} starred repositories")
        print(f"Fetching READMEs concurrently...\n")

        # Step 2: Fetch all READMEs concurrently (THE MAGIC!)
        tasks = [
            fetch_readme(gh, repo['owner']['login'], repo['name'])
            for repo in starred_repos
        ]

        # Wait for all README fetches to complete
        results = await asyncio.gather(*tasks)

        # Step 3: Process results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"✓ Successfully fetched {len(successful)} READMEs")
        print(f"✗ Failed to fetch {len(failed)} READMEs\n")

        # Show sample results
        print("Sample READMEs:\n")
        for result in successful[:5]:  # Show first 5
            print(f"Repository: {result['repo']}")
            print(f"README: {result['readme_name']} ({result['size']} bytes)")
            print(f"Preview: {result['content'][:100]}...")
            print("-" * 60)

        print(f"\nRate limit remaining: {gh.rate_limit.remaining}")

        return results

if __name__ == "__main__":
    asyncio.run(fetch_all_starred_readmes())
```

### Fetch READMEs with Progress Tracking

```python
import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException

async def fetch_readme_with_progress(gh: GitHubAPI, owner: str, repo: str, index: int, total: int):
    """Fetch README with progress reporting"""
    try:
        readme_data = await gh.getitem(f"/repos/{owner}/{repo}/readme")
        content = base64.b64decode(readme_data['content']).decode('utf-8')

        print(f"[{index}/{total}] ✓ {owner}/{repo}")

        return {
            'repo': f"{owner}/{repo}",
            'readme_name': readme_data['name'],
            'content': content,
            'success': True
        }
    except GitHubException:
        print(f"[{index}/{total}] ✗ {owner}/{repo} - No README")
        return {
            'repo': f"{owner}/{repo}",
            'success': False
        }

async def fetch_starred_readmes_with_progress(max_repos: int = None):
    """Fetch READMEs with progress tracking"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Get starred repos
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            if max_repos and len(starred_repos) >= max_repos:
                break

        total = len(starred_repos)
        print(f"Fetching READMEs for {total} repositories...\n")

        # Fetch all READMEs concurrently with progress
        tasks = [
            fetch_readme_with_progress(
                gh,
                repo['owner']['login'],
                repo['name'],
                idx + 1,
                total
            )
            for idx, repo in enumerate(starred_repos)
        ]

        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r['success']]
        print(f"\n✓ Total successful: {len(successful)}/{total}")
        print(f"Rate limit remaining: {gh.rate_limit.remaining}")

        return results

if __name__ == "__main__":
    # Fetch first 10 starred repos' READMEs
    asyncio.run(fetch_starred_readmes_with_progress(max_repos=10))
```

### Complete Example: Starred Repos with Full Details

```python
import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException
from typing import Dict, Any

async def fetch_repo_complete_info(gh: GitHubAPI, repo_data: Dict[Any, Any]):
    """
    Fetch complete information for a repository including README.
    Takes basic repo data and enriches it with README content.
    """
    owner = repo_data['owner']['login']
    name = repo_data['name']

    try:
        # Fetch README
        readme_data = await gh.getitem(f"/repos/{owner}/{name}/readme")
        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')

        return {
            'full_name': repo_data['full_name'],
            'description': repo_data['description'],
            'stars': repo_data['stargazers_count'],
            'language': repo_data['language'],
            'url': repo_data['html_url'],
            'readme_name': readme_data['name'],
            'readme_size': readme_data['size'],
            'readme_content': readme_content,
            'has_readme': True
        }
    except GitHubException:
        return {
            'full_name': repo_data['full_name'],
            'description': repo_data['description'],
            'stars': repo_data['stargazers_count'],
            'language': repo_data['language'],
            'url': repo_data['html_url'],
            'has_readme': False
        }

async def get_starred_repos_complete():
    """
    Get all starred repos with complete information including READMEs.
    This demonstrates the full power of async concurrent requests!
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        print("Step 1: Fetching starred repositories...")
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)

        print(f"Found {len(starred_repos)} starred repositories")
        print(f"\nStep 2: Fetching READMEs concurrently...")

        # Fetch all additional info concurrently
        tasks = [fetch_repo_complete_info(gh, repo) for repo in starred_repos]
        complete_info = await asyncio.gather(*tasks)

        # Analyze results
        with_readme = [r for r in complete_info if r['has_readme']]
        without_readme = [r for r in complete_info if not r['has_readme']]

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total repositories: {len(complete_info)}")
        print(f"With README: {len(with_readme)} ({len(with_readme)/len(complete_info)*100:.1f}%)")
        print(f"Without README: {len(without_readme)} ({len(without_readme)/len(complete_info)*100:.1f}%)")
        print(f"\nRate limit remaining: {gh.rate_limit.remaining}")

        # Show some examples
        print(f"\n{'='*60}")
        print("Sample repositories with READMEs:")
        print(f"{'='*60}\n")

        for repo in with_readme[:3]:
            print(f"Repository: {repo['full_name']}")
            print(f"Description: {repo['description']}")
            print(f"Stars: {repo['stars']} ⭐")
            print(f"Language: {repo['language']}")
            print(f"README: {repo['readme_name']} ({repo['readme_size']} bytes)")
            print(f"README Preview: {repo['readme_content'][:150]}...")
            print(f"URL: {repo['url']}")
            print("-" * 60)

        return complete_info

if __name__ == "__main__":
    results = asyncio.run(get_starred_repos_complete())
```

### Performance Comparison: gidgethub vs PyGithub

Here's a timing comparison to show the performance difference:

```python
import asyncio
import aiohttp
import os
import base64
import time
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException

async def timed_fetch_readmes_gidgethub(max_repos: int = 20):
    """Time how long it takes to fetch READMEs with gidgethub (concurrent)"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        # Get starred repos
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            if len(starred_repos) >= max_repos:
                break

        # Fetch READMEs concurrently
        async def fetch_readme(repo):
            try:
                readme = await gh.getitem(f"/repos/{repo['owner']['login']}/{repo['name']}/readme")
                return base64.b64decode(readme['content']).decode('utf-8')
            except:
                return None

        tasks = [fetch_readme(repo) for repo in starred_repos]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]

    elapsed = time.time() - start_time

    print(f"{'='*60}")
    print(f"gidgethub (Async/Concurrent)")
    print(f"{'='*60}")
    print(f"Repos processed: {len(starred_repos)}")
    print(f"READMEs fetched: {len(successful)}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Average per repo: {elapsed/len(starred_repos):.2f} seconds")
    print()

    return elapsed

def timed_fetch_readmes_pygithub(max_repos: int = 20):
    """Time how long it takes to fetch READMEs with PyGithub (sequential)"""
    from github import Github, Auth

    oauth_token = os.getenv("GITHUB_TOKEN")
    auth = Auth.Token(oauth_token)

    start_time = time.time()

    with Github(auth=auth) as g:
        user = g.get_user()
        starred = user.get_starred()

        successful = 0
        for i, repo in enumerate(starred):
            if i >= max_repos:
                break

            try:
                readme = repo.get_readme()
                content = readme.decoded_content.decode('utf-8')
                successful += 1
            except:
                pass

    elapsed = time.time() - start_time

    print(f"{'='*60}")
    print(f"PyGithub (Sync/Sequential)")
    print(f"{'='*60}")
    print(f"Repos processed: {max_repos}")
    print(f"READMEs fetched: {successful}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Average per repo: {elapsed/max_repos:.2f} seconds")
    print()

    return elapsed

async def compare_performance(max_repos: int = 20):
    """Compare performance between gidgethub and PyGithub"""
    print(f"\nPerformance Comparison: Fetching {max_repos} READMEs\n")

    # Test gidgethub (async)
    gidget_time = await timed_fetch_readmes_gidgethub(max_repos)

    # Test PyGithub (sync)
    pygithub_time = timed_fetch_readmes_pygithub(max_repos)

    # Show comparison
    print(f"{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    speedup = pygithub_time / gidget_time
    print(f"gidgethub is {speedup:.1f}x FASTER than PyGithub!")
    print(f"Time saved: {pygithub_time - gidget_time:.2f} seconds")
    print()

if __name__ == "__main__":
    asyncio.run(compare_performance(max_repos=20))
```

### Use Case: Export Starred Repos with READMEs to JSON

```python
import asyncio
import aiohttp
import os
import base64
import json
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException

async def export_starred_to_json(output_file: str = "starred_repos.json"):
    """Export all starred repos with READMEs to JSON file"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "my-script", oauth_token=oauth_token)

        print("Fetching starred repositories...")
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)

        print(f"Fetching READMEs for {len(starred_repos)} repos...")

        async def get_repo_with_readme(repo):
            result = {
                'full_name': repo['full_name'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'language': repo['language'],
                'url': repo['html_url'],
                'topics': repo.get('topics', []),
            }

            try:
                readme = await gh.getitem(f"/repos/{repo['owner']['login']}/{repo['name']}/readme")
                result['readme'] = base64.b64decode(readme['content']).decode('utf-8')
            except:
                result['readme'] = None

            return result

        tasks = [get_repo_with_readme(repo) for repo in starred_repos]
        results = await asyncio.gather(*tasks)

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported to {output_file}")
        print(f"Rate limit remaining: {gh.rate_limit.remaining}")

if __name__ == "__main__":
    asyncio.run(export_starred_to_json())
```

### Best Practices for Handling Missing READMEs

Since not all repositories have READMEs, here are recommended patterns:

#### Pattern 1: Return None for Missing READMEs

```python
async def safe_fetch_readme(gh: GitHubAPI, owner: str, repo: str):
    """Fetch README, return None if not found"""
    try:
        readme = await gh.getitem(f"/repos/{owner}/{repo}/readme")
        return base64.b64decode(readme['content']).decode('utf-8')
    except GitHubException as e:
        if e.status_code == 404:
            return None  # No README found - this is OK
        raise  # Re-raise other errors
```

#### Pattern 2: Return Status Object

```python
async def fetch_readme_with_status(gh: GitHubAPI, owner: str, repo: str):
    """Fetch README with detailed status"""
    try:
        readme = await gh.getitem(f"/repos/{owner}/{repo}/readme")
        return {
            'has_readme': True,
            'name': readme['name'],
            'size': readme['size'],
            'content': base64.b64decode(readme['content']).decode('utf-8')
        }
    except GitHubException as e:
        return {
            'has_readme': False,
            'error_code': e.status_code,
            'error_message': str(e)
        }
```

#### Pattern 3: Use return_exceptions=True

```python
async def fetch_all_readmes_safe(starred_repos):
    """Fetch READMEs, handling exceptions gracefully"""
    tasks = [
        gh.getitem(f"/repos/{repo['owner']['login']}/{repo['name']}/readme")
        for repo in starred_repos
    ]

    # return_exceptions=True means exceptions won't crash the gather
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"Successful: {len(successful)}, Failed: {len(failed)}")
    return successful
```

#### Pattern 4: Fallback to Description

```python
async def get_repo_summary(gh: GitHubAPI, repo):
    """Get README or fallback to description"""
    try:
        readme = await gh.getitem(f"/repos/{repo['owner']['login']}/{repo['name']}/readme")
        content = base64.b64decode(readme['content']).decode('utf-8')
        return {
            'repo': repo['full_name'],
            'summary_source': 'readme',
            'summary': content[:500]  # First 500 chars
        }
    except GitHubException:
        # Fallback to description if no README
        return {
            'repo': repo['full_name'],
            'summary_source': 'description',
            'summary': repo['description'] or 'No description available'
        }
```

#### Testing Which Repos Have READMEs

You can run the test script to see which of your starred repos have READMEs:

```bash
python test_readme_existence.py
```

This will show you:
- Which repos have READMEs (✓)
- Which repos don't have READMEs (✗)
- Statistics (percentage with/without)
- README file names and sizes

## Summary

- Use **`getiter()`** for paginated results (like starred repos) - it automatically handles all pages
- Use **`getitem()`** for single items (like a specific repo or README)
- gidgethub is fully async - perfect for fetching data from multiple users concurrently
- Direct API endpoint access means you follow GitHub's documentation exactly
- **Much faster than PyGithub** when fetching additional data (like READMEs) for multiple repos - can be 10-20x faster!
- README content is base64 encoded - use `base64.b64decode()` to decode it
- Use `asyncio.gather()` to fetch READMEs for all repos concurrently

## Resources

- [gidgethub Documentation](https://gidgethub.readthedocs.io/en/latest/)
- [gidgethub GitHub Repository](https://github.com/gidgethub/gidgethub)
- [gidgethub.abc API Reference](https://gidgethub.readthedocs.io/en/latest/abc.html)
- [GitHub REST API - Starring Endpoints](https://docs.github.com/en/rest/activity/starring)
