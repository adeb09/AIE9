# aiohttp GET Request Scaffolding

A collection of Python scaffolding templates for making GET requests using the aiohttp library.

## Installation

```bash
pip install aiohttp
```

## Basic GET Request

```python
import aiohttp
import asyncio

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # Check status
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"Error: {response.status}")
                return None

# Run the async function
async def main():
    url = "https://api.github.com/user/starred"
    data = await fetch_data(url)
    print(data)

# Execute
asyncio.run(main())
```

## With Authentication Headers (GitHub API)

```python
import aiohttp
import asyncio
import os

async def fetch_github_data(url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error {response.status}: {error_text}")

async def main():
    token = os.getenv("GITHUB_TOKEN")
    url = "https://api.github.com/user/starred"

    try:
        data = await fetch_github_data(url, token)
        print(f"Retrieved {len(data)} starred repos")
        for repo in data:
            print(f"- {repo['full_name']}: {repo['description']}")
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(main())
```

## With Error Handling and Retry Logic

```python
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3
) -> Optional[Dict[Any, Any]]:
    """Fetch URL with retry logic"""

    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise Exception("Authentication failed - check your token")
                elif response.status == 403:
                    raise Exception("Rate limit exceeded or forbidden")
                elif response.status == 404:
                    raise Exception("Resource not found")
                else:
                    print(f"Attempt {attempt + 1} failed with status {response.status}")

        except aiohttp.ClientError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return None

async def main():
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        url = "https://api.github.com/user/starred?per_page=100"
        data = await fetch_with_retry(session, url)

        if data:
            print(f"Success! Retrieved {len(data)} repos")
        else:
            print("Failed to retrieve data")

asyncio.run(main())
```

## Fetching Multiple URLs Concurrently

```python
import aiohttp
import asyncio
import os

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch a single URL"""
    async with session.get(url) as response:
        return await response.json()

async def fetch_all(urls: list[str], token: str):
    """Fetch multiple URLs concurrently"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def main():
    token = os.getenv("GITHUB_TOKEN")

    # Fetch multiple pages concurrently
    urls = [
        "https://api.github.com/user/starred?per_page=100&page=1",
        "https://api.github.com/user/starred?per_page=100&page=2",
        "https://api.github.com/user/starred?per_page=100&page=3",
    ]

    results = await fetch_all(urls, token)

    total_repos = sum(len(page) for page in results if isinstance(page, list))
    print(f"Total repos fetched: {total_repos}")

asyncio.run(main())
```

## Getting Repository by ID

```python
import aiohttp
import asyncio
import os

async def get_repo_by_id(repo_id: int, token: str):
    """Get repository information by numeric ID"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    url = f"https://api.github.com/repositories/{repo_id}"

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error {response.status}: {await response.text()}")
                return None

async def main():
    token = os.getenv("GITHUB_TOKEN")
    repo_id = 123456789  # Example repo ID

    repo_data = await get_repo_by_id(repo_id, token)

    if repo_data:
        print(f"Name: {repo_data['full_name']}")
        print(f"Description: {repo_data['description']}")
        print(f"Stars: {repo_data['stargazers_count']}")

asyncio.run(main())
```

## Pagination Helper

```python
import aiohttp
import asyncio
import os
from typing import List, Dict, Any

async def fetch_all_pages(base_url: str, token: str, per_page: int = 100) -> List[Dict[Any, Any]]:
    """Fetch all pages from a paginated GitHub API endpoint"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    all_results = []
    page = 1

    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            url = f"{base_url}?per_page={per_page}&page={page}"

            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Error on page {page}: {response.status}")
                    break

                data = await response.json()

                # If empty, we've reached the end
                if not data:
                    break

                all_results.extend(data)
                print(f"Fetched page {page}: {len(data)} items")

                page += 1

    return all_results

async def main():
    token = os.getenv("GITHUB_TOKEN")
    base_url = "https://api.github.com/user/starred"

    all_repos = await fetch_all_pages(base_url, token)
    print(f"\nTotal starred repositories: {len(all_repos)}")

    # Process repos
    for repo in all_repos[:5]:  # Show first 5
        print(f"- {repo['full_name']}")

asyncio.run(main())
```

## Complete Example: Fetch Starred Repos and Get Details by ID

```python
import aiohttp
import asyncio
import os
from typing import List, Dict, Any

async def fetch_starred_repos(token: str, per_page: int = 100) -> List[Dict[Any, Any]]:
    """Fetch all starred repositories"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    all_repos = []
    page = 1

    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            url = f"https://api.github.com/user/starred?per_page={per_page}&page={page}"

            async with session.get(url) as response:
                if response.status != 200:
                    break

                data = await response.json()
                if not data:
                    break

                all_repos.extend(data)
                page += 1

    return all_repos

async def get_repo_details_by_id(repo_id: int, token: str) -> Dict[Any, Any]:
    """Get detailed repository information by ID"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    url = f"https://api.github.com/repositories/{repo_id}"

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return {}

async def main():
    token = os.getenv("GITHUB_TOKEN")

    # Fetch all starred repos
    print("Fetching starred repositories...")
    starred_repos = await fetch_starred_repos(token)
    print(f"Found {len(starred_repos)} starred repositories\n")

    # Get details for first 3 repos by ID
    if starred_repos:
        print("Fetching details by repo ID:")
        for repo in starred_repos[:3]:
            repo_id = repo['id']
            details = await get_repo_details_by_id(repo_id, token)

            if details:
                print(f"\nRepo ID: {repo_id}")
                print(f"  Name: {details['full_name']}")
                print(f"  Description: {details.get('description', 'No description')}")
                print(f"  Stars: {details['stargazers_count']}")
                print(f"  Language: {details.get('language', 'N/A')}")

asyncio.run(main())
```

## Key Points

1. **Always use async/await** - aiohttp is asynchronous
2. **Use ClientSession as context manager** - Ensures proper cleanup
3. **Check response.status** - Handle different HTTP status codes
4. **Use headers for authentication** - Pass GitHub token in Authorization header
5. **Handle errors gracefully** - Use try/except and retry logic
6. **Concurrent requests** - Use `asyncio.gather()` for multiple URLs
7. **Pagination** - Loop through pages until empty response

## Common Status Codes

- `200` - Success
- `401` - Authentication failed (bad token)
- `403` - Rate limit exceeded or forbidden
- `404` - Resource not found
- `422` - Validation failed

## Environment Setup

Set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN="your_token_here"
```

Or use a `.env` file with `python-dotenv`:

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
```
