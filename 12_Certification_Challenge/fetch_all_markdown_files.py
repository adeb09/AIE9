"""
Fetch all markdown files from the root directory of repositories.
This goes beyond just README files to capture all .md files shown on GitHub.
"""

import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException
from typing import List, Dict, Any


async def get_root_markdown_files(gh: GitHubAPI, owner: str, repo: str) -> List[Dict[Any, Any]]:
    """
    Get all markdown files from the root directory of a repository.
    This includes README.md, CONTRIBUTING.md, CHANGELOG.md, etc.
    """
    try:
        # Get contents of root directory
        contents = await gh.getitem(f"/repos/{owner}/{repo}/contents/")

        # Filter for markdown files (case-insensitive)
        markdown_files = [
            file for file in contents
            if file['type'] == 'file' and file['name'].lower().endswith('.md')
        ]

        return markdown_files
    except GitHubException as e:
        print(f"Error fetching contents for {owner}/{repo}: {e}")
        return []


async def fetch_markdown_content(gh: GitHubAPI, owner: str, repo: str, file_path: str):
    """Fetch content of a specific markdown file"""
    try:
        file_data = await gh.getitem(f"/repos/{owner}/{repo}/contents/{file_path}")
        content = base64.b64decode(file_data['content']).decode('utf-8')

        return {
            'name': file_data['name'],
            'path': file_data['path'],
            'size': file_data['size'],
            'content': content,
            'success': True
        }
    except GitHubException as e:
        return {
            'path': file_path,
            'success': False,
            'error': str(e)
        }


async def get_all_markdown_from_repo(owner: str, repo: str):
    """
    Get ALL markdown files from a repository's root directory.
    Not just README, but also CONTRIBUTING.md, CHANGELOG.md, etc.
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "markdown-fetcher", oauth_token=oauth_token)

        print(f"Fetching markdown files from {owner}/{repo}...\n")

        # Step 1: Get list of markdown files in root
        markdown_files = await get_root_markdown_files(gh, owner, repo)

        if not markdown_files:
            print("No markdown files found in root directory")
            return []

        print(f"Found {len(markdown_files)} markdown file(s) in root directory:")
        for file in markdown_files:
            print(f"  - {file['name']} ({file['size']} bytes)")

        # Step 2: Fetch content of all markdown files concurrently
        print(f"\nFetching content of all markdown files...\n")

        tasks = [
            fetch_markdown_content(gh, owner, repo, file['path'])
            for file in markdown_files
        ]
        results = await asyncio.gather(*tasks)

        # Display results
        successful = [r for r in results if r['success']]

        for result in successful:
            print(f"{'='*70}")
            print(f"File: {result['name']} ({result['size']} bytes)")
            print(f"{'='*70}")
            print(result['content'][:300] + "...")  # Show first 300 chars
            print()

        return results


async def compare_readme_vs_all_markdown(owner: str, repo: str):
    """
    Compare what /readme returns vs all markdown files in root.
    This demonstrates the difference!
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "markdown-fetcher", oauth_token=oauth_token)

        print(f"Comparing /readme endpoint vs all markdown files for {owner}/{repo}\n")

        # Method 1: Using /readme endpoint
        print("="*70)
        print("METHOD 1: Using /readme endpoint")
        print("="*70)
        try:
            readme = await gh.getitem(f"/repos/{owner}/{repo}/readme")
            print(f"✓ Found: {readme['name']}")
            print(f"  Size: {readme['size']} bytes")
            print(f"  This is what the /readme endpoint returns")
        except GitHubException:
            print("✗ No README found")

        print()

        # Method 2: List all markdown files
        print("="*70)
        print("METHOD 2: All markdown files in root directory")
        print("="*70)
        markdown_files = await get_root_markdown_files(gh, owner, repo)

        if markdown_files:
            print(f"✓ Found {len(markdown_files)} markdown file(s):")
            for file in markdown_files:
                is_readme = file['name'].lower().startswith('readme')
                marker = "← This is what /readme returns" if is_readme else ""
                print(f"  - {file['name']} ({file['size']} bytes) {marker}")
        else:
            print("✗ No markdown files found")

        print()
        print("="*70)
        print("CONCLUSION")
        print("="*70)
        print("The /readme endpoint only returns the README file.")
        print("Other .md files (CONTRIBUTING, CHANGELOG, etc.) require separate API calls.")


async def fetch_all_markdown_from_starred_repos(max_repos: int = 10):
    """
    Fetch ALL markdown files from your starred repositories.
    This gets README.md, CONTRIBUTING.md, CHANGELOG.md, and more!
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "markdown-fetcher", oauth_token=oauth_token)

        print(f"Fetching markdown files from first {max_repos} starred repos...\n")

        # Get starred repos
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            if len(starred_repos) >= max_repos:
                break

        # For each repo, get all markdown files
        async def process_repo(repo):
            owner = repo['owner']['login']
            name = repo['name']

            # Get list of markdown files
            markdown_files = await get_root_markdown_files(gh, owner, name)

            # Fetch content of all markdown files concurrently
            if markdown_files:
                tasks = [
                    fetch_markdown_content(gh, owner, name, file['path'])
                    for file in markdown_files
                ]
                contents = await asyncio.gather(*tasks)

                return {
                    'repo': repo['full_name'],
                    'markdown_files': [c for c in contents if c['success']],
                    'file_count': len(markdown_files)
                }
            else:
                return {
                    'repo': repo['full_name'],
                    'markdown_files': [],
                    'file_count': 0
                }

        # Process all repos concurrently
        results = await asyncio.gather(*[process_repo(repo) for repo in starred_repos])

        # Display summary
        print(f"\n{'='*70}")
        print("MARKDOWN FILES SUMMARY")
        print(f"{'='*70}\n")

        for result in results:
            print(f"Repository: {result['repo']}")
            if result['file_count'] > 0:
                print(f"  Markdown files ({result['file_count']}):")
                for md_file in result['markdown_files']:
                    print(f"    - {md_file['name']} ({md_file['size']} bytes)")
            else:
                print("  No markdown files in root directory")
            print()

        total_files = sum(r['file_count'] for r in results)
        repos_with_markdown = sum(1 for r in results if r['file_count'] > 0)

        print(f"{'='*70}")
        print(f"Total markdown files found: {total_files}")
        print(f"Repos with markdown files: {repos_with_markdown}/{len(results)}")
        print(f"Rate limit remaining: {gh.rate_limit.remaining}")

        return results


async def fetch_starred_repos_with_docs(max_repos: int = None) -> List[Dict[Any, Any]]:
    """
    Fetch all starred repositories for the authenticated GitHub user, then
    concurrently retrieve documentation for each repo using the following strategy:

      1. Try the /readme endpoint first (covers README.md, README.rst, etc.)
      2. If no README exists (404), fall back to fetching ALL markdown files
         found in the root directory of the repository.
      3. If neither exists, the repo is recorded with an empty docs list.

    All per-repo fetches run concurrently via asyncio.gather, so even a large
    starred list is handled as fast as the GitHub rate limit allows.

    Args:
        max_repos: Optional cap on the number of starred repos to process.
                   Defaults to None (fetch all starred repos).

    Returns:
        A list of dicts, one per repo, with keys:
            repo         - "owner/name"
            description  - repo description string
            stars        - stargazer count
            language     - primary language
            url          - HTML URL
            doc_source   - "readme" | "root_markdown" | None
            docs         - list of {name, path, size, content} dicts
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "markdown-fetcher", oauth_token=oauth_token)

        # ── Step 1: Collect starred repos (sequential; getiter handles pagination) ──
        print("Fetching starred repositories...")
        starred_repos: List[Dict[Any, Any]] = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            if max_repos and len(starred_repos) >= max_repos:
                break

        print(f"Found {len(starred_repos)} starred repositories")
        print("Fetching documentation for all repos concurrently...\n")

        # ── Step 2: Define per-repo coroutine ──────────────────────────────────────
        async def fetch_repo_docs(repo: Dict[Any, Any]) -> Dict[Any, Any]:
            owner = repo["owner"]["login"]
            name = repo["name"]
            full_name = repo["full_name"]

            base = {
                "repo": full_name,
                "description": repo.get("description"),
                "stars": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "url": repo.get("html_url"),
            }

            # Try README first via the dedicated /readme endpoint
            try:
                readme_data = await gh.getitem(f"/repos/{owner}/{name}/readme")
                content = base64.b64decode(readme_data["content"]).decode("utf-8")
                return {
                    **base,
                    "doc_source": "readme",
                    "docs": [
                        {
                            "name": readme_data["name"],
                            "path": readme_data["path"],
                            "size": readme_data["size"],
                            "content": content,
                        }
                    ],
                }
            except GitHubException as e:
                if e.status_code != 404:
                    # Surface unexpected errors without crashing the whole gather
                    print(f"  Warning: unexpected error fetching README for {full_name}: {e}")

            # README absent — fall back to all root-level .md files
            markdown_files = await get_root_markdown_files(gh, owner, name)
            if markdown_files:
                tasks = [
                    fetch_markdown_content(gh, owner, name, f["path"])
                    for f in markdown_files
                ]
                file_results = await asyncio.gather(*tasks)
                return {
                    **base,
                    "doc_source": "root_markdown",
                    "docs": [r for r in file_results if r.get("success")],
                }

            # No documentation found at all
            return {**base, "doc_source": None, "docs": []}

        # ── Step 3: Fan out — all repos fetched concurrently ──────────────────────
        results: List[Dict[Any, Any]] = await asyncio.gather(
            *[fetch_repo_docs(repo) for repo in starred_repos]
        )

        # ── Summary ────────────────────────────────────────────────────────────────
        with_readme = [r for r in results if r["doc_source"] == "readme"]
        with_md = [r for r in results if r["doc_source"] == "root_markdown"]
        no_docs = [r for r in results if r["doc_source"] is None]

        print(f"\n{'='*70}")
        print("DOCUMENTATION FETCH SUMMARY")
        print(f"{'='*70}")
        print(f"Total repos processed : {len(results)}")
        print(f"  README found        : {len(with_readme)}")
        print(f"  Root markdown files : {len(with_md)}")
        print(f"  No docs found       : {len(no_docs)}")
        if gh.rate_limit:
            print(f"Rate limit remaining  : {gh.rate_limit.remaining}")

        return results


async def get_specific_markdown_files(owner: str, repo: str, filenames: List[str]):
    """
    Fetch specific markdown files by name.
    Useful when you know which files you want (README.md, CONTRIBUTING.md, etc.)
    """
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "markdown-fetcher", oauth_token=oauth_token)

        print(f"Fetching specific markdown files from {owner}/{repo}:\n")

        async def fetch_file(filename):
            try:
                file_data = await gh.getitem(f"/repos/{owner}/{repo}/contents/{filename}")
                content = base64.b64decode(file_data['content']).decode('utf-8')
                print(f"✓ Found {filename} ({file_data['size']} bytes)")
                return {
                    'filename': filename,
                    'content': content,
                    'size': file_data['size'],
                    'found': True
                }
            except GitHubException as e:
                print(f"✗ {filename} not found (HTTP {e.status_code})")
                return {
                    'filename': filename,
                    'found': False
                }

        # Fetch all specified files concurrently
        results = await asyncio.gather(*[fetch_file(f) for f in filenames])

        found_files = [r for r in results if r['found']]
        print(f"\nSuccessfully fetched {len(found_files)}/{len(filenames)} files")

        return results


if __name__ == "__main__":
    print("Example 1: Get all markdown files from a specific repo")
    print("="*70)
    asyncio.run(get_all_markdown_from_repo("gidgethub", "gidgethub"))

    print("\n\n")

    print("Example 2: Compare /readme vs all markdown files")
    print("="*70)
    asyncio.run(compare_readme_vs_all_markdown("PyGithub", "PyGithub"))

    print("\n\n")

    print("Example 3: Fetch specific markdown files")
    print("="*70)
    filenames = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE.md"]
    asyncio.run(get_specific_markdown_files("gidgethub", "gidgethub", filenames))

    print("\n\n")

    print("Example 4: Get all markdown from starred repos")
    print("="*70)
    asyncio.run(fetch_all_markdown_from_starred_repos(max_repos=5))

    print("\n\n")

    print("Example 5: Fetch starred repos – README first, root .md files as fallback")
    print("="*70)
    asyncio.run(fetch_starred_repos_with_docs(max_repos=10))
