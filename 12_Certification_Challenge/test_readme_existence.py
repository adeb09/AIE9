"""
Test script to check which starred repos have READMEs
This demonstrates that not all repos have READMEs
"""

import asyncio
import aiohttp
import os
import base64
from gidgethub.aiohttp import GitHubAPI
from gidgethub import GitHubException


async def check_readme_existence():
    """Check which starred repos have READMEs"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "readme-checker", oauth_token=oauth_token)

        print("Fetching your starred repositories...\n")

        # Get first 20 starred repos
        starred_repos = []
        async for repo in gh.getiter("/user/starred"):
            starred_repos.append(repo)
            if len(starred_repos) >= 20:
                break

        print(f"Checking README existence for {len(starred_repos)} repos...\n")

        # Check each repo for README
        async def check_readme(repo):
            owner = repo['owner']['login']
            name = repo['name']
            full_name = repo['full_name']

            try:
                # Try to fetch README
                readme_data = await gh.getitem(f"/repos/{owner}/{name}/readme")

                return {
                    'repo': full_name,
                    'has_readme': True,
                    'readme_name': readme_data['name'],
                    'readme_size': readme_data['size']
                }
            except GitHubException as e:
                # No README found (404 error)
                return {
                    'repo': full_name,
                    'has_readme': False,
                    'error_status': e.status_code
                }

        # Check all repos concurrently
        results = await asyncio.gather(*[check_readme(repo) for repo in starred_repos])

        # Separate repos with and without READMEs
        with_readme = [r for r in results if r['has_readme']]
        without_readme = [r for r in results if not r['has_readme']]

        # Display results
        print(f"{'='*70}")
        print(f"REPOSITORIES WITH README ({len(with_readme)}/{len(results)})")
        print(f"{'='*70}")
        for r in with_readme:
            print(f"✓ {r['repo']:<40} {r['readme_name']:>20} ({r['readme_size']:>6} bytes)")

        print(f"\n{'='*70}")
        print(f"REPOSITORIES WITHOUT README ({len(without_readme)}/{len(results)})")
        print(f"{'='*70}")
        for r in without_readme:
            print(f"✗ {r['repo']:<40} (HTTP {r['error_status']})")

        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Total repositories: {len(results)}")
        print(f"With README: {len(with_readme)} ({len(with_readme)/len(results)*100:.1f}%)")
        print(f"Without README: {len(without_readme)} ({len(without_readme)/len(results)*100:.1f}%)")
        print(f"\nRate limit remaining: {gh.rate_limit.remaining}")


async def test_specific_repos():
    """Test some well-known repos to show README behavior"""
    oauth_token = os.getenv("GITHUB_TOKEN")

    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "readme-checker", oauth_token=oauth_token)

        # Test repos - mix of with/without READMEs
        test_repos = [
            ("gidgethub", "gidgethub"),  # Has README
            ("torvalds", "linux"),        # Has README
            ("python", "cpython"),        # Has README
        ]

        print("Testing specific repositories:\n")

        for owner, repo in test_repos:
            try:
                readme_data = await gh.getitem(f"/repos/{owner}/{repo}/readme")
                print(f"✓ {owner}/{repo}")
                print(f"  README file: {readme_data['name']}")
                print(f"  Size: {readme_data['size']} bytes")
                print(f"  Download URL: {readme_data['download_url']}")
                print()
            except GitHubException as e:
                print(f"✗ {owner}/{repo}")
                print(f"  Status: {e.status_code}")
                print(f"  Error: {e}")
                print()


if __name__ == "__main__":
    print("Test 1: Check README existence in your starred repos")
    print("="*70)
    asyncio.run(check_readme_existence())

    print("\n\n")

    print("Test 2: Check specific well-known repositories")
    print("="*70)
    asyncio.run(test_specific_repos())
