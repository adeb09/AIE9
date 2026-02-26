# Concurrency Strategy: Collect-Then-Gather vs. Stream-And-Task

When fetching documentation for all starred repositories, there are two viable
async strategies. This document explains the tradeoff.

---

## Strategy A: Collect All Repos First, Then Fan Out (current approach)

```
getiter  →  [repo1, repo2, ..., repoN]  →  asyncio.gather(all N coroutines)
```

Pagination runs to completion before any README fetch begins. Once the full
list is in memory, all per-repo coroutines are submitted to the event loop at
once via `asyncio.gather`.

```python
starred_repos = []
async for repo in gh.getiter("/user/starred"):
    starred_repos.append(repo)

results = await asyncio.gather(
    *[fetch_repo_docs(repo) for repo in starred_repos]
)
```

**Pros**
- Simple and easy to reason about — `N` is known before any doc fetching starts.
- Progress logging and error handling are straightforward.
- `asyncio.gather` surfaces exceptions cleanly; no risk of silently dropped tasks.

**Cons**
- Pagination and doc fetching are strictly sequential phases — no overlap.
- If pagination is slow (e.g. 10,000+ starred repos), you wait for it entirely
  before the first README request is made.

---

## Strategy B: Kick Off Doc Fetch as Each Repo Arrives

```
getiter yields repo1  →  asyncio.create_task(fetch_repo_docs(repo1))
getiter yields repo2  →  asyncio.create_task(fetch_repo_docs(repo2))
...
getiter yields repoN  →  asyncio.create_task(fetch_repo_docs(repoN))
await asyncio.gather(*all tasks)
```

A task is scheduled immediately as each repo comes out of the async iterator,
so pagination and doc fetching overlap in wall-clock time.

```python
tasks = []
async for repo in gh.getiter("/user/starred"):
    tasks.append(asyncio.create_task(fetch_repo_docs(repo)))

results = await asyncio.gather(*tasks)
```

**Pros**
- Pagination and doc fetching run concurrently — repo 1's README fetch can
  complete while pages 3, 4, and 5 of starred repos are still being fetched.
- Maximum throughput when the starred list is very large.

**Cons**
- `asyncio.create_task` schedules the coroutine immediately. An exception raised
  before `gather` is awaited becomes an unhandled task exception unless you wrap
  carefully with `return_exceptions=True` or individual try/except blocks.
- Slightly harder to reason about: `N` is not known upfront, and tasks are
  already in flight during the collection loop.

---

## When Does the Difference Actually Matter?

The performance gap between the two strategies depends on how long pagination
takes relative to how long the doc fetches take.

| Starred repo count | Paginated API calls | Pagination time (est.) | Worth overlapping? |
|--------------------|--------------------|-----------------------|--------------------|
| < 500              | ≤ 5                | ~1 second             | No                 |
| 500 – 2,000        | 5 – 20             | ~2–4 seconds          | Marginally         |
| 10,000+            | 100+               | ~20–30 seconds        | Yes                |

For most users (dozens to a few hundred starred repos), pagination finishes in
under a second. The bulk of wall-clock time is spent in the concurrent doc
fetches themselves, so Strategy A and B produce essentially the same total
runtime.

---

## The Real Bottleneck: GitHub's Rate Limit

Both strategies are ultimately bounded by the same constraint: GitHub allows
**5,000 API requests per hour** for authenticated users. Fetching docs for 500
repos consumes roughly 500–1,000 requests (one `/readme` call, plus potentially
one `/contents/` call and N file fetches for the fallback path). No amount of
async concurrency changes that budget.

This means the practical speedup from overlapping pagination (Strategy B) is
small compared to the time saved by running all doc fetches concurrently
(which both strategies do).

---

## GitHub API Rate Limits

Understanding rate limits is critical when making many concurrent requests.

### REST API (what this code uses)

- **5,000 requests per hour** for authenticated requests (personal access token
  via `GITHUB_TOKEN`)
- The limit resets on a **rolling 1-hour window**, not at a fixed clock time
- Remaining requests and the reset timestamp are available after any request:

```python
print(gh.rate_limit.remaining)       # requests left
print(gh.rate_limit.reset_datetime)  # when the window resets
```

### GitHub Apps get more headroom

If using a GitHub App installation token instead of a personal access token,
the limit scales with the number of repositories the app is installed on — up
to **15,000 requests per hour**.

### GraphQL has a separate budget

The GraphQL API (v4) uses a **points-based system** rather than a flat request
count. Simple queries cost 1 point; queries returning many nodes cost more. The
budget is 5,000 points per hour, but a heavily nested query can cost
significantly more than 1 point.

### Search API is separately capped

The `/search` endpoints have their own lower limit of **30 requests per minute**
(authenticated), completely independent of the 5,000/hour pool.

### Secondary rate limits

Beyond the hourly quota, GitHub enforces secondary limits on the number of
**concurrent requests and requests per second**. Hammering the API with hundreds
of simultaneous requests — exactly what `asyncio.gather` does — can trigger
these. In practice this is rarely hit at the scale of a personal starred list,
but worth knowing if you scale this up significantly.

---

## Recommendation

Use **Strategy A** unless you are working with a user who has thousands of
starred repositories. It is simpler, safer around exception handling, and the
performance difference is negligible at typical scale.

Revisit **Strategy B** if profiling shows that pagination is a meaningful
fraction of total wall-clock time — i.e. when starred repo count exceeds
roughly 2,000.

---

## Can You Know the Total Starred Repo Count Ahead of Time?

Not directly from a single REST API call. This is a genuine gap in GitHub's
REST API v3 — the `/user` endpoint exposes `public_repos`, `followers`, and
`following`, but there is no `starred_count` field.

Knowing `N` upfront would allow smarter decisions: logging the total before
iteration begins, or conditionally switching to Strategy B when the count
exceeds a threshold. There are two workarounds.

### Workaround 1: Parse the `Link` Response Header

When you make the first paginated request to `/user/starred?per_page=100`,
GitHub returns a `Link` header:

```
Link: <...?page=2>; rel="next", <...?page=47>; rel="last"
```

Parsing the `rel="last"` page number and multiplying by `per_page` gives an
approximate total (the final page is usually partial, so this is an upper
bound, not exact). This costs one API call and works within the REST API.

The catch: `gidgethub`'s `getiter` abstracts pagination away entirely. To
inspect the `Link` header you would need to drop down to a raw `getitem` call
on the first page before handing off to `getiter`.

### Workaround 2: GitHub GraphQL API (exact count, one call)

GitHub's v4 GraphQL API exposes the total directly:

```graphql
query {
  viewer {
    starredRepositories {
      totalCount
    }
  }
}
```

`gidgethub` supports GraphQL via `gh.graphql(query)`, so this fits naturally
into the same async session:

```python
result = await gh.graphql("query { viewer { starredRepositories { totalCount } } }")
total = result["viewer"]["starredRepositories"]["totalCount"]
print(f"Fetching docs for {total} starred repos...")
```

This is the cleanest solution — one request, exact count, no header parsing.
