# Production-Grade Retries with stamina

A practical guide using the [stamina](https://github.com/hynek/stamina) library (docs: [stamina.hynek.me](https://stamina.hynek.me/en/stable/)).

---

## Install

```bash
pip install stamina
```

---

## 1. Decorator: retry whole functions

Retry a callable when specific exceptions are raised. **You must pass `on=`** (stamina does not retry on generic `Exception` by default to avoid misuse).

```python
import httpx
import stamina

@stamina.retry(on=httpx.HTTPError, attempts=3)
def do_it(code: int) -> httpx.Response:
    resp = httpx.get(f"https://httpbin.org/status/{code}")
    resp.raise_for_status()
    return resp
```

- Retries up to **3 times** on `httpx.HTTPError` (or subclasses).
- Uses **exponential backoff + jitter** by default.
- Type hints of the decorated function are preserved.

---

## 2. Backoff hook: retry only when it makes sense

To retry only on **server errors (5xx)** and not on 4xx (e.g. 404, 403), use a **callable** for `on` that inspects the exception and returns `True`/`False` (or a custom wait in seconds / `timedelta`).

```python
def retry_only_on_5xx(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.HTTPError)

@stamina.retry(on=retry_only_on_5xx, attempts=3)
def do_it(code: int) -> httpx.Response:
    resp = httpx.get(f"https://httpbin.org/status/{code}")
    resp.raise_for_status()
    return resp
```

Returning a **float** or **timedelta** from the hook sets a **custom backoff** for that retry (e.g. from a `Retry-After` header); it bypasses the normal exponential backoff.

---

## 3. Retry parameters (production tuning)

| Parameter       | Default | Meaning |
|----------------|---------|--------|
| `on`           | *(required)* | Exception type(s) or callable (backoff hook). |
| `attempts`     | `10`    | Max number of attempts. `None` = no limit. |
| `timeout`      | `45.0`  | Max total time (seconds or `timedelta`) for all retries. `None` = no limit. |
| `wait_initial` | `0.1`   | Initial backoff before first retry. |
| `wait_max`     | `5.0`   | Cap on backoff between retries. |
| `wait_jitter`  | `1.0`   | Max random jitter added to backoff. |
| `wait_exp_base`| `2`     | Base for exponential backoff. |

Backoff formula:  
`min(wait_max, wait_initial * wait_exp_base^(attempt-1) + random(0, wait_jitter))`

Example with stricter limits and `timedelta`:

```python
import datetime as dt

@stamina.retry(
    on=httpx.HTTPError,
    attempts=5,
    timeout=dt.timedelta(seconds=30),
    wait_initial=0.5,
    wait_max=10.0,
)
def fetch(url: str) -> httpx.Response:
    return httpx.get(url)
```

---

## 4. Retry only a block of code (`retry_context`)

### First principles: why a special construct is needed

A decorator like `@stamina.retry` works by wrapping an entire function — it can call that function again from the top on failure. But sometimes you only want to retry **a few lines inside** a function, not the whole thing.

The two tools Python gives you for wrapping blocks of code are:

- **Context managers** (`with` statement) — can run teardown after a block, but **cannot re-execute the same block**
- **Iterators** (`for` statement) — can run a block multiple times, but **cannot catch exceptions** that escape the block body

Neither alone is enough. stamina combines them.

### How it works

```python
for attempt in stamina.retry_context(on=httpx.HTTPError, attempts=3):
    with attempt:
        resp = httpx.get("https://httpbin.org/status/500")
        resp.raise_for_status()
```

There are two moving parts:

**1. The `for` loop — `retry_context` is an iterator**

`retry_context` yields an `Attempt` object on each iteration. The loop is what makes re-execution possible — when an exception escapes the `with attempt:` block and is caught, control returns to the `for` header, which decides whether to yield another attempt or stop.

**2. The `with attempt:` block — each `Attempt` is a context manager**

The `Attempt` context manager's `__exit__` method is where the exception gets intercepted. When your code raises, `__exit__` asks: "Is this retriable? Have we exceeded `attempts` or `timeout`?" If yes to retry → it **suppresses** the exception, records it, waits for the backoff, and signals the iterator to yield the next attempt. If no → it re-raises and the loop ends.

The flow:

```
for attempt in retry_context(...)     ← iterator yields attempt #1
    with attempt:                      ← context manager enters
        <your code raises SomeError>
    __exit__ sees SomeError
        → retriable + attempts remain?
            YES → suppress, wait, loop back → yields attempt #2
            NO  → re-raise → loop ends, exception propagates
```

### When to use `retry_context` instead of `@stamina.retry`

- You only want to retry **part of a function**, not the whole thing
- The function does setup work before the retryable section (e.g. builds a request, acquires a resource) that you don't want repeated
- You're inside an async function and need `async for`

### The `attempt` object gives you visibility

```python
for attempt in stamina.retry_context(on=httpx.HTTPError, attempts=3):
    with attempt:
        if attempt.num > 1:
            print(f"Retry #{attempt.num}, next wait ≥ {attempt.next_wait:.1f}s")
        resp = httpx.get("https://httpbin.org/status/500")
        resp.raise_for_status()
```

| Attribute | Type | Meaning |
|---|---|---|
| `attempt.num` | `int` | Which attempt this is (starts at 1) |
| `attempt.next_wait` | `float` | Seconds of backoff before the next attempt (lower bound, excludes jitter) |

### Async — identical, just `async for`

```python
async for attempt in stamina.retry_context(on=aiohttp.ClientError, attempts=3):
    with attempt:
        result = await gh.getitem(...)
```

The `with attempt:` stays a regular (sync) context manager even in async code — only the iteration is async. `__exit__` is called synchronously the moment the exception escapes; the async waiting happens inside the iterator between yields.

### One-sentence summary

`retry_context` is an **iterator of context managers**: the `for` loop provides re-execution, the `with` block provides exception interception, and stamina's logic lives in between — deciding whether to suppress, wait, and loop again, or let the exception through.

Same parameters as `stamina.retry()`.

---

## 5. Retry a single call (no decorator)

Use `RetryingCaller` (sync) or `AsyncRetryingCaller` (async) to retry one function/method call without decorating it.

```python
def do_something(url: str, some_kw: int):
    resp = httpx.get(url)
    resp.raise_for_status()
    return resp

rc = stamina.RetryingCaller(attempts=5)

# Option A: pass exception type each time
rc(httpx.HTTPError, do_something, "https://httpbin.org/status/500", some_kw=42)

# Option B: bind exception type once, then call
bound_rc = rc.on(httpx.HTTPError)
bound_rc(do_something, "https://httpbin.org/status/500", some_kw=42)
```

---

## 6. Async (asyncio and Trio)

Same API: use `@stamina.retry` on async functions and `async for` with `retry_context`.

```python
import datetime as dt

@stamina.retry(
    on=httpx.HTTPError,
    attempts=3,
    timeout=dt.timedelta(seconds=10),
)
async def do_it_async(code: int) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://httpbin.org/status/{code}")
    resp.raise_for_status()
    return resp

# Retry block (async)
async def with_block(code: int) -> httpx.Response:
    async for attempt in stamina.retry_context(on=httpx.HTTPError, attempts=3):
        with attempt:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://httpbin.org/status/{code}")
            resp.raise_for_status()
    return resp
```

For one-off async calls, use `stamina.AsyncRetryingCaller` and `bound_rc = arc.on(httpx.HTTPError)`.

---

## 7. Testing

**Disable retries entirely** (e.g. in pytest):

```python
import pytest
import stamina

@pytest.fixture(autouse=True, scope="session")
def deactivate_retries():
    stamina.set_active(False)
```

**Keep retries but remove backoff and cap attempts** (fast tests with retry behavior):

```python
stamina.set_testing(True)                    # no backoff, 1 attempt by default
stamina.set_testing(True, attempts=2)        # no backoff, max 2 attempts
stamina.set_testing(True, attempts=3, cap=True)  # cap at 3 if decorator allows more

# ... run code that uses retry_context or decorators ...

stamina.set_testing(False)
```

`set_testing` can also be used as a context manager.

---

## 8. Instrumentation (observability)

stamina calls **on-retry hooks** when a retry is scheduled (before waiting). Default behavior:

- If **prometheus-client** is installed: counter `stamina_retries_total` (labels: `callable`, `retry_num`, `error_type`).
- If **structlog** is installed: structlog at warning level.
- Else: standard library **logging** at warning level.

**Custom hook:**

```python
def my_hook(details: stamina.instrumentation.RetryDetails) -> None:
    print("retry scheduled", details.name, details.retry_num, details.caused_by)

stamina.instrumentation.set_on_retry_hooks([my_hook])
```

**Disable instrumentation:**

```python
stamina.instrumentation.set_on_retry_hooks([])
```

**Manual integrations:**

```python
from stamina.instrumentation import (
    PrometheusOnRetryHook,
    StructlogOnRetryHook,
    LoggingOnRetryHook,
)
stamina.instrumentation.set_on_retry_hooks([PrometheusOnRetryHook, StructlogOnRetryHook])
```

**Lazy hook** (e.g. for CLIs) via `RetryHookFactory`: pass a callable that returns the actual hook so heavy imports run only on first retry.

---

## 9. Summary checklist for production

- Use **explicit `on=`** (exception type or backoff hook).
- Prefer a **backoff hook** when you must retry only on a subset of errors (e.g. 5xx only).
- Set **`attempts`** and/or **`timeout`** so retries are bounded.
- Rely on default **exponential backoff + jitter** unless you need custom waits (e.g. `Retry-After` via hook return value).
- In tests: **`stamina.set_active(False)`** or **`stamina.set_testing(True, attempts=1)`** to keep tests fast and deterministic.
- Use built-in or custom **instrumentation** (Prometheus, structlog, logging, hooks) for visibility.

---

## 10. Real-world example: GitHub API fetcher (`fetch_repo_docs`)

### Context

`fetch_all_markdown_files.py` fetches documentation from GitHub starred repos. The inner coroutine `fetch_repo_docs` makes three distinct network calls:

| Call | Helper / location |
|---|---|
| `gh.getitem(.../readme)` | Inline in `fetch_repo_docs` |
| `gh.getitem(.../contents/)` | Inside `get_root_markdown_files` |
| `gh.getitem(.../contents/{path})` | Inside `fetch_markdown_content` |

All of these run concurrently via `asyncio.gather`, so transient failures affect individual repos silently rather than crashing the whole run.

### Where retry helps vs. hurts

**Retry — yes** (transient failures):
- `aiohttp.ClientError` — connection resets, timeouts, DNS hiccups
- `GitHubException` with a **5xx** status — GitHub server errors (502, 503 are common)
- `GitHubException` with **429** — rate limit exceeded; ideally return the `Retry-After` seconds as a custom backoff from the hook

**Retry — no** (deterministic errors):
- **404** — README genuinely doesn't exist; already used as a branch condition to fall back to root markdown files
- **403 Forbidden** — permissions problem that won't resolve on retry

### A concrete bug retry would fix

When a transient **503** hits the README fetch, the current code hits the `!= 404` branch, prints a warning, and silently falls through to the markdown fallback. This wastes extra API calls and loses the README. A retry prevents that entirely.

### Implementation

Define a shared backoff hook, then apply it to the two helper functions and the README block:

```python
import stamina
import aiohttp
from gidgethub import GitHubException

def _is_retriable(exc: Exception) -> bool:
    """Retry on transient network/server errors; skip deterministic 404/403."""
    if isinstance(exc, GitHubException):
        return exc.status_code not in (404, 403)
    return isinstance(exc, aiohttp.ClientError)


@stamina.retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def get_root_markdown_files(gh, owner, repo):
    # existing implementation unchanged
    ...


@stamina.retry(on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0)
async def fetch_markdown_content(gh, owner, repo, file_path):
    # existing implementation unchanged
    ...


async def fetch_repo_docs(repo):
    owner = repo["owner"]["login"]
    name  = repo["name"]
    ...

    # README fetch: use retry_context so 404 still propagates normally
    try:
        async for attempt in stamina.retry_context(
            on=_is_retriable, attempts=3, wait_initial=0.5, wait_max=10.0
        ):
            with attempt:
                readme_data = await gh.getitem(f"/repos/{owner}/{name}/readme")
        content = base64.b64decode(readme_data["content"]).decode("utf-8")
        return {**base, "doc_source": "readme", "docs": [...]}
    except GitHubException as e:
        if e.status_code != 404:
            print(f"Warning: unexpected error for {name}: {e}")

    # Fallback: root-level markdown files (get_root_markdown_files already retries)
    ...
```

### Decision table

| Call | Retry? | Reason |
|---|---|---|
| `.../readme` on transient errors | **Yes** | Prevents silent fallback on 503/network blips |
| `.../contents/` | **Yes** | Transient failure here loses the whole repo's file list |
| `.../contents/{path}` | **Yes** | Prevents silent `success: False` for individual files |
| 404 on `/readme` | **No** | Intentional branch condition, not an error |
| 403 Forbidden | **No** | Deterministic; retrying wastes rate-limit quota |

---

## References

- **GitHub:** [github.com/hynek/stamina](https://github.com/hynek/stamina)
- **Docs:** [stamina.hynek.me/en/stable/](https://stamina.hynek.me/en/stable/)
- **API:** [stamina.hynek.me/en/stable/api.html](https://stamina.hynek.me/en/stable/api.html)
- **Tutorial:** [stamina.hynek.me/en/stable/tutorial.html](https://stamina.hynek.me/en/stable/tutorial.html)
- **Testing:** [stamina.hynek.me/en/stable/testing.html](https://stamina.hynek.me/en/stable/testing.html)
- **Instrumentation:** [stamina.hynek.me/en/stable/instrumentation.html](https://stamina.hynek.me/en/stable/instrumentation.html)
