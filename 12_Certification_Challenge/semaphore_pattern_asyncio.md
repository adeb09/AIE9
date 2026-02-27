# Semaphore Pattern for Concurrent Tasks in asyncio (Python)

A **semaphore** is a concurrency primitive that **limits how many coroutines can be in a given section of code at the same time**. It’s commonly used to cap concurrency (e.g. “at most N tasks at once”) or to rate-limit (e.g. “at most N requests at a time”).

## What It Does

- **`asyncio.Semaphore(n)`** – allows at most `n` “holders” at any moment.
- **Acquire**: `async with semaphore:` – if fewer than `n` are inside, you enter; otherwise you wait.
- **Release**: when you leave the `async with` block, one slot is freed and a waiting coroutine can proceed.

So you get **bounded concurrency**: many tasks can be scheduled, but only `n` run the protected part concurrently.

## Example: Limiting Concurrent Work

```python
import asyncio

async def limited_task(semaphore, task_id):
    async with semaphore:  # At most N coroutines here at once
        print(f"Task {task_id} running")
        await asyncio.sleep(1)
        print(f"Task {task_id} done")

async def main():
    semaphore = asyncio.Semaphore(3)  # Only 3 at a time
    tasks = [limited_task(semaphore, i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

Here, 10 tasks are created, but only 3 are inside the `async with semaphore:` block at any time. So you get concurrency (many tasks in flight) without unbounded load (e.g. 10 simultaneous HTTP requests).

## Why Use It

| Use case | Idea |
|----------|------|
| **Rate limiting** | Don’t open too many connections or requests to a server. |
| **Bounded concurrency** | Run 1000 tasks but only 10 “active” at once to avoid memory/CPU spikes. |
| **Resource protection** | Only N coroutines use a shared resource (e.g. DB connection pool) at a time. |

## Summary

- **Semaphore(n)** = “at most n coroutines in this section at once.”
- **`async with semaphore:`** = “wait for a slot, run, then release the slot.”
- Pattern: create one semaphore, share it across all tasks that need the same limit, and wrap the part you want to limit in `async with semaphore:`.
