# CodeNarrator Instructions

## Goal
Turn CodeNarrator into a local-first autonomous codebase understanding agent.

## What the system should do
- Inspect a local repository
- Track what files/modules it has explored
- Build and refine an architecture summary
- Identify uncertainty or missing context
- Decide what file/module to inspect next
- Stop when understanding is coherent enough

## Priorities
- Keep the architecture simple and explicit
- Reuse the current FastAPI/backend structure where possible
- Prefer small Python modules over framework-heavy abstractions
- Optimize for local execution on a MacBook
- Avoid overengineering
- Favor learning and portfolio strength over production polish

## Engineering style
- Readable code over clever abstractions
- Small, well-named functions
- Explicit control flow
- Minimal but useful comments
- Incremental changes, not large rewrites