---
name: git-identity
description: Ensure all Git commits use the user's configured Git identity and never override it with Cursor, Agent, or synthetic identities.
---

# Git Identity Policy

When performing Git operations:

- Always use the Git identity already configured by the user.
- Never set `user.name`.
- Never set `user.email`.
- Never use `git -c user.name=...`.
- Never use `git -c user.email=...`.
- Never use "CursorAgent", "Cursor", "AI Agent", or any synthetic identity as the commit author.
- Never modify `.git/config` to change the user's identity.
- Do not run `git config user.name`.
- Do not run `git config user.email`.
- Allow Git to resolve the identity normally from repository configuration, falling back to the user's global configuration.
