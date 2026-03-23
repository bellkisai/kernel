# shrimpk-openclaw

ShrimPK Echo Memory plugin for OpenClaw — persistent AI memory that works automatically.

## What It Does

- Injects relevant memories into every agent prompt (invisible, automatic)
- Stores user messages for future recall across sessions
- Works locally, offline, no cloud required
- 3.50ms response time at 100K memories

## Installation

1. Install ShrimPK: download MSI from [bellkis.com](https://bellkis.com)
2. Install plugin:
   ```bash
   openclaw plugin add shrimpk-memory
   ```

## How It Works

```
User sends message → OpenClaw agent starts
                        │
                  [before_prompt_build hook]
                        │
                  ShrimPK echo (3.50ms)
                        │
                  Memories injected into system prompt
                        │
                  LLM generates response with memory context
                        │
                  [agent_end hook]
                        │
                  User message stored for future recall
```

No MCP tool calls. No manual configuration. No agent code changes.

## Configuration

In `.claw/config.json`:
```json
{
  "plugins": {
    "shrimpk-memory": {
      "daemon_url": "http://127.0.0.1:11435",
      "max_results": 5,
      "enabled": true
    }
  }
}
```

## Requirements

- ShrimPK daemon running (`shrimpk-daemon` or installed via MSI)
- OpenClaw v1.0+

## License

Apache-2.0
