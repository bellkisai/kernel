/**
 * ShrimPK Echo Memory plugin for OpenClaw.
 *
 * Provides automatic, invisible persistent memory for all OpenClaw agents.
 * Memories are stored in the ShrimPK daemon (localhost:11435) and injected
 * into agent context before every prompt — no MCP, no tool calls, no latency.
 *
 * Installation: openclaw plugin add shrimpk-memory
 * Requirements: ShrimPK daemon running (shrimpk-daemon or via MSI installer)
 */

import { ShrimPKClient } from './client';

// Default daemon URL — can be overridden via plugin config
const DEFAULT_DAEMON_URL = 'http://127.0.0.1:11435';
const DEFAULT_MAX_RESULTS = 5;

interface PluginConfig {
  daemon_url?: string;
  max_results?: number;
  enabled?: boolean;
}

/**
 * OpenClaw plugin registration entry point.
 *
 * Called by the OpenClaw Gateway at startup. Registers lifecycle hooks
 * for automatic memory injection and storage.
 */
export function register(api: any) {
  // Read plugin config (user can override daemon URL in .claw/config.json)
  const config: PluginConfig = api.getConfig?.() ?? {};
  const daemonUrl = config.daemon_url ?? DEFAULT_DAEMON_URL;
  const maxResults = config.max_results ?? DEFAULT_MAX_RESULTS;
  const enabled = config.enabled !== false; // default true

  if (!enabled) {
    console.log('[shrimpk] Plugin disabled via config');
    return;
  }

  const client = new ShrimPKClient(daemonUrl);

  console.log(`[shrimpk] Echo Memory plugin loaded (daemon: ${daemonUrl})`);

  // Check daemon health on startup (non-blocking)
  client.health().then(healthy => {
    if (healthy) {
      console.log('[shrimpk] Daemon connected — memory active');
    } else {
      console.warn('[shrimpk] Daemon not reachable — memory disabled until daemon starts');
    }
  });

  /**
   * Hook: before_prompt_build
   *
   * Fires before every LLM call. Searches ShrimPK for memories relevant
   * to the user's message and injects them into the system prompt.
   *
   * This is the core integration — memories appear automatically without
   * the agent needing to call any tools.
   */
  api.registerHook('before_prompt_build', async (context: any) => {
    try {
      // Extract the user's message
      const userMessage = context.userMessage?.text
        ?? context.messages?.filter((m: any) => m.role === 'user').pop()?.content;

      if (!userMessage) return {};

      // Search for relevant memories (3.50ms P50 at 100K memories)
      const memories = await client.echo(userMessage, maxResults);

      if (memories.length === 0) return {};

      // Format memory block for system prompt injection
      const block = formatMemoryBlock(memories);

      return {
        prependSystemContext: block,
      };
    } catch (err) {
      // Never break the agent — silently skip if daemon is down
      return {};
    }
  });

  /**
   * Hook: agent_end
   *
   * Fires after each agent turn completes. Stores the user's message
   * in ShrimPK for future recall across sessions.
   */
  api.registerHook('agent_end', async (context: any) => {
    try {
      const userMessage = context.userMessage?.text
        ?? context.messages?.filter((m: any) => m.role === 'user').pop()?.content;

      if (!userMessage) return;

      // Fire-and-forget — don't block the response
      client.store(userMessage, 'openclaw').catch(() => {});
    } catch {
      // Silently ignore storage failures
    }
  });
}

/**
 * Format echo results into a system prompt injection block.
 */
function formatMemoryBlock(memories: Array<{ content: string; similarity: number }>): string {
  const items = memories
    .map((m, i) => `${i + 1}. ${m.content}`)
    .join('\n');

  return `[Echo Memory] Relevant context from previous conversations:\n${items}\nUse these memories naturally if relevant to the current conversation.\n`;
}
