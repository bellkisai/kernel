/**
 * ShrimPK daemon HTTP client.
 *
 * Communicates with the ShrimPK daemon's REST API on localhost.
 * All methods are async, all errors are caught — never throws.
 */

interface EchoResult {
  content: string;
  similarity: number;
  final_score: number;
  source: string;
  memory_id: string;
}

interface EchoResponse {
  results: EchoResult[];
  count: number;
  elapsed_ms: number;
}

interface HealthResponse {
  status: string;
  memories: number;
  version: string;
}

export class ShrimPKClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://127.0.0.1:11435', timeout: number = 5000) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.timeout = timeout;
  }

  /**
   * Check if the ShrimPK daemon is running and healthy.
   */
  async health(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeout);

      const resp = await fetch(`${this.baseUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!resp.ok) return false;
      const data: HealthResponse = await resp.json();
      return data.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * Search for memories relevant to a query.
   *
   * Returns up to `maxResults` echo results sorted by relevance.
   * Returns empty array on any error (daemon down, timeout, etc.).
   */
  async echo(query: string, maxResults: number = 5): Promise<EchoResult[]> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeout);

      const resp = await fetch(`${this.baseUrl}/api/echo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_results: maxResults }),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!resp.ok) return [];
      const data: EchoResponse = await resp.json();
      return data.results ?? [];
    } catch {
      return [];
    }
  }

  /**
   * Store a memory in ShrimPK for future recall.
   *
   * Fire-and-forget — errors are silently ignored.
   */
  async store(text: string, source: string = 'openclaw'): Promise<void> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeout);

      await fetch(`${this.baseUrl}/api/store`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source }),
        signal: controller.signal,
      });
      clearTimeout(timer);
    } catch {
      // Silently ignore — storage is fire-and-forget
    }
  }
}
