import { z } from 'zod';

// ============================================
// Zod Schemas — Runtime Validation
// ============================================

export const SourceSchema = z.object({
  document_id: z.string(),
  filename: z.string(),
  similarity_score: z.number(),
  version_date: z.string().nullable().optional(),
  content_preview: z.string(),
  chunk_id: z.string(),
  metadata: z.record(z.unknown()).optional(),
});

export const ChatResponseSchema = z.object({
  response: z.string(),
  query: z.string(),
  sources: z.array(SourceSchema).nullable().optional().default(null),
  execution_time_ms: z.number(),
  model: z.string(),
  run_id: z.string().nullable(),
});

export const FeedbackPayloadSchema = z.object({
  run_id: z.string(),
  feedback_type: z.enum(['like', 'dislike']),
  comment: z.string().optional(),
});

export const FeedbackResponseSchema = z.object({
  status: z.enum(['recorded', 'accepted']),
});

export const SettingsSchema = z.object({
  topK: z.number().min(1).max(20).default(5),
  temperature: z.number().min(0).max(1).default(0.7),
  includeSources: z.boolean().default(true),
});

// ============================================
// TypeScript Interfaces
// ============================================

export interface ChatRequest {
  query: string;
  top_k: number;
  include_sources: boolean;
  temperature: number;
}

export interface Source {
  document_id: string;
  filename: string;
  similarity_score: number;
  version_date?: string | null;
  content_preview: string;
  chunk_id: string;
  metadata?: Record<string, unknown>;
}

export interface ChatResponse {
  response: string;
  query: string;
  sources: Source[] | null;
  execution_time_ms: number;
  model: string;
  run_id: string | null;
}

export interface FeedbackPayload {
  run_id: string;
  feedback_type: 'like' | 'dislike';
  comment?: string;
}

export interface FeedbackResponse {
  status: 'recorded' | 'accepted';
}

export interface ChatSettings {
  topK: number;
  temperature: number;
  includeSources: boolean;
}

// ============================================
// API Client
// ============================================

const API_BASE_URL = import.meta.env.PUBLIC_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT = 30000;

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<Response> {
  const { timeout = REQUEST_TIMEOUT, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${API_BASE_URL}/v1/health`, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
}

export const checkBackendHealth = checkHealth;

async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API error: ${response.statusText}`);
  }

  const data = await response.json();

  // Validate response with Zod
  try {
    return ChatResponseSchema.parse(data);
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('API response validation error:', error.errors);
      // Log the actual response for debugging
      console.error('Actual response:', JSON.stringify(data, null, 2));
    }
    throw new Error('Invalid response from server');
  }
}

export async function chatWithAgent(
  query: string,
  settings: ChatSettings
): Promise<ChatResponse> {
  return sendChat({
    query,
    top_k: settings.topK,
    temperature: settings.temperature,
    include_sources: settings.includeSources,
  });
}

export async function submitFeedback(
  run_id: string,
  feedback_type: 'like' | 'dislike',
  comment?: string
): Promise<FeedbackResponse> {
  const payload: FeedbackPayload = { run_id, feedback_type };
  if (comment) {
    payload.comment = comment;
  }

  // Validate payload
  try {
    FeedbackPayloadSchema.parse(payload);
  } catch (error) {
    throw new Error('Invalid feedback payload');
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/v1/feedback`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Feedback API error: ${response.statusText}`);
  }

  const data = await response.json();

  // Validate response with Zod
  try {
    return FeedbackResponseSchema.parse(data);
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Feedback response validation error:', error.errors);
    }
    throw new Error('Invalid feedback response from server');
  }
}

// ============================================
// Settings Validation
// ============================================

export function validateSettings(settings: unknown): ChatSettings {
  try {
    return SettingsSchema.parse(settings);
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.warn('Settings validation error, using defaults:', error.errors);
      return SettingsSchema.parse({});
    }
    throw error;
  }
}

// ============================================
// Decision & Health Types
// ============================================

export interface DecisionRecord {
  run_id: string;
  timestamp: string;
  query_preview: string;
  decision_quality: 'optimal' | 'suboptimal' | 'poor';
  model_used: string;
  chain_length: number;
  tools_used: string[];
  latency_ms: number;
  reasoning_summary: string | null;
}

export interface DecisionsResponse {
  total: number;
  page: number;
  per_page: number;
  decisions: DecisionRecord[];
  aggregates: Record<string, unknown> | null;
}

// ============================================
// Decision API
// ============================================

export async function fetchDecisions(): Promise<DecisionsResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/v1/decisions`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ============================================
// Streaming Types & Client
// ============================================

export interface StreamTokenEvent {
  type: 'token';
  content: string;
}

export interface StreamToolCallEvent {
  type: 'tool_call';
  tool: string;
  args: Record<string, unknown>;
}

export interface StreamToolResultEvent {
  type: 'tool_result';
  tool: string;
  summary: string;
}

export interface StreamDoneEvent {
  type: 'done';
  response: string;
  query: string;
  sources: Source[] | null;
  execution_time_ms: number;
  llm_latency_ms: number;
  model: string;
  run_id: string | null;
  usage_metadata: Record<string, number> | null;
  agent_type: string;
  tools_used: string[];
  chain_length: number;
  decision_quality: string;
  reasoning_summary: string;
}

export interface StreamErrorEvent {
  type: 'error';
  message: string;
}

export type StreamEvent =
  | StreamTokenEvent
  | StreamToolCallEvent
  | StreamToolResultEvent
  | StreamDoneEvent
  | StreamErrorEvent;

export type StreamEventCallback = (event: StreamEvent) => void;
export type StreamDoneCallback = (event: StreamDoneEvent) => void;
export type StreamErrorCallback = (error: Error) => void;

/**
 * Send a chat query via POST SSE streaming.
 *
 * Reads the ``Response`` body as a byte stream, parses SSE frames, and
 * invokes the provided callbacks for each event type.
 *
 * @param query          - User's question.
 * @param settings       - Chat settings (topK, temperature, etc.).
 * @param onEvent        - Called for every SSE event with the parsed payload.
 * @param onDone         - Called when the stream completes (after ``done`` event).
 * @param onError        - Called on connection / parse errors.
 * @returns              - A promise that resolves when the stream ends.
 */
export async function streamChat(
  query: string,
  settings: ChatSettings,
  onEvent: StreamEventCallback,
  onDone: StreamDoneCallback,
  onError: StreamErrorCallback,
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/v1/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: settings.topK,
        temperature: settings.temperature,
        include_sources: settings.includeSources,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.statusText}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';
    let currentData = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE frames from the buffer.
      // SSE frame format:
      //   event: <type>\n
      //   data: <json>\n
      //   \n
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // keep incomplete tail in buffer

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6).trim();
        } else if (line === '' && currentData) {
          // Empty line = end of SSE frame
          try {
            const parsed = JSON.parse(currentData) as StreamEvent;
            onEvent(parsed);

            if (parsed.type === 'done') {
              onDone(parsed);
            }
          } catch (parseError) {
            console.warn('SSE parse error:', parseError, 'raw:', currentData);
          }
          currentEvent = '';
          currentData = '';
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error : new Error(String(error)));
  }
}

export { getErrorMessage };
