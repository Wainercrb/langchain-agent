import { z } from 'zod';

// ============================================
// Zod Schemas — Runtime Validation
// ============================================

export const SourceSchema = z.object({
  content: z.string(),
  similarity_score: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export const ChatResponseSchema = z.object({
  response: z.string(),
  query: z.string(),
  sources: z.array(SourceSchema).optional().default([]),
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
  content: string;
  similarity_score: number;
  metadata?: Record<string, unknown>;
}

export interface ChatResponse {
  response: string;
  query: string;
  sources: Source[];
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

export { getErrorMessage };
