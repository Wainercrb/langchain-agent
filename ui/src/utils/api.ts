interface ChatRequest {
  query: string;
  top_k: number;
  include_sources: boolean;
  temperature: number;
}

interface Source {
  content: string;
  similarity_score: number;
  metadata: Record<string, unknown>;
}

interface ChatResponse {
  response: string;
  query: string;
  sources: Source[];
  execution_time_ms: number;
  model: string;
  run_id: string | null;
}

interface FeedbackPayload {
  run_id: string;
  feedback_type: 'like' | 'dislike';
  comment?: string;
}

interface FeedbackResponse {
  status: 'recorded' | 'accepted';
}

interface ChatSettings {
  topK: number;
  temperature: number;
  includeSources: boolean;
}

const API_BASE_URL = 'http://localhost:8000';
const REQUEST_TIMEOUT = 30000; // 30 seconds

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

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

    const response = await fetch(`${API_BASE_URL}/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `API error: ${response.statusText}`
      );
    }

    const data = await response.json();
    return data as ChatResponse;
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.');
      }
      throw error;
    }
    throw new Error('Failed to send chat message');
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
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const payload: FeedbackPayload = { run_id, feedback_type };
    if (comment) {
      payload.comment = comment;
    }

    const response = await fetch(`${API_BASE_URL}/v1/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Feedback API error: ${response.statusText}`
      );
    }

    const data = await response.json();
    return data as FeedbackResponse;
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('Feedback request timed out. Please try again.');
      }
      throw error;
    }
    throw new Error('Failed to submit feedback');
  }
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}
