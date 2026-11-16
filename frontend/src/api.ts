import type { ApiErrorBody, HealthResponse, PredictionResponse } from './types'

export class OsteoInsightApiError extends Error {
  status: number
  code: string

  constructor(message: string, status: number, code = 'request_failed') {
    super(message)
    this.name = 'OsteoInsightApiError'
    this.status = status
    this.code = code
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch('/api/health', { signal })
  if (!response.ok) throw new OsteoInsightApiError('Backend health check failed.', response.status)
  return response.json() as Promise<HealthResponse>
}

export async function predictImage(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch('/api/predict', {
    method: 'POST',
    body,
    signal,
  })
  if (!response.ok) {
    let payload: ApiErrorBody = {}
    try {
      payload = (await response.json()) as ApiErrorBody
    } catch {
      // Keep the stable fallback message below.
    }
    const detail = typeof payload.detail === 'string' ? payload.detail : payload.detail?.message
    const code = typeof payload.detail === 'object' ? payload.detail?.code : undefined
    throw new OsteoInsightApiError(detail || 'Analysis request failed.', response.status, code)
  }
  return response.json() as Promise<PredictionResponse>
}
