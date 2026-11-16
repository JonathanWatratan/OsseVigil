export type AppPage = 'workspace' | 'research' | 'about'

export interface CategoryScore {
  category: 'cancer' | 'normal'
  display_name: 'Cancer' | 'Normal'
  score: number
  percentage: number
}

export interface PredictionResponse {
  schema_version: 'osteoinsight.classification.v1'
  model: {
    name: 'osteoinsight-resnet18'
    task: 'dataset_category_classification'
    training_dataset: 'Roboflow Bone Cancer Detection v1'
  }
  model_status: 'ready'
  model_version: string
  image: { width: number; height: number; format: 'JPEG' | 'PNG' }
  inference_ms: number
  predicted_category: 'cancer' | 'normal'
  predicted_display_name: 'Cancer' | 'Normal'
  scores: CategoryScore[]
  interpretation: string
  uncertainty_status: 'specialist_review_required'
  warnings: string[]
  disclaimer: string
}

export interface HealthResponse {
  status: 'ready' | 'model_not_ready'
  model_status: 'ready' | 'model_not_ready'
  model_version: string
  task: 'dataset_category_classification'
  checkpoint: string
  device: string
  detail: string
  class_names: ['cancer', 'normal']
}

export interface ApiErrorBody {
  detail?: {
    code?: string
    message?: string
  } | string
}
