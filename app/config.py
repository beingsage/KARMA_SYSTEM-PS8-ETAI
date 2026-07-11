from pathlib import Path
import os
import sys


def load_env_file(env_file: Path = None) -> dict:
    """Load configuration from .env.local file."""
    if env_file is None:
        # Look for .env.local in the repository root directory
        env_file = Path(__file__).resolve().parents[1] / ".env.local"
    
    config = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    
    return config


def get_env(key: str, default: str = "", env_config: dict = None) -> str:
    """Get environment variable with fallback to .env.local file.
    
    Priority:
    1. Environment variables (command-line override)
    2. .env.local file
    3. Default value
    """
    # First check actual environment variables (allows command-line override)
    if key in os.environ:
        return os.environ[key]
    
    # Then check loaded config from .env.local
    if env_config and key in env_config:
        return env_config[key]
    
    # Finally use default
    return default


# Load configuration from .env.local file at module initialization
_env_config = load_env_file()

# Print which config file was loaded
_env_file_path = Path(__file__).resolve().parents[1] / ".env.local"
if _env_file_path.exists():
    print(f"[CONFIG] Loaded configuration from: {_env_file_path}", file=sys.stderr)
else:
    print(f"[CONFIG] Using defaults (.env.local not found, create one from .env.example)", file=sys.stderr)


class Settings:
    app_name: str = "industrial-pdf-graph-backend"
    data_dir: Path = Path(get_env("STRUCTURED_DATA_DIR", "data", _env_config))
    uploads_dir: Path = data_dir / "uploads"
    jobs_dir: Path = data_dir / "jobs"
    
    # Neo4j
    neo4j_uri: str = get_env("NEO4J_URI", "bolt://localhost:7687", _env_config)
    neo4j_user: str = get_env("NEO4J_USER", "neo4j", _env_config)
    neo4j_password: str = get_env("NEO4J_PASSWORD", "industrial_graph_password", _env_config)
    
    # LLM models
    qwen_model: str = get_env("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct", _env_config)
    
    # P&ID Symbol Detection - YOLOv12
    pid_yolo_weights: str = get_env("PID_YOLO_WEIGHTS", "", _env_config)
    pid_yolo_model_name: str = get_env("PID_YOLO_MODEL", "yolov8n.pt", _env_config)
    
    # Zero-shot Object Detection - GroundingDINO
    grounding_dino_model: str = get_env("GROUNDING_DINO_MODEL", "groundingdino_swint_ogc.pth", _env_config)
    
    # Segmentation - SAM2
    sam_model_type: str = get_env("SAM_MODEL_TYPE", "vit_b", _env_config)
    sam_model_name: str = get_env("SAM_MODEL_NAME", "sam_vit_b_01ec64.pth", _env_config)
    
    # Entity Extraction - GLiNER
    gliner_model: str = get_env("GLINER_MODEL", "urchade/gliner_medium-v2.1", _env_config)
    gliner_fine_tuned: str = get_env("GLINER_FINETUNED", "models/gliner-industrial-v1", _env_config)
    
    # Relation Extraction - GLiREL
    glirel_model: str = get_env("GLIREL_MODEL", "jackboyla/glirel-large-v0", _env_config)
    
    # Entity Linking - BLINK
    blink_model: str = get_env("BLINK_MODEL", "facebook/genre-linking-blink", _env_config)
    
    # Embeddings - BGE-M3
    embedding_model: str = get_env("EMBEDDING_MODEL", "BAAI/bge-m3", _env_config)
    
    # Reranker - BGE-Reranker-v2
    reranker_model: str = get_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3", _env_config)
    
    # Entity extraction fallback model
    entity_model: str = get_env("ENTITY_MODEL", "dslim/bert-base-NER", _env_config)
    
    # Vector Database - Qdrant
    qdrant_host: str = get_env("QDRANT_HOST", "localhost", _env_config)
    qdrant_port: int = int(get_env("QDRANT_PORT", "6333", _env_config))
    qdrant_collection: str = get_env("QDRANT_COLLECTION", "industrial_documents", _env_config)
    qdrant_vector_size: int = 1024  # BGE-M3 outputs 1024D vectors
    
    # Graph Reasoning - Microsoft GraphRAG
    graphrag_enabled: bool = get_env("GRAPHRAG_ENABLED", "true", _env_config).lower() == "true"
    graphrag_config_dir: str = get_env("GRAPHRAG_CONFIG_DIR", "models/graphrag_config", _env_config)
    
    # Foundation LLM - Qwen 3
    qwen3_model: str = get_env("QWEN3_MODEL", "Qwen/Qwen2.5-0.5B-Instruct", _env_config)
    qwen3_context_length: int = int(get_env("QWEN3_CONTEXT_LENGTH", "32768", _env_config))
    qwen3_temperature: float = float(get_env("QWEN3_TEMPERATURE", "0.7", _env_config))
    qwen3_max_tokens: int = int(get_env("QWEN3_MAX_TOKENS", "2048", _env_config))
    
    # Time-Series - TimesFM (Google)
    timesfm_model: str = get_env("TIMESFM_MODEL", "google/timesfm-1.0-200m", _env_config)
    timesfm_context_len: int = int(get_env("TIMESFM_CONTEXT_LEN", "512", _env_config))
    timesfm_forecast_len: int = int(get_env("TIMESFM_FORECAST_LEN", "128", _env_config))
    
    # Remaining Useful Life - Temporal Fusion Transformer
    tft_model: str = get_env("TFT_MODEL", "models/tft_rul_predictor", _env_config)
    tft_lookback_window: int = int(get_env("TFT_LOOKBACK", "100", _env_config))
    tft_forecast_window: int = int(get_env("TFT_FORECAST", "50", _env_config))
    
    # Agent Framework - LangGraph
    langgraph_enabled: bool = get_env("LANGGRAPH_ENABLED", "true", _env_config).lower() == "true"
    agent_model: str = get_env("AGENT_MODEL", "qwen3", _env_config)
    
    # ============================================================================
    # Execution Mode Configuration
    # ============================================================================
    # Options: 'cpu' (no time restriction, for local testing) or 'gpu' (optimized for speed on server)
    execution_mode: str = get_env("EXECUTION_MODE", "cpu", _env_config).lower()
    
    # Device settings - can override with specific values or use AUTO for mode-based defaults
    _device_detection: str = get_env("DEVICE_FOR_DETECTION", "AUTO", _env_config)
    _device_segmentation: str = get_env("DEVICE_FOR_SEGMENTATION", "AUTO", _env_config)
    _device_extraction: str = get_env("DEVICE_FOR_EXTRACTION", "AUTO", _env_config)
    _device_embedding: str = get_env("DEVICE_FOR_EMBEDDING", "AUTO", _env_config)
    
    @property
    def device_for_detection(self) -> str:
        if self._device_detection != "AUTO":
            return self._device_detection
        return "cpu" if self.execution_mode == "cpu" else "cuda:0"
    
    @property
    def device_for_segmentation(self) -> str:
        if self._device_segmentation != "AUTO":
            return self._device_segmentation
        return "cpu" if self.execution_mode == "cpu" else "cuda:0"
    
    @property
    def device_for_extraction(self) -> str:
        if self._device_extraction != "AUTO":
            return self._device_extraction
        return "cpu" if self.execution_mode == "cpu" else "cuda:0"
    
    @property
    def device_for_embedding(self) -> str:
        if self._device_embedding != "AUTO":
            return self._device_embedding
        return "cpu" if self.execution_mode == "cpu" else "cuda:0"
    
    # Processing timeouts (in seconds)
    table_extraction_timeout: int = int(get_env("TABLE_EXTRACTION_TIMEOUT", "0", _env_config))  # 0 = no limit
    ocr_processing_timeout: int = int(get_env("OCR_PROCESSING_TIMEOUT", "0", _env_config))    # 0 = no limit
    pdf_render_timeout: int = int(get_env("PDF_RENDER_TIMEOUT", "0", _env_config))            # 0 = no limit
    
    # Logging configuration
    log_level: str = get_env("LOG_LEVEL", "INFO", _env_config)
    verbose: bool = get_env("VERBOSE", "false", _env_config).lower() == "true"
    
    # Machine Learning Pipeline Configuration - Enable/disable specific pipeline stages
    enable_predictive_maintenance: bool = get_env("ENABLE_PM", "true", _env_config).lower() == "true"
    enable_rul_prediction: bool = get_env("ENABLE_RUL", "true", _env_config).lower() == "true"
    enable_anomaly_detection: bool = get_env("ENABLE_ANOMALY", "true", _env_config).lower() == "true"
    enable_root_cause_analysis: bool = get_env("ENABLE_RCA", "true", _env_config).lower() == "true"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.jobs_dir.mkdir(parents=True, exist_ok=True)

# Print execution mode on startup
print(f"[CONFIG] Execution Mode: {settings.execution_mode.upper()}", file=sys.stderr)
if settings.execution_mode == "cpu":
    print("[CONFIG] Running in CPU mode (no time restrictions, for local testing)", file=sys.stderr)
else:
    print(f"[CONFIG] Running in GPU mode (optimized for speed)", file=sys.stderr)
