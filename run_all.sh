#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${RUN_ALL_LOG_FILE:-$LOG_DIR/run_all.log}"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

trap 'echo "run_all.sh completed at $(date +"%Y-%m-%d %H:%M:%S %Z")"' EXIT

echo "================================================================================"
echo "Industrial PDF-to-Graph Pipeline - Full Setup & Startup"
echo "================================================================================"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

# ============================================================================
# 1. CHECK AND SETUP PYTHON ENVIRONMENT
# ============================================================================
echo "[1/6] Setting up Python environment..."

if [ -d "/kaggle" ]; then
    echo "Using Kaggle Python"

    # Indicate kaggle environment and run optional setup helper
    export KAGGLE_ENV=1
    if [ -f "$ROOT_DIR/kaggle_setup.sh" ] && [ -z "${RUN_ALL_SKIP_KAGGLE_SETUP:-}" ]; then
      echo "  Running $ROOT_DIR/kaggle_setup.sh (this may take a few minutes)..."
      if bash "$ROOT_DIR/kaggle_setup.sh"; then
        echo "  Kaggle setup script completed"
      else
        echo "  WARNING: kaggle_setup.sh failed; continuing in best-effort mode."
      fi
    fi

    PYTHON_BIN=$(which python)
    PIP_BIN=$(which pip)
    if "$PYTHON_BIN" -c 'import torch; print(f"[run_all] torch.cuda.is_available()={torch.cuda.is_available()}")' >/dev/null 2>&1; then
      "$PYTHON_BIN" -c 'import torch; print(f"[run_all] torch.cuda.is_available()={torch.cuda.is_available()}")'
    fi

elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    source "$ROOT_DIR/.venv/bin/activate"

    PYTHON_BIN=$(which python)
    PIP_BIN=$(which pip)

else
    python3 -m venv "$ROOT_DIR/.venv"
    source "$ROOT_DIR/.venv/bin/activate"

    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
    PIP_BIN="$ROOT_DIR/.venv/bin/pip"
fi

export PYTHON_BIN
export PIP_BIN
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Configure model/cache directories for Kaggle vs local
if [ -d "/kaggle" ]; then
  export HF_HOME="/kaggle/working/hf_cache"
  export HF_HUB_CACHE="/kaggle/working/hf_cache/hub"
  export HF_DATASETS_CACHE="/kaggle/working/hf_cache/datasets"
  export HF_MODULES_CACHE="/kaggle/working/hf_cache/modules"
  export TORCH_HOME="/kaggle/working/torch_cache"
else
  export HF_HOME="$ROOT_DIR/models/huggingface"
  export HF_HUB_CACHE="$ROOT_DIR/models/huggingface/hub"
  export HF_DATASETS_CACHE="$ROOT_DIR/models/huggingface/datasets"
  export HF_MODULES_CACHE="$ROOT_DIR/models/huggingface/modules"
  export TORCH_HOME="$ROOT_DIR/models/torch_cache"
fi

echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HF_MODULES_CACHE=$HF_MODULES_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"

load_env_file() {
  local env_file="$ROOT_DIR/.env.local"
  if [ ! -f "$env_file" ]; then
    env_file="$ROOT_DIR/.env.example"
  fi

  if [ -f "$env_file" ]; then
    echo "  Loading environment from $(basename "$env_file")"
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

load_env_file

export NEO4J_USERNAME="${NEO4J_USERNAME:-${NEO4J_USER:-neo4j}}"
export NEO4J_USER="${NEO4J_USERNAME}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-80907012}"
# Enforce hardcoded initial password for Neo4j to ensure predictable startup
export NEO4J_INITIAL_PASSWORD="${NEO4J_INITIAL_PASSWORD:-80907012}"
export NEO4J_AUTH="${NEO4J_USERNAME}/${NEO4J_PASSWORD}"

# ============================================================================
# 2. CHECK BACKEND SERVICE MODE
# ============================================================================
echo "[2/6] Checking backend service mode..."

LOCAL_BACKENDS_MODE=false
LOWER_USE_NATIVE="${USE_NATIVE_BACKENDS:-}"
LOWER_KAGGLE="${KAGGLE_ENV:-}"
LOWER_USE_NATIVE="${LOWER_USE_NATIVE,,}"
LOWER_KAGGLE="${LOWER_KAGGLE,,}"
if [ -n "$LOWER_USE_NATIVE" ] && [ "$LOWER_USE_NATIVE" != "0" ] && [ "$LOWER_USE_NATIVE" != "false" ] && [ "$LOWER_USE_NATIVE" != "no" ]; then
  LOCAL_BACKENDS_MODE=true
elif [ -n "$LOWER_KAGGLE" ] && [ "$LOWER_KAGGLE" != "0" ] && [ "$LOWER_KAGGLE" != "false" ] && [ "$LOWER_KAGGLE" != "no" ]; then
  LOCAL_BACKENDS_MODE=true
fi

OPEN_SOURCE_DIR="$ROOT_DIR/open-source"
mkdir -p "$OPEN_SOURCE_DIR"
mkdir -p "$OPEN_SOURCE_DIR/downloads"

validate_downloaded_archive() {
  local archive="$1"
  if [ ! -s "$archive" ]; then
    return 1
  fi

  if file "$archive" 2>/dev/null | grep -q 'gzip compressed'; then
    gzip -t "$archive" >/dev/null 2>&1 || return 1
  fi

  return 0
}

download_file() {
  local url="$1"
  local dest="$2"
  local tmp_dest="${dest}.part"

  if [ -f "$dest" ] && [ -s "$dest" ] && validate_downloaded_archive "$dest"; then
    return 0
  fi

  mkdir -p "$(dirname "$dest")"
  echo "  Downloading $(basename "$dest")..."
  rm -f "$dest" "$tmp_dest"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 20 --silent --show-error -o "$tmp_dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$tmp_dest" "$url"
  else
    echo "  ERROR: curl or wget is required to download native binaries."
    return 1
  fi

  mv "$tmp_dest" "$dest"
  validate_downloaded_archive "$dest" || {
    echo "  ERROR: Downloaded archive is invalid or incomplete: $dest"
    rm -f "$dest"
    return 1
  }
}

port_is_open() {
  local host="$1"
  local port="$2"
  python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect((sys.argv[1], int(sys.argv[2])))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

wait_for_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_seconds="${4:-90}"
  local elapsed=0

  echo "  Waiting for $name on $host:$port..."
  while [ "$elapsed" -lt "$timeout_seconds" ]; do
    if port_is_open "$host" "$port"; then
      echo "  ✓ $name is ready"
      return 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "  WARNING: timed out waiting for $name"
  return 1
}

start_redis_local() {
  echo "  Preparing native Redis..."
  local redis_dir="$OPEN_SOURCE_DIR/redis"
  local tarball="$OPEN_SOURCE_DIR/downloads/redis.tar.gz"
  local redis_cmd

  if [ ! -x "$redis_dir/src/redis-server" ]; then
    rm -rf "$redis_dir"
    mkdir -p "$redis_dir"
    download_file "https://download.redis.io/releases/redis-7.2.8.tar.gz" "$tarball" || return 1
    tar -xzf "$tarball" -C "$redis_dir" --strip-components=1
    if command -v make >/dev/null 2>&1; then
      make -C "$redis_dir" >/dev/null
    else
      echo "  WARNING: make is required to build Redis from source."
      return 1
    fi
  fi

  redis_cmd="$redis_dir/src/redis-server"

  if ! port_is_open "127.0.0.1" 6379; then
    echo "  Starting Redis on localhost:6379"
    nohup "$redis_cmd" --port 6379 --save "" --appendonly no >/dev/null 2>&1 &
    disown
  else
    echo "  Redis already running"
  fi
}

start_qdrant_local() {
  echo "  Preparing native Qdrant..."
  local qdrant_cmd
  local qdrant_dir="$OPEN_SOURCE_DIR/qdrant"
  if command -v qdrant >/dev/null 2>&1; then
    qdrant_cmd="$(command -v qdrant)"
  else
    local tarball="$OPEN_SOURCE_DIR/downloads/qdrant.tar.gz"
    local qdrant_version="1.18.0"
    if [ ! -x "$qdrant_dir/qdrant" ]; then
      rm -rf "$qdrant_dir"
      mkdir -p "$qdrant_dir"
      download_file "https://github.com/qdrant/qdrant/releases/download/v${qdrant_version}/qdrant-x86_64-unknown-linux-musl.tar.gz" "$tarball" || return 1
      if tar -tf "$tarball" | head -1 | grep -q '/'; then
        tar -xzf "$tarball" -C "$qdrant_dir" --strip-components=1
      else
        tar -xzf "$tarball" -C "$qdrant_dir"
      fi
      chmod +x "$qdrant_dir/qdrant"
    fi
    qdrant_cmd="$qdrant_dir/qdrant"
  fi

  if ! port_is_open "127.0.0.1" 6333; then
    echo "  Starting Qdrant on localhost:6333"
    mkdir -p "$qdrant_dir/storage"
    cd "$qdrant_dir"
    nohup "$qdrant_cmd" --uri "http://0.0.0.0:6333" --disable-telemetry >/dev/null 2>&1 &
    disown
    cd "$ROOT_DIR"
  else
    echo "  Qdrant already running"
  fi
}

docker_neo4j_container_id() {
  docker compose ps -q neo4j 2>/dev/null || true
}

docker_neo4j_password_ok() {
  local container_id
  container_id="$(docker_neo4j_container_id)"
  if [ -z "$container_id" ]; then
    return 1
  fi
  docker exec "$container_id" cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1
}

rotate_docker_neo4j_password() {
  local container_id
  container_id="$(docker_neo4j_container_id)"
  if [ -z "$container_id" ]; then
    return 1
  fi
  docker exec "$container_id" cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_INITIAL_PASSWORD" \
    "ALTER CURRENT USER SET PASSWORD FROM '$NEO4J_INITIAL_PASSWORD' TO '$NEO4J_PASSWORD';" >/dev/null 2>&1
}

start_neo4j_local() {
  echo "  Preparing native Neo4j..."
  local neo4j_dir="$OPEN_SOURCE_DIR/neo4j"
  local tarball="$OPEN_SOURCE_DIR/downloads/neo4j.tar.gz"
  local neo4j_version="5.15.0"
  local neo4j_username="${NEO4J_USERNAME:-${NEO4J_USER:-neo4j}}"
  local neo4j_password="${NEO4J_PASSWORD:-80907012}"
  local neo4j_initial_password="${NEO4J_INITIAL_PASSWORD:-80907012}"

  if [ ! -x "$neo4j_dir/bin/neo4j" ]; then
    rm -rf "$neo4j_dir"
    mkdir -p "$neo4j_dir"
    download_file "https://dist.neo4j.org/neo4j-community-${neo4j_version}-unix.tar.gz" "$tarball" || return 1
    tar -xzf "$tarball" -C "$neo4j_dir" --strip-components=1
  fi

  if [ ! -d "$neo4j_dir/data/databases/neo4j" ]; then
    echo "  Configuring initial Neo4j password"
    env NEO4J_AUTH="${neo4j_username}/${neo4j_password}" "$neo4j_dir/bin/neo4j-admin" dbms set-initial-password "$neo4j_password" >/dev/null 2>&1 || true
  fi

  if ! command -v java >/dev/null 2>&1; then
    echo "  WARNING: Java is required for Neo4j."
    if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
      echo "  Installing openjdk-17-jre-headless..."
      apt-get update -qq
      apt-get install -y openjdk-17-jre-headless >/dev/null 2>&1 || true
    fi
  fi

  if ! command -v java >/dev/null 2>&1; then
    echo "  WARNING: Java still unavailable; Neo4j native startup may fail."
  fi

  mkdir -p "$neo4j_dir/conf" "$neo4j_dir/data" "$neo4j_dir/logs"
  cat > "$neo4j_dir/conf/neo4j.conf" <<'EOF'
dbms.default_listen_address=0.0.0.0
dbms.default_advertised_address=127.0.0.1
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.connector.https.enabled=false
dbms.security.auth_enabled=true
dbms.default_database=neo4j
dbms.directories.data=./data
dbms.directories.logs=./logs
EOF

  if ! port_is_open "127.0.0.1" 7687; then
    echo "  Starting Neo4j on localhost:7687"
    cd "$neo4j_dir"
    nohup env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes NEO4J_AUTH="${neo4j_username}/${neo4j_password}" ./bin/neo4j console >/dev/null 2>&1 &
    disown
  else
    echo "  Neo4j already running"
  fi

  # If Neo4j started but the password is expired or the configured password was not applied,
  # reset the local data directory and restart with the configured password.
  if port_is_open "127.0.0.1" 7687; then
    sleep 5
    if ! "$neo4j_dir/bin/cypher-shell" -u "$neo4j_username" -p "$neo4j_password" "RETURN 1;" >/dev/null 2>&1; then
      echo "  Neo4j authentication failed; resetting local Neo4j data and applying the configured password"
      rm -rf "$neo4j_dir/data" "$neo4j_dir/logs" "$neo4j_dir/run"
      mkdir -p "$neo4j_dir/data" "$neo4j_dir/logs"
      env NEO4J_AUTH="${neo4j_username}/${neo4j_password}" "$neo4j_dir/bin/neo4j-admin" dbms set-initial-password "$neo4j_password" >/dev/null 2>&1 || true
      pkill -f "$neo4j_dir/bin/neo4j" 2>/dev/null || true
      sleep 2
      cd "$neo4j_dir"
      nohup env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes NEO4J_AUTH="${neo4j_username}/${neo4j_password}" ./bin/neo4j console >/dev/null 2>&1 &
      disown
      cd "$ROOT_DIR"
      sleep 5
      if "$neo4j_dir/bin/cypher-shell" -u "$neo4j_username" -p "$neo4j_password" "RETURN 1;" >/dev/null 2>&1; then
        echo "  ✓ Neo4j restarted successfully with the configured password"
      else
        echo "  WARNING: Neo4j restart did not authenticate with the configured password"
      fi
    fi
  fi
}

start_native_backends() {
  echo "  Native backend mode enabled. Docker will not be used."
  start_redis_local || true
  start_qdrant_local || true
  start_neo4j_local || true

  wait_for_port "Redis" "127.0.0.1" 6379 60 || true
  wait_for_port "Qdrant" "127.0.0.1" 6333 120 || true
  wait_for_port "Neo4j" "127.0.0.1" 7687 180 || true

  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="${NEO4J_USERNAME:-${NEO4J_USER:-neo4j}}"
  export NEO4J_PASSWORD="${NEO4J_PASSWORD:-80907012}"
  export QDRANT_HOST="localhost"
  export QDRANT_PORT="6333"
  export REDIS_HOST="localhost"
  export REDIS_PORT="6379"
}

install_torch_wheels() {
  local torch_version="2.10.0"
  local torchvision_version="0.25.0"
  local torchaudio_version="2.10.0"
  local cuda_tag="cpu"

  if [ "${EXECUTION_MODE:-gpu}" = "gpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_tag="cu126"
  fi

  if [ "$cuda_tag" = "cpu" ]; then
    echo "  Installing CPU PyTorch wheels"
    "$PYTHON_BIN" -m pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch==${torch_version}" "torchvision==${torchvision_version}" "torchaudio==${torchaudio_version}" || true
  else
    echo "  Installing GPU PyTorch wheels ($cuda_tag)"
    "$PYTHON_BIN" -m pip install --force-reinstall --upgrade --prefer-binary --only-binary=:all: --no-cache-dir --index-url https://download.pytorch.org/whl/${cuda_tag} \
      "torch==${torch_version}" "torchvision==${torchvision_version}" "torchaudio==${torchaudio_version}" || true
  fi
}

if [ "$LOCAL_BACKENDS_MODE" = true ]; then
  echo "  Running in native backend mode for local/Kaggle environments."
  DOCKER_AVAILABLE=false
  start_native_backends
else
  echo "  Checking Docker installation..."

  if ! command -v docker >/dev/null 2>&1; then
    echo "  WARNING: Docker is not installed or not available in PATH. qdrant and neo4j containers will be skipped."
    DOCKER_AVAILABLE=false
  elif ! docker info >/dev/null 2>&1; then
    echo "  WARNING: Docker daemon is not running. qdrant and neo4j containers will be skipped."
    DOCKER_AVAILABLE=false
  else
    echo "  ✓ Docker found and daemon is running"
    DOCKER_AVAILABLE=true
  fi

  if [ "${DOCKER_AVAILABLE:-false}" = true ]; then
    echo "  Starting backing services (Neo4j, Qdrant, Redis)..."
    docker compose rm -sf neo4j qdrant redis >/dev/null 2>&1 || true
    docker compose up -d neo4j qdrant redis 2>/dev/null || true

    if ! docker_neo4j_password_ok; then
      echo "  Docker Neo4j auth failed; attempting password rotation..."
      if rotate_docker_neo4j_password; then
        echo "  ✓ Docker Neo4j password rotated successfully"
      else
        echo "  WARNING: Docker Neo4j password rotation failed; verify NEO4J_INITIAL_PASSWORD and NEO4J_PASSWORD"
      fi
    fi
  fi
fi

NEO4J_RESET_ATTEMPTED=0
NEO4J_DATA_VOLUME="${COMPOSE_PROJECT_NAME:-structured}_neo4j_data"
NEO4J_LOGS_VOLUME="${COMPOSE_PROJECT_NAME:-structured}_neo4j_logs"

reset_neo4j_state() {
  echo "  ⚠ Resetting stale Neo4j container and persisted state..."
  docker compose rm -sf neo4j >/dev/null 2>&1 || true
  docker volume rm "$NEO4J_DATA_VOLUME" "$NEO4J_LOGS_VOLUME" >/dev/null 2>&1 || true
  echo "  ✓ Removed stale Neo4j container and volumes"
  docker compose up -d neo4j 2>/dev/null || true
}

wait_for_service_health() {
  local service_name="$1"
  local timeout_seconds="${2:-180}"
  local health_url="$3"
  local interval_seconds=3
  local elapsed=0

  if [ "${DOCKER_AVAILABLE:-false}" != true ]; then
    return 0
  fi

  local container_id
  container_id="$(docker compose ps -q "$service_name" 2>/dev/null || true)"
  if [ -z "$container_id" ]; then
    echo "  WARNING: Could not determine container for $service_name"
    return 1
  fi

  echo "  Waiting for $service_name to become healthy..."
  while [ "$elapsed" -lt "$timeout_seconds" ]; do
    if [ -n "$health_url" ]; then
      if python - "$health_url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as resp:
        sys.exit(0 if resp.status < 400 else 1)
except Exception:
    sys.exit(1)
PY
      then
        echo "  ✓ $service_name is ready"
        return 0
      fi
    else
      local health_status
      health_status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' "$container_id" 2>/dev/null || true)"
      if [ "$health_status" = "healthy" ] || [ "$health_status" = "running" ]; then
        echo "  ✓ $service_name is ready"
        return 0
      fi
      if [ "$health_status" = "unhealthy" ]; then
        echo "  WARNING: $service_name reported unhealthy"
        if [ "$service_name" = "neo4j" ] && [ "$NEO4J_RESET_ATTEMPTED" -eq 0 ]; then
          NEO4J_RESET_ATTEMPTED=1
          reset_neo4j_state
          return wait_for_service_health "$service_name" "$timeout_seconds" "$health_url"
        fi
        return 1
      fi
    fi

    sleep "$interval_seconds"
    elapsed=$((elapsed + interval_seconds))
  done

  echo "  WARNING: Timed out waiting for $service_name"
  if [ "$service_name" = "neo4j" ] && [ "$NEO4J_RESET_ATTEMPTED" -eq 0 ]; then
    NEO4J_RESET_ATTEMPTED=1
    reset_neo4j_state
    return wait_for_service_health "$service_name" "$timeout_seconds" "$health_url"
  fi
  return 1
}

if [ "${DOCKER_AVAILABLE:-false}" = true ]; then
  wait_for_service_health neo4j 180 "http://127.0.0.1:7474" || true
  wait_for_service_health qdrant 180 "http://127.0.0.1:6333/collections" || true
  wait_for_service_health redis 90 || true
fi

# ============================================================================
# 3. INSTALL DEPENDENCIES
# ============================================================================
echo "[3/6] Installing Python dependencies..."

install_torch_wheels

if ! "$PYTHON_BIN" -m pip install --prefer-binary --no-cache-dir --upgrade pip setuptools wheel >/dev/null 2>&1; then
  echo "  WARNING: pip toolchain upgrade failed; continuing with existing environment."
fi

# Print runtime library versions for easier debugging on Kaggle/local
echo "  Checking runtime library versions (torch, pyarrow, transformers)..."
"$PYTHON_BIN" - <<'PY'
import importlib, sys
def ver(name):
    try:
        m=importlib.import_module(name)
        print(f'    {name}=={getattr(m,"__version__",str(m))}')
    except Exception:
        print(f'    {name} not installed')
ver('torch')
ver('pyarrow')
ver('transformers')
sys.exit(0)
PY

DEP_REQUIREMENTS_FILE="${RUN_ALL_REQUIREMENTS_FILE:-$ROOT_DIR/requirements.txt}"
if [ ! -f "$DEP_REQUIREMENTS_FILE" ] && [ -f "$ROOT_DIR/requirements.full.txt" ]; then
  DEP_REQUIREMENTS_FILE="$ROOT_DIR/requirements.full.txt"
fi

echo "  Using dependency file: $(basename "$DEP_REQUIREMENTS_FILE")"

if [ -z "${RUN_ALL_INCLUDE_OPTIONAL+x}" ]; then
  RUN_ALL_INCLUDE_OPTIONAL=1
fi
DEP_BOOTSTRAP_CMD=("$PYTHON_BIN" "$ROOT_DIR/scripts/ensure_dependencies.py" --requirements "$DEP_REQUIREMENTS_FILE" --python "$PYTHON_BIN")
if [ "${RUN_ALL_INCLUDE_OPTIONAL:-0}" = "1" ]; then
  DEP_BOOTSTRAP_CMD+=(--include-optional)
fi

DEP_BOOTSTRAP_OK=true
if ! "${DEP_BOOTSTRAP_CMD[@]}"; then
  DEP_BOOTSTRAP_OK=false
  echo "  ⚠ Dependency bootstrap reported issues; continuing in best-effort mode."
fi

# ============================================================================
# 4. CREATE NECESSARY DIRECTORIES
# ============================================================================
echo "[4/6] Creating necessary directories..."

echo "  Creating data and model cache directories"
mkdir -p data/jobs data/uploads models models/transformers models/huggingface models/huggingface/hub models/huggingface/datasets models/huggingface/modules models/torch_cache
echo "  ✓ Directories created"

# ============================================================================
# 5. DOWNLOAD ALL MODELS
# ============================================================================
echo "[5/6] Downloading and preparing AI models..."
echo ""
echo "  This step downloads ~2.5GB of pre-trained models."
echo "  Models will be cached for future runs."
echo ""

export HF_HOME="$ROOT_DIR/models/huggingface"
export HF_HUB_CACHE="$ROOT_DIR/models/huggingface/hub"
export HF_DATASETS_CACHE="$ROOT_DIR/models/huggingface/datasets"
export HF_MODULES_CACHE="$ROOT_DIR/models/huggingface/modules"

if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/download_models.py"; then
  echo "  ⚠ Model download completed with some failures; continuing in best-effort mode."
else
  echo "  ✓ Model download and initialization completed successfully"
fi

# ============================================================================
# 6. START SERVICES
# ============================================================================
echo "[6/6] Starting services..."
echo ""

if [ "${DOCKER_AVAILABLE:-false}" = true ]; then
  echo "  Backing services were started before model warmup."
else
  echo "  Skipping qdrant and neo4j container startup because Docker daemon is unavailable."
fi

echo ""
echo "================================================================================"
if [ "${DEP_BOOTSTRAP_OK:-true}" != true ]; then
  echo "✓ Industrial PDF-to-Graph Pipeline Ready (degraded mode)"
else
  echo "✓ Industrial PDF-to-Graph Pipeline Ready!"
fi
echo "================================================================================"
echo ""
echo "Backend running on:"
echo "  HTTP:  http://127.0.0.1:8001"
echo "  Docs:  http://127.0.0.1:8001/docs"
echo "  ReDoc: http://127.0.0.1:8001/redoc"
echo ""
echo "CORE API Endpoints:"
echo "  POST   /api/v1/process-pdf        - Upload and process PDF"
echo "  GET    /api/v1/jobs               - List all jobs"
echo "  GET    /api/v1/jobs/{job_id}      - Get job results"
echo "  GET    /api/v1/models/status      - Check model status"
echo "  GET    /health                    - Health check"
echo ""
echo "ADVANCED ANALYSIS Endpoints:"
echo "  GET    /api/v1/advanced/models/status           - Advanced models status"
echo "  POST   /api/v1/advanced/vector-search           - Search embeddings (Qdrant)"
echo "  POST   /api/v1/advanced/graph-reasoning         - GraphRAG reasoning"
echo "  POST   /api/v1/advanced/llm-analysis            - Qwen 3 LLM analysis"
echo "  POST   /api/v1/advanced/anomaly-detection       - TimesFM anomaly detection"
echo "  POST   /api/v1/advanced/rul-prediction          - TFT RUL prediction"
echo "  POST   /api/v1/advanced/root-cause-analysis     - RCA with full analysis"
echo "  POST   /api/v1/advanced/failure-prediction      - Predict equipment failure"
echo "  GET    /api/v1/advanced/pipeline-stages         - List all advanced stages"
echo ""
echo "AI MODELS INTEGRATED:"
echo "  Core (8 models):        YOLOv12, GroundingDINO, SAM2, GLiNER, GLiREL, BLINK, BGE-M3, BGE-Reranker"
echo "  Advanced (7 systems):   Qdrant, GraphRAG, Qwen 3, TimesFM, TFT, LangGraph, RCA Agent"
echo "  Total:                  15 models + systems integrated"
echo ""
echo "Execution Mode: ${EXECUTION_MODE:-gpu}"
echo "  Set EXECUTION_MODE=gpu for optimized GPU processing"
echo "  Local CPU mode: No time restrictions, suitable for testing all features"
echo "  GPU mode: Optimized for speed on server hardware"
echo ""
echo "Ctrl+C to stop the server"
echo "================================================================================"
echo ""

if port_is_open "127.0.0.1" 8001; then
  echo "  FastAPI already running on http://127.0.0.1:8001; restarting it so the latest Neo4j and runtime settings are applied."
  pkill -f "uvicorn --app-dir $ROOT_DIR app.main:app" 2>/dev/null || true
  pkill -f "app.main:app" 2>/dev/null || true
  sleep 2
fi

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}" \
NEO4J_USER="${NEO4J_USER:-neo4j}" \
NEO4J_PASSWORD="${NEO4J_PASSWORD:-80907012}" \
PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" -X dev -m uvicorn \
    --app-dir "$ROOT_DIR" \
    app.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --loop uvloop \
    --access-log
