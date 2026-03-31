#!/bin/bash
# spark-config — Capture, manage, and restore DGX Spark configurations
# Usage: spark-config <command> [args]
#
# Commands:
#   snapshot <name> [description]  — Capture current running state as a named config
#   list                          — List saved configs
#   show <name>                   — Show config details
#   diff <name>                   — Compare current state vs saved config
#   apply <name>                  — Apply a saved config (interactive confirmation)
#   backup                        — Sync all configs to homeserver
#   restore-from-backup           — Pull configs from homeserver (disaster recovery)

set -euo pipefail

CONFIG_DIR="/home/claude/spark-configs"
HOMESERVER_BACKUP="${SPARK_BACKUP_DEST:?Set SPARK_BACKUP_DEST environment variable}"
SSH_KEY="${SPARK_SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[spark-config]${NC} $*"; }
warn() { echo -e "${YELLOW}[spark-config]${NC} $*"; }
err() { echo -e "${RED}[spark-config]${NC} $*" >&2; }

# ──────────────────────────────────────────────────────────────────────
# SNAPSHOT — Capture current running state
# ──────────────────────────────────────────────────────────────────────
cmd_snapshot() {
    local NAME="${1:?Usage: spark-config snapshot <name> [description]}"
    local DESC="${2:-Snapshot taken $(date -Iseconds)}"
    local SNAP_DIR="$CONFIG_DIR/$NAME"

    if [ -d "$SNAP_DIR" ]; then
        warn "Config '$NAME' already exists. Overwriting."
        rm -rf "$SNAP_DIR"
    fi

    mkdir -p "$SNAP_DIR"
    log "Capturing config '$NAME'..."

    # ── System info ──
    log "  System info..."
    cat > "$SNAP_DIR/system.json" << SYSEOF
{
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "arch": "$(uname -m)",
    "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')",
    "cuda": "$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | awk '{print $NF}' || echo 'unknown')",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')",
    "total_ram_gb": $(free -g | awk '/Mem:/{print $2}'),
    "snapshot_date": "$(date -Iseconds)",
    "description": "$DESC"
}
SYSEOF

    # ── Container configs ──
    log "  Container configs..."
    mkdir -p "$SNAP_DIR/containers"
    for CONTAINER in $(docker ps -a --format '{{.Names}}' | sort); do
        docker inspect "$CONTAINER" > "$SNAP_DIR/containers/${CONTAINER}.json" 2>/dev/null
    done

    # ── Generate docker-compose.yml ──
    log "  Generating docker-compose.yml..."
    _generate_compose "$SNAP_DIR"

    # ── Sysctl tuning ──
    log "  Sysctl config..."
    sysctl -a 2>/dev/null | grep -E "^(vm\.(swappiness|min_free_kbytes|dirty_ratio|dirty_background_ratio|overcommit)|net\.(core\.(rmem_max|wmem_max|somaxconn)|ipv4\.tcp_(rmem|wmem|congestion_control|tw_reuse)))" > "$SNAP_DIR/sysctl.conf" 2>/dev/null || true

    # ── Network config ──
    log "  Network config..."
    mkdir -p "$SNAP_DIR/network"
    nmcli connection show 2>/dev/null > "$SNAP_DIR/network/connections.txt" || true
    for CONN in $(nmcli -t -f NAME connection show 2>/dev/null); do
        SAFE_NAME=$(echo "$CONN" | tr ' ' '_')
        nmcli connection show "$CONN" > "$SNAP_DIR/network/${SAFE_NAME}.conf" 2>/dev/null || true
    done

    # ── NM connection files (if readable) ──
    for F in /run/NetworkManager/system-connections/*.nmconnection; do
        [ -f "$F" ] && cp "$F" "$SNAP_DIR/network/" 2>/dev/null || true
    done

    # ── Docker images with digests ──
    log "  Image inventory..."
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -v '<none>' > "$SNAP_DIR/images.txt"

    # ── Model inventory ──
    log "  Model inventory..."
    _capture_models "$SNAP_DIR"

    # ── GPU memory allocation ──
    log "  GPU state..."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$SNAP_DIR/gpu_processes.csv" 2>/dev/null || true
    nvidia-smi -q > "$SNAP_DIR/gpu_full.txt" 2>/dev/null || true

    # ── Prometheus/monitoring config ──
    log "  Monitoring config..."
    mkdir -p "$SNAP_DIR/monitoring"
    # Capture which metrics endpoints exist
    for PORT in 8000 8001 8002 9100 9400; do
        if curl -sf "http://localhost:$PORT/metrics" > /dev/null 2>&1; then
            echo "localhost:$PORT" >> "$SNAP_DIR/monitoring/metrics_endpoints.txt"
        fi
    done

    # ── Docker volumes ──
    log "  Volume inventory..."
    docker volume ls --format '{{.Name}} {{.Driver}}' > "$SNAP_DIR/volumes.txt"

    # ── Generate restore script ──
    log "  Generating restore.sh..."
    _generate_restore "$SNAP_DIR" "$NAME"

    # ── Summary ──
    local SIZE=$(du -sh "$SNAP_DIR" | cut -f1)
    log "Config '$NAME' saved to $SNAP_DIR ($SIZE)"
    log "Containers captured: $(ls "$SNAP_DIR/containers/" | wc -l)"
    log "Run 'spark-config backup' to sync to homeserver"
}

# ──────────────────────────────────────────────────────────────────────
# Generate docker-compose.yml from running containers
# ──────────────────────────────────────────────────────────────────────
_generate_compose() {
    local DIR="$1"
    local COMPOSE="$DIR/docker-compose.yml"

    cat > "$COMPOSE" << 'HEADER'
# Auto-generated by spark-config snapshot
# This file captures the complete container stack configuration.
# Usage: docker compose -f docker-compose.yml up -d

services:
HEADER

    for CJSON in "$DIR"/containers/*.json; do
        [ -f "$CJSON" ] || continue
        local CNAME=$(basename "$CJSON" .json)

        # Skip infrastructure containers (veth, etc)
        [[ "$CNAME" == veth* ]] && continue

        python3 - "$CJSON" "$CNAME" >> "$COMPOSE" 2>/dev/null << 'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)[0]

name = sys.argv[2]
config = data["Config"]
host_config = data["HostConfig"]

image = config.get("Image", "unknown")
ports = []
for container_port, bindings in (host_config.get("PortBindings") or {}).items():
    if bindings:
        for b in bindings:
            hp = b.get("HostPort", "")
            if hp:
                ports.append(f'      - "{hp}:{container_port}"')

volumes = []
for m in data.get("Mounts", []):
    src = m.get("Source", m.get("Name", ""))
    dst = m.get("Destination", "")
    if src and dst:
        volumes.append(f'      - {src}:{dst}')

env_vars = []
# Filter out base image env vars (NVIDIA, CUDA, build-time vars)
SKIP_PREFIXES = (
    "PATH=", "JAVA_HOME=", "LANG=", "NVARCH=", "NVIDIA_REQUIRE_CUDA=",
    "NV_CUDA_", "CUDA_VERSION=", "LD_LIBRARY_PATH=", "NVIDIA_VISIBLE_",
    "NVIDIA_DRIVER_", "NV_NVTX_", "NV_LIBNPP_", "NV_LIBCUSPARSE_",
    "NV_LIBCUBLAS_", "NV_LIBNCCL_", "NCCL_VERSION=", "NVIDIA_PRODUCT_",
    "DEBIAN_FRONTEND=", "UV_HTTP_TIMEOUT=", "UV_INDEX_STRATEGY=",
    "UV_LINK_MODE=", "VLLM_ENABLE_CUDA_COMPATIBILITY=", "TORCH_CUDA_ARCH_LIST=",
    "VLLM_USAGE_SOURCE=", "NEO4J_SHA256=", "NEO4J_TARBALL=", "NEO4J_EDITION=",
    "NEO4J_HOME=", "ASAN_OPTIONS=", "ASAN_SYMBOLIZER_PATH=", "RUST_BACKTRACE=",
)
for e in (config.get("Env") or []):
    if not any(e.startswith(p) for p in SKIP_PREFIXES):
        env_vars.append(f'      - {e}')

cmd_parts = config.get("Cmd") or []
entrypoint = config.get("Entrypoint") or []

mem_limit = host_config.get("Memory", 0)
shm_size = host_config.get("ShmSize", 0)
ipc_mode = host_config.get("IpcMode", "")
restart = host_config.get("RestartPolicy", {}).get("Name", "")

# GPU config
device_requests = host_config.get("DeviceRequests") or []
needs_gpu = any(d.get("Driver") == "nvidia" or "gpu" in str(d.get("Capabilities", [])).lower() for d in device_requests)

print(f"  {name}:")
print(f"    image: {image}")

if entrypoint and cmd_parts:
    all_args = entrypoint + cmd_parts
    # Filter out the entrypoint binary itself
    if all_args:
        cmd_str = " ".join(f'"{a}"' if " " in a else a for a in cmd_parts)
        print(f"    command: >")
        for a in cmd_parts:
            print(f"      {a}")
elif cmd_parts:
    print(f"    command: >")
    for a in cmd_parts:
        print(f"      {a}")

if ports:
    print("    ports:")
    for p in ports:
        print(p)

if volumes:
    print("    volumes:")
    for v in volumes:
        print(v)

if env_vars:
    print("    environment:")
    for e in env_vars:
        print(e)

if restart:
    print(f"    restart: {restart}")

if needs_gpu:
    print("    deploy:")
    print("      resources:")
    print("        reservations:")
    print("          devices:")
    print("            - driver: nvidia")
    print("              count: all")
    print("              capabilities: [gpu]")

if ipc_mode == "host":
    print("    ipc: host")

if shm_size and shm_size > 0:
    shm_gb = shm_size / (1024**3)
    if shm_gb >= 1:
        print(f"    shm_size: '{int(shm_gb)}gb'")

if mem_limit and mem_limit > 0:
    mem_gb = mem_limit / (1024**3)
    print(f"    mem_limit: '{mem_gb:.1f}g'")

print()
PYEOF
    done
}

# ──────────────────────────────────────────────────────────────────────
# Capture model inventory
# ──────────────────────────────────────────────────────────────────────
_capture_models() {
    local DIR="$1"
    local MODELS_FILE="$DIR/models.json"

    python3 - > "$MODELS_FILE" 2>/dev/null << 'PYEOF'
import json, os, glob

models = []
hf_cache = "/home/<user>/.cache/huggingface/hub"
if os.path.exists(hf_cache):
    for d in os.listdir(hf_cache):
        if d.startswith("models--"):
            model_name = d.replace("models--", "").replace("--", "/")
            model_path = os.path.join(hf_cache, d)
            # Get size
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            models.append({
                "name": model_name,
                "cache_path": model_path,
                "size_gb": round(total_size / (1024**3), 2)
            })

# GLiNER has its own cache
gliner_cache = "/home/<user>/gliner-env/hf-cache/hub"
if os.path.exists(gliner_cache):
    for d in os.listdir(gliner_cache):
        if d.startswith("models--"):
            model_name = d.replace("models--", "").replace("--", "/")
            model_path = os.path.join(gliner_cache, d)
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            models.append({
                "name": model_name,
                "cache_path": model_path,
                "size_gb": round(total_size / (1024**3), 2),
                "note": "GLiNER cache"
            })

print(json.dumps({"models": models}, indent=2))
PYEOF
}

# ──────────────────────────────────────────────────────────────────────
# Generate restore script
# ──────────────────────────────────────────────────────────────────────
_generate_restore() {
    local DIR="$1"
    local NAME="$2"
    local RESTORE="$DIR/restore.sh"

    cat > "$RESTORE" << 'RESTEOF'
#!/bin/bash
# Restore script for spark config
# Generated by spark-config snapshot
# Usage: ./restore.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN="${1:-}"

run() {
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] $*"
    else
        echo "[RUNNING] $*"
        eval "$@"
    fi
}

echo "=== Spark Config Restore ==="
echo "Config: $(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/system.json')); print(d.get('description',''))")"
echo "Snapshot date: $(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/system.json')); print(d.get('snapshot_date',''))")"
echo ""

# Step 1: Apply sysctl tuning
echo "--- Step 1: Sysctl tuning ---"
if [ -f "$SCRIPT_DIR/sysctl.conf" ]; then
    run "sudo cp '$SCRIPT_DIR/sysctl.conf' /etc/sysctl.d/99-spark-tuning.conf"
    run "sudo sysctl -p /etc/sysctl.d/99-spark-tuning.conf"
fi

# Step 2: Apply network config
echo "--- Step 2: Network config ---"
for F in "$SCRIPT_DIR"/network/*.nmconnection; do
    [ -f "$F" ] || continue
    FNAME=$(basename "$F")
    run "sudo cp '$F' '/run/NetworkManager/system-connections/$FNAME'"
    run "sudo chmod 600 '/run/NetworkManager/system-connections/$FNAME'"
    run "sudo cp '$F' '/etc/NetworkManager/system-connections/$FNAME'"
    run "sudo chmod 600 '/etc/NetworkManager/system-connections/$FNAME'"
done
run "sudo systemctl reload NetworkManager"

# Step 3: Start containers via compose
echo "--- Step 3: Start containers ---"
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    run "docker compose -f '$SCRIPT_DIR/docker-compose.yml' up -d"
else
    echo "No docker-compose.yml found — start containers manually"
fi

echo ""
echo "=== Restore complete ==="
echo "Verify: docker ps && free -h && nvidia-smi"
RESTEOF

    chmod +x "$RESTORE"
}

# ──────────────────────────────────────────────────────────────────────
# LIST — Show saved configs
# ──────────────────────────────────────────────────────────────────────
cmd_list() {
    echo -e "${CYAN}Saved configurations:${NC}"
    echo ""
    printf "%-25s %-20s %s\n" "NAME" "DATE" "DESCRIPTION"
    printf "%-25s %-20s %s\n" "----" "----" "-----------"

    for DIR in "$CONFIG_DIR"/*/; do
        [ -d "$DIR" ] || continue
        local NAME=$(basename "$DIR")
        if [ -f "$DIR/system.json" ]; then
            local DATE=$(python3 -c "import json; d=json.load(open('$DIR/system.json')); print(d.get('snapshot_date','?')[:19])" 2>/dev/null || echo "?")
            local DESC=$(python3 -c "import json; d=json.load(open('$DIR/system.json')); print(d.get('description','')[:50])" 2>/dev/null || echo "")
            printf "%-25s %-20s %s\n" "$NAME" "$DATE" "$DESC"
        fi
    done
}

# ──────────────────────────────────────────────────────────────────────
# SHOW — Display config details
# ──────────────────────────────────────────────────────────────────────
cmd_show() {
    local NAME="${1:?Usage: spark-config show <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    echo -e "${CYAN}=== Config: $NAME ===${NC}"
    if [ -f "$DIR/system.json" ]; then
        python3 -c "
import json
with open('$DIR/system.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'  {k}: {v}')
"
    fi

    echo ""
    echo -e "${CYAN}Containers:${NC}"
    for C in "$DIR"/containers/*.json; do
        [ -f "$C" ] || continue
        local CN=$(basename "$C" .json)
        local IMG=$(python3 -c "import json; d=json.load(open('$C'))[0]; print(d['Config']['Image'])" 2>/dev/null)
        echo "  - $CN ($IMG)"
    done

    echo ""
    echo -e "${CYAN}Models:${NC}"
    if [ -f "$DIR/models.json" ]; then
        python3 -c "
import json
with open('$DIR/models.json') as f:
    d = json.load(f)
for m in d.get('models', []):
    print(f'  - {m[\"name\"]} ({m[\"size_gb\"]} GB)')
"
    fi

    echo ""
    echo -e "${CYAN}Files:${NC}"
    find "$DIR" -type f | sort | while read F; do
        echo "  $(echo "$F" | sed "s|$DIR/||")"
    done
}

# ──────────────────────────────────────────────────────────────────────
# DIFF — Compare current state vs saved config
# ──────────────────────────────────────────────────────────────────────
cmd_diff() {
    local NAME="${1:?Usage: spark-config diff <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    log "Comparing current state to config '$NAME'..."

    # Compare running containers vs saved
    echo -e "\n${CYAN}Container differences:${NC}"
    local RUNNING=$(docker ps --format '{{.Names}}' | sort)
    local SAVED=$(ls "$DIR/containers/" 2>/dev/null | sed 's/.json$//' | sort)

    local ONLY_RUNNING=$(comm -23 <(echo "$RUNNING") <(echo "$SAVED"))
    local ONLY_SAVED=$(comm -13 <(echo "$RUNNING") <(echo "$SAVED"))
    local COMMON=$(comm -12 <(echo "$RUNNING") <(echo "$SAVED"))

    [ -n "$ONLY_RUNNING" ] && echo -e "  ${GREEN}Running but not in config:${NC} $ONLY_RUNNING"
    [ -n "$ONLY_SAVED" ] && echo -e "  ${RED}In config but not running:${NC} $ONLY_SAVED"

    for C in $COMMON; do
        # Compare images
        local CUR_IMG=$(docker inspect "$C" --format '{{.Config.Image}}' 2>/dev/null)
        local SAV_IMG=$(python3 -c "import json; d=json.load(open('$DIR/containers/$C.json'))[0]; print(d['Config']['Image'])" 2>/dev/null)
        if [ "$CUR_IMG" != "$SAV_IMG" ]; then
            echo -e "  ${YELLOW}$C image changed:${NC} $SAV_IMG → $CUR_IMG"
        fi
    done

    # Compare sysctl
    echo -e "\n${CYAN}Sysctl differences:${NC}"
    if [ -f "$DIR/sysctl.conf" ]; then
        local CURRENT_SYSCTL=$(mktemp)
        sysctl -a 2>/dev/null | grep -E "^(vm\.(swappiness|min_free_kbytes|dirty_ratio)|net\.(core\.(rmem_max|wmem_max)|ipv4\.tcp_congestion_control))" | sort > "$CURRENT_SYSCTL"
        local SAVED_SYSCTL=$(sort "$DIR/sysctl.conf")
        diff <(echo "$SAVED_SYSCTL") "$CURRENT_SYSCTL" || echo "  (no differences)"
        rm -f "$CURRENT_SYSCTL"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# APPLY — Restore a saved config
# ──────────────────────────────────────────────────────────────────────
cmd_apply() {
    local NAME="${1:?Usage: spark-config apply <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    warn "This will stop all current containers and apply config '$NAME'."
    warn "Run '$DIR/restore.sh --dry-run' first to preview."
    echo ""
    read -p "Proceed? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi

    # Stop current containers
    log "Stopping current containers..."
    docker stop $(docker ps -q) 2>/dev/null || true

    # Run restore
    log "Applying config '$NAME'..."
    bash "$DIR/restore.sh"
}

# ──────────────────────────────────────────────────────────────────────
# BACKUP — Sync configs to homeserver
# ──────────────────────────────────────────────────────────────────────
cmd_backup() {
    log "Syncing configs to homeserver..."

    # Use SSH key if available
    local SSH_OPTS=""
    [ -f "$SSH_KEY" ] && SSH_OPTS="-e 'ssh -i $SSH_KEY'"

    rsync -avz --delete \
        ${SSH_OPTS} \
        "$CONFIG_DIR/" \
        "$HOMESERVER_BACKUP/" 2>&1

    log "Backup complete. Configs synced to homeserver."
}

# ──────────────────────────────────────────────────────────────────────
# RESTORE-FROM-BACKUP — Pull configs from homeserver (disaster recovery)
# ──────────────────────────────────────────────────────────────────────
cmd_restore_from_backup() {
    log "Pulling configs from homeserver..."

    local SSH_OPTS=""
    [ -f "$SSH_KEY" ] && SSH_OPTS="-e 'ssh -i $SSH_KEY'"

    mkdir -p "$CONFIG_DIR"
    rsync -avz \
        ${SSH_OPTS} \
        "$HOMESERVER_BACKUP/" \
        "$CONFIG_DIR/" 2>&1

    log "Configs restored from homeserver."
    cmd_list
}

# ──────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    snapshot)           shift; cmd_snapshot "$@" ;;
    list|ls)            cmd_list ;;
    show)               shift; cmd_show "$@" ;;
    diff)               shift; cmd_diff "$@" ;;
    apply)              shift; cmd_apply "$@" ;;
    backup)             cmd_backup ;;
    restore-from-backup) cmd_restore_from_backup ;;
    help|--help|-h)
        echo "spark-config — DGX Spark configuration management"
        echo ""
        echo "Commands:"
        echo "  snapshot <name> [desc]   Capture current running state"
        echo "  list                     List saved configs"
        echo "  show <name>              Show config details"
        echo "  diff <name>              Compare current vs saved"
        echo "  apply <name>             Apply a saved config"
        echo "  backup                   Sync all configs to homeserver"
        echo "  restore-from-backup      Pull configs from homeserver"
        ;;
    *)
        err "Unknown command: $1"
        exit 1
        ;;
esac
