#!/usr/bin/env bash
# Launch a scenario × seed matrix with bounded parallelism and restart safety.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="${REPO_ROOT}/configs/experiments"
OUTPUT_ROOT="${REPO_ROOT}/outputs/swarm_dataset_v1"
SEED_SPEC="0:19"
JOBS="${JOBS:-4}"
SCENARIO_GLOB="*.yaml"
FORMATS=""
RUN_VALIDATION_LEVEL="standard"
CATALOG_VALIDATION_LEVEL="quick"
FORCE=0
DRY_RUN=0
NO_CATALOG=0
CONFIGS=()

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_experiments.sh [options]

Options:
  --config PATH            Add one scenario file (repeatable).
  --config-dir DIR         Directory scanned for YAML/JSON files.
  --scenario GLOB          Filter config basenames (default: *.yaml).
  --seeds SPEC             Inclusive range/list: 0:19, 40-59, or 1,4,9.
  --jobs N                 Maximum concurrent simulations (default: 4 or $JOBS).
  --output-root DIR        Dataset root.
  --python PATH            Python interpreter.
  --formats LIST           Override formats, e.g. parquet or parquet,jsonl.
  --validation-level NAME  Per-run validation: quick, standard, or full.
  --catalog-level NAME     Dataset catalog validation: quick, standard, or full.
  --force                  Replace completed runs instead of skipping them.
  --dry-run                Validate configs and print commands only.
  --no-catalog             Do not build the dataset catalog at the end.
  -h, --help               Show this message.

Bash 5+ is required because the launcher uses wait -n for its job pool.
Each successful run is atomically finalized under raw/<run_id>/ and receives a
_SUCCESS marker. The final catalog contains manifests, schemas, checksums,
DuckDB views, and dataset-level quality results.
USAGE
}

while (($#)); do
  case "$1" in
    --config) CONFIGS+=("$2"); shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --scenario) SCENARIO_GLOB="$2"; shift 2 ;;
    --seeds) SEED_SPEC="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --formats) FORMATS="$2"; shift 2 ;;
    --validation-level) RUN_VALIDATION_LEVEL="$2"; shift 2 ;;
    --catalog-level) CATALOG_VALIDATION_LEVEL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-catalog) NO_CATALOG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((BASH_VERSINFO[0] < 5)); then
  echo "Bash 5 or newer is required; found ${BASH_VERSION}." >&2
  exit 2
fi
if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi
case "${RUN_VALIDATION_LEVEL}" in quick|standard|full) ;; *) echo "Invalid --validation-level" >&2; exit 2 ;; esac
case "${CATALOG_VALIDATION_LEVEL}" in quick|standard|full) ;; *) echo "Invalid --catalog-level" >&2; exit 2 ;; esac

if ((${#CONFIGS[@]} == 0)); then
  if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Config directory not found: ${CONFIG_DIR}" >&2
    exit 2
  fi
  while IFS= read -r -d '' path; do
    basename_value="$(basename -- "${path}")"
    if [[ "${basename_value}" == ${SCENARIO_GLOB} ]]; then
      CONFIGS+=("${path}")
    fi
  done < <(find "${CONFIG_DIR}" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) -print0 | sort -z)
fi
if ((${#CONFIGS[@]} == 0)); then
  echo "No scenario configs matched." >&2
  exit 2
fi

mapfile -t SEEDS < <("${PYTHON_BIN}" - "${SEED_SPEC}" <<'PY'
import sys

spec = sys.argv[1]
values: set[int] = set()
for token in spec.split(','):
    token = token.strip()
    if not token:
        continue
    separator = ':' if ':' in token else '-' if '-' in token[1:] else None
    if separator is None:
        values.add(int(token))
        continue
    left, right = token.split(separator, 1)
    start, end = int(left), int(right)
    step = 1 if end >= start else -1
    values.update(range(start, end + step, step))
for value in sorted(values):
    if value < 0:
        raise SystemExit("seeds must be non-negative")
    print(value)
PY
)
if ((${#SEEDS[@]} == 0)); then
  echo "No seeds were produced from: ${SEED_SPEC}" >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Config not found: ${config}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m drone_swarm.cli validate-scenario --config "${config}" >/dev/null
done

mkdir -p "${OUTPUT_ROOT}/logs"
FAILURE_FILE="${OUTPUT_ROOT}/logs/failures.$$.tsv"
: > "${FAILURE_FILE}"

run_one() {
  local config="$1"
  local seed="$2"
  local stem
  stem="$(basename -- "${config}")"
  stem="${stem%.*}"
  local log_path="${OUTPUT_ROOT}/logs/${stem}__seed-${seed}.log"
  local command=(
    "${PYTHON_BIN}" -m drone_swarm.cli run
    --config "${config}"
    --seed "${seed}"
    --output-root "${OUTPUT_ROOT}"
    --validation-level "${RUN_VALIDATION_LEVEL}"
  )
  if ((FORCE)); then
    command+=(--force)
  else
    command+=(--resume)
  fi
  if [[ -n "${FORMATS}" ]]; then
    command+=(--formats "${FORMATS}")
  fi

  if ((DRY_RUN)); then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  printf '[start] %s seed=%s\n' "${stem}" "${seed}"
  local status=0
  set +e
  "${command[@]}" >"${log_path}" 2>&1
  status=$?
  set -e
  if ((status == 0)); then
    printf '[done ] %s seed=%s\n' "${stem}" "${seed}"
    return 0
  fi

  printf '[fail ] %s seed=%s (log: %s)\n' "${stem}" "${seed}" "${log_path}" >&2
  printf '%s\t%s\t%s\t%s\n' "${stem}" "${seed}" "${status}" "${log_path}" >>"${FAILURE_FILE}"
  return "${status}"
}

running=0
wait_for_one() {
  local status=0
  set +e
  wait -n
  status=$?
  set -e
  running=$((running - 1))
  return "${status}"
}

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${config}" "${seed}" &
    running=$((running + 1))
    if ((running >= JOBS)); then
      wait_for_one || true
    fi
  done
done
while ((running > 0)); do
  wait_for_one || true
done

catalog_status=0
if ((!DRY_RUN && !NO_CATALOG)); then
  set +e
  "${PYTHON_BIN}" -m drone_swarm.cli catalog \
    --output-root "${OUTPUT_ROOT}" \
    --strict \
    --validation-level "${CATALOG_VALIDATION_LEVEL}"
  catalog_status=$?
  set -e
fi

if [[ -s "${FAILURE_FILE}" ]] || ((catalog_status != 0)); then
  echo "[FAIL] Experiment matrix did not complete successfully." >&2
  if [[ -s "${FAILURE_FILE}" ]]; then
    echo "Run failure table: ${FAILURE_FILE}" >&2
    cat "${FAILURE_FILE}" >&2
  fi
  if ((catalog_status != 0)); then
    echo "Dataset catalog validation exit code: ${catalog_status}" >&2
  fi
  exit 1
fi
rm -f "${FAILURE_FILE}"
printf '[PASS] Completed %d scenario(s) × %d seed(s).\n' "${#CONFIGS[@]}" "${#SEEDS[@]}"
