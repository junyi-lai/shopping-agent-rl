#!/usr/bin/env bash
set -euo pipefail

SHOP_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${SHOP_SEARCH_INDEX:-${SHOP_ENV_ROOT}/search_engine/products.sqlite3}"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "ShopSimulator index is missing: ${INDEX_PATH}" >&2
  echo "Run from the repository root: python scripts/00_environment.py setup" >&2
  exit 1
fi

export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOP_ENV_CONFIG="${SHOP_ENV_CONFIG:-${SHOP_ENV_ROOT}/configs/environment.json}"
export SHOP_SEARCH_INDEX="${INDEX_PATH}"
export SHOP_MAX_STEPS="${SHOP_MAX_STEPS:-35}"
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-8}"
export SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"

cd "${SHOP_ENV_ROOT}/shop_env"
exec python pack_api.py
