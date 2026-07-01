#!/usr/bin/env bash
# Install the Hermes Forge plugin into Hermes Agent.
set -euo pipefail

PLUGIN_SRC="$(cd "$(dirname "$0")/../src/hermes_forge_plugin" && pwd)"
PLUGIN_DEST="${HERMES_PLUGINS:-$HOME/.hermes/plugins/hermes-forge}"

echo "Installing hermes-forge plugin to $PLUGIN_DEST ..."
mkdir -p "$(dirname "$PLUGIN_DEST")"
cp -r "$PLUGIN_SRC" "$PLUGIN_DEST"

if command -v hermes &>/dev/null; then
  hermes plugins enable hermes-forge 2>/dev/null && echo "✓ Plugin enabled in Hermes" || echo "ℹ Run 'hermes plugins enable hermes-forge' to activate"
  echo "ℹ Plugin takes effect on next session (hermes reset to reload now)"
else
  echo "ℹ hermes CLI not found. After installing Hermes, run:"
  echo "   hermes plugins enable hermes-forge"
fi

echo "✓ hermes-forge plugin installed"
echo "ℹ Proxy mode (optional):"
echo "   hermes-forge proxy --backend-url <your-llm-url> --port 8081"
echo "   Then set: hermes config set model.base_url http://localhost:8081/v1"
