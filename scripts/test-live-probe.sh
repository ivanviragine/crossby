#!/usr/bin/env bash
set -euo pipefail

SUPPORTED_TOOLS=(claude copilot gemini codex cursor opencode)

tool_binary() {
    case "$1" in
        claude|copilot|gemini|codex|opencode)
            printf '%s\n' "$1"
            ;;
        cursor)
            printf 'agent\n'
            ;;
        *)
            return 1
            ;;
    esac
}

export RUN_LIVE_PROBE_TESTS=1

if [[ -n "${CROSSBY_LIVE_PROBE_TOOLS:-}" ]]; then
    IFS=',' read -r -a requested_tools <<<"${CROSSBY_LIVE_PROBE_TOOLS}"
    validated_tools=()
    for raw_tool in "${requested_tools[@]}"; do
        tool="$(printf '%s' "$raw_tool" | xargs)"
        [[ -z "$tool" ]] && continue
        case " ${SUPPORTED_TOOLS[*]} " in
            *" $tool "*) validated_tools+=("$tool") ;;
            *)
                echo "Unknown live probe tool: $tool" >&2
                echo "Supported tools: ${SUPPORTED_TOOLS[*]}" >&2
                exit 1
                ;;
        esac
    done
    if [[ ${#validated_tools[@]} -eq 0 ]]; then
        echo "CROSSBY_LIVE_PROBE_TOOLS did not contain any supported tools." >&2
        exit 1
    fi
else
    validated_tools=()
    for tool in "${SUPPORTED_TOOLS[@]}"; do
        if command -v "$(tool_binary "$tool")" >/dev/null 2>&1; then
            validated_tools+=("$tool")
        fi
    done
    if [[ ${#validated_tools[@]} -eq 0 ]]; then
        echo "No probeable AI tools found in PATH." >&2
        echo "Install a supported CLI or set CROSSBY_LIVE_PROBE_TOOLS explicitly." >&2
        exit 1
    fi
fi

selected_tools=""
for tool in "${validated_tools[@]}"; do
    if [[ -n "$selected_tools" ]]; then
        selected_tools+=","
    fi
    selected_tools+="$tool"
done
export CROSSBY_LIVE_PROBE_TOOLS="$selected_tools"

if [[ $# -gt 0 ]]; then
    uv run pytest -m "live_probe" "$@"
else
    uv run pytest -m "live_probe" tests/live/
fi
