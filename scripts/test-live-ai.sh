#!/usr/bin/env bash
set -euo pipefail

SCRIPTABLE_TOOLS=(claude copilot gemini codex cursor opencode)

export RUN_LIVE_AI_TESTS=1

if [[ -n "${CROSSBY_LIVE_AI_TOOLS:-}" ]]; then
    IFS=',' read -r -a requested_tools <<<"${CROSSBY_LIVE_AI_TOOLS}"
    validated_tools=()
    for raw_tool in "${requested_tools[@]}"; do
        tool="$(printf '%s' "$raw_tool" | xargs)"
        [[ -z "$tool" ]] && continue
        case " ${SCRIPTABLE_TOOLS[*]} " in
            *" $tool "*) validated_tools+=("$tool") ;;
            *)
                echo "Unknown live AI tool: $tool" >&2
                echo "Supported tools: ${SCRIPTABLE_TOOLS[*]}" >&2
                exit 1
                ;;
        esac
    done
    if [[ ${#validated_tools[@]} -eq 0 ]]; then
        echo "CROSSBY_LIVE_AI_TOOLS did not contain any supported tools." >&2
        exit 1
    fi
    selected_tools=""
    for tool in "${validated_tools[@]}"; do
        if [[ -n "$selected_tools" ]]; then
            selected_tools+=","
        fi
        selected_tools+="$tool"
    done
    export CROSSBY_LIVE_AI_TOOLS="$selected_tools"
else
    configured_tools=()
    for tool in "${SCRIPTABLE_TOOLS[@]}"; do
        upper_tool="$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
        env_key="CROSSBY_LIVE_MODEL_${upper_tool}"
        if [[ -n "${!env_key:-}" ]]; then
            configured_tools+=("$tool")
        fi
    done

    if [[ ${#configured_tools[@]} -eq 0 ]]; then
        echo "No live AI tools configured." >&2
        echo "Set CROSSBY_LIVE_AI_TOOLS or one or more CROSSBY_LIVE_MODEL_<TOOL> variables." >&2
        exit 1
    fi

    selected_tools=""
    for tool in "${configured_tools[@]}"; do
        if [[ -n "$selected_tools" ]]; then
            selected_tools+=","
        fi
        selected_tools+="$tool"
    done
    export CROSSBY_LIVE_AI_TOOLS="$selected_tools"
fi

if [[ $# -gt 0 ]]; then
    uv run pytest -m "live_ai" "$@"
else
    uv run pytest -m "live_ai" tests/live/
fi
