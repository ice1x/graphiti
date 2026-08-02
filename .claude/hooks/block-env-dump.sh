#!/usr/bin/env bash
# PreToolUse guard for the Bash tool: block commands that dump environment
# variables wholesale. Such dumps splash secrets (API keys, tokens) into the
# session transcript, terminal scrollback, and the model context. Reading a
# single named variable is fine; dumping the whole environment is not.
#
# Reads the hook payload JSON on stdin, denies with a PreToolUse decision when a
# dump pattern matches, otherwise stays silent and allows the command.
set -euo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

deny() {
  jq -n --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

# 1) `docker inspect` template that projects the container environment.
if printf '%s' "$cmd" | grep -Eq '\.Config\.Env'; then
  deny "Blocked: this 'docker inspect' template exposes .Config.Env, which dumps the container's whole environment (API keys/tokens leak into the transcript). Query only the fields you need, e.g. --format '{{.Config.Image}} {{range .Mounts}}{{.Source}}->{{.Destination}} {{end}} {{range \$p,\$c:=.NetworkSettings.Ports}}{{\$p}}={{\$c}} {{end}}'. To confirm a secret is set, test it — never print its value."
fi

# 2) Reading a process environment via /proc/<pid>/environ (self, a pid, glob, or $var).
if printf '%s' "$cmd" | grep -Eq '/proc/([0-9]+|self|\*|\$[A-Za-z_{])[^[:space:]]*/environ'; then
  deny "Blocked: reading /proc/<pid>/environ dumps a process's full environment (secret leak). If you need one variable, read it by name instead."
fi

# 3) `export -p` lists every exported variable together with its value.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|])[[:space:]]*export[[:space:]]+-p([[:space:]]|$)'; then
  deny "Blocked: 'export -p' lists every exported variable with its value (secret leak). Fetch a single variable by name instead: printenv NAME."
fi

# 4) A bare `set` prints all shell variables. `set -e`, `set -o pipefail`, etc. are fine.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|])[[:space:]]*set[[:space:]]*($|[|])'; then
  deny "Blocked: a bare 'set' prints every shell variable (secret leak). Use 'set' with explicit options (set -e, set -o pipefail), or fetch a single variable by name."
fi

# 5) `printenv` with no argument dumps the entire environment. `printenv NAME` is fine.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|])[[:space:]]*printenv[[:space:]]*($|[|>])'; then
  deny "Blocked: bare 'printenv' dumps the entire environment (secret leak). Pass a variable name: printenv NAME."
fi

# 6) A bare `env` (or `env | ...`) prints the entire environment. `env NAME=val cmd` is fine.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|])[[:space:]]*env[[:space:]]*($|[|>])'; then
  deny "Blocked: bare 'env' prints the entire environment (secret leak). To set a var for one command use 'env NAME=val cmd'; to read one var use 'printenv NAME'."
fi

# 7) `docker exec/run ... env|printenv` used as a dump of the container environment.
if printf '%s' "$cmd" | grep -Eq 'docker[[:space:]]+(exec|run)([[:space:]]|$)'; then
  if printf '%s' "$cmd" | grep -Eq '(printenv[[:space:]]*($|[|>&])|[[:space:]]env[[:space:]]*($|[|>&]))'; then
    deny "Blocked: 'docker exec/run ... env|printenv' dumps the container environment (secret leak). Inspect only the specific variable you need, and never print its value."
  fi
fi

# No dump pattern matched — allow silently.
exit 0
