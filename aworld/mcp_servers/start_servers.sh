#!/bin/bash

handle_interrupt() {
    echo "Caught SIGINT (Ctrl+C). Executing cleanup commands..."
    echo "Shutting down localhost ports..."
    lsof -i :2000-2001 | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9
    echo "Cleanup done."
    exit 1
}

# trap SIGINT (Ctrl+C) and call handle_interrupt function
trap handle_interrupt SIGINT

# include the list of servers
server_pids=()

python launcher.py --port 2000 --sse-path "/sse" &
LAUNCHER_PID=$!
server_pids+=($LAUNCHER_PID)
echo "Started Aworld MCP Server with PID $LAUNCHER_PID on port 2000 with SSE path /sse"

npx @playwright/mcp@latest --port 2001 &
PLAYWRIGHT_PID=$!
server_pids+=($PLAYWRIGHT_PID)
echo "Started Playwright instance with PID $PLAYWRIGHT_PID on port 2001"

# wait for all servers to finish
wait "${server_pids[@]}"
echo "All servers have been stopped."