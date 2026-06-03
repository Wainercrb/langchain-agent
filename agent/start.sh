#!/bin/bash
set -e

# Runs both the FastAPI server and the APScheduler cronjob in the same
# container. Bash stays as PID 1 so it can forward SIGTERM to both
# children on `docker stop` — otherwise the cronjob would be killed
# unceremoniously when the container's PID namespace is torn down.
#
# Layout:
#   PID 1  bash (start.sh)
#   ├─ child  cronjob.py  (background, scheduler + maintenance jobs)
#   └─ child  server.py   (background, FastAPI)
#
# On SIGTERM/SIGINT, both children get SIGTERM with a 10s grace window
# before SIGKILL. If either child exits unexpectedly, we tear down the
# other and exit, so the container is restarted by `restart: unless-stopped`.

# echo "[start.sh] Launching cronjob in background..."
python cronjob.py &
CRON_PID=$!

echo "[start.sh] Launching FastAPI server in background..."
python server.py &
SERVER_PID=$!

echo "[start.sh] cronjob PID=$CRON_PID  server PID=$SERVER_PID"

shutdown() {
    echo "[start.sh] Received signal, sending SIGTERM to both children..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$CRON_PID" 2>/dev/null || true

    # Wait up to 10s for both children to exit gracefully
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null && ! kill -0 "$CRON_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    # Force-kill anything still running
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    kill -KILL "$CRON_PID" 2>/dev/null || true
    wait 2>/dev/null
    exit 0
}
trap shutdown SIGTERM SIGINT

# Block until either child exits. If one dies unexpectedly, we tear
# down the other so the container is restarted and the dead process
# is replaced (instead of running half a system).
wait -n

EXIT_CODE=$?
echo "[start.sh] A child process exited with code $EXIT_CODE"
shutdown
