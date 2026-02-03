#!/bin/bash
set -e  # Exit on first failure

echo "=== Act! Demo Mode Verification ==="
echo ""

# 1. Unit tests for act_fetch
echo "[1/5] Running unit tests..."
pytest tests/backend/test_act_fetch.py -v --tb=short
echo "✓ Unit tests passed"
echo ""

# 2. Backend eval with latency check
echo "[2/5] Running Act! eval..."
python -m backend.eval.act
echo "✓ Act! eval passed"
echo ""

# 3. Full CI (linting, type checks, all tests)
echo "[3/5] Running full CI..."
./scripts/ci.sh
echo "✓ CI passed"
echo ""

# 4. Start backend in demo mode for E2E
echo "[4/5] Starting demo backend..."
export ACME_DEMO_MODE=true
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 5  # Wait for startup

# 5. Playwright E2E tests
echo "[5/5] Running Playwright E2E tests..."
cd frontend
npm run test:e2e -- demo-mode.spec.ts --reporter=list
E2E_EXIT=$?
cd ..

# Cleanup
kill $BACKEND_PID 2>/dev/null || true

if [ $E2E_EXIT -ne 0 ]; then
    echo ""
    echo "============================================"
    echo "✗ E2E tests failed"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "✓ ALL CHECKS PASSED - Ready to send to Ken!"
echo "============================================"
