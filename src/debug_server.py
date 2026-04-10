"""Diagnostic script to pinpoint where server.main() hangs."""
import sys
print("Step 1: Importing server module...", flush=True)
try:
    from agent_banana import server
    print("Step 2: Import succeeded.", flush=True)
except Exception as e:
    print(f"Step 2: Import FAILED: {e}", flush=True)
    sys.exit(1)

print("Step 3: Calling main()...", flush=True)
try:
    server.main()
except Exception as e:
    print(f"Step 4: main() FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
