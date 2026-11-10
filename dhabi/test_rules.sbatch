#!/bin/bash
# =====================================================================
# 1-minute test of the security rules. Starts no model and opens only one
# tiny test port on 127.0.0.1, to show that the rule-3 check sees it.
# Run from the repo folder on Dhabi:   sbatch dhabi/test_rules.sbatch
# Then read the log:                   cat test_rules_<job number>.log
# =====================================================================
#SBATCH --job-name=test_rules
#SBATCH --nodelist=t01pdscgpu18
#SBATCH --reservation=security      # the security team's reservation of node 18
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --output=test_rules_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
source dhabi/check_rules.sh                     # rules 1 and 2

echo "GPUs this job may use:"
nvidia-smi -L | awk -v v="$CUDA_VISIBLE_DEVICES" 'BEGIN{print "  (CUDA_VISIBLE_DEVICES=" v ")"} {print "  " $0}'

# one tiny test server on localhost, picked free by the system, then rule 3
python3 -c '
import http.server, socketserver
s = socketserver.TCPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
print("test port", s.server_address[1], flush=True); s.timeout = 20; s.handle_request()
' &
sleep 3
check_localhost_only                            # rule 3
kill %1 2>/dev/null
[ "$CHECKED_PORTS" -ge 1 ] || { echo "TEST FAILED: the rule-3 check did not see the test port"; exit 1; }
echo "TEST PASSED: all three rules checked"
