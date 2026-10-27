import json, subprocess, tempfile, glob, os
from http.server import BaseHTTPRequestHandler, HTTPServer

SUBMISSION = "/workspace/predictions.csv"
SCORE_LOG = "/logs/verifier/validation_scores.jsonl"
COMPETITION = "{comp_id}"

# Prepare the competition data ONCE at server start (mlebench prepare is heavy
# and must not re-run per submission; concurrent prepares spawn runaway procs).
_PREPARED = False
def _ensure_prepared() -> None:
    global _PREPARED
    if not _PREPARED:
        subprocess.run(["mlebench", "prepare", "-c", COMPETITION], check=True, capture_output=True)
        _PREPARED = True

def grade() -> float:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", "/root/.kaggle")
    _ensure_prepared()
    with tempfile.TemporaryDirectory() as work:
        subs = f"{work}/subs.jsonl"
        with open(subs, "w") as f:
            f.write(json.dumps({"competition_id": COMPETITION, "submission_path": SUBMISSION}) + "\n")
        r = subprocess.run(["mlebench", "grade", subs, "--output-dir", f"{work}/out"], capture_output=True, text=True)
        g = glob.glob(f"{work}/out/*grading_report*.json")
        if g:
            rep = json.load(open(g[0]))
            comps = rep.get("competition_reports") or []
            if comps:
                return float(comps[0].get("score"))
        raise RuntimeError(f"grade produced no score; rc={r.returncode} {r.stdout[-200:]!r}")

def record(score):
    os.makedirs(os.path.dirname(SCORE_LOG), exist_ok=True)
    with open(SCORE_LOG, "a") as f:
        f.write(json.dumps({"submission_score": score}) + "\n")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(body).encode())
    def do_POST(self):
        try:
            s = grade(); record(s); self._send(200, {"score": s})
        except Exception as e:
            self._send(500, {"error": str(e)})
    do_GET = do_POST
    def log_message(self, *a): pass

HTTPServer(("0.0.0.0", 5000), H).serve_forever()
