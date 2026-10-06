"""Make a Harbor job resumable again after a resume (Harbor bug, harbor c178c20).

Harbor saves each trial's agent env with secrets hidden: "****" on a fresh run but
"[REDACTED]" on a resumed one. The job's plan keeps "****", so the next `harbor job
resume` cannot match the resumed trials and fails with "Existing trial config does
not match planned job config". This rewrites OPENAI_API_KEY "[REDACTED]" -> the value
in the job's own config.json, in every trial's config.json and lock.json.

    python3 scripts/eval/fix_redacted_key.py jobs/<job-name>   (run.sh does this before every resume)
"""
import json, sys
from pathlib import Path

job = Path(sys.argv[1])
if not (job / "config.json").is_file():
    sys.exit(0)  # nothing to resume yet
planned = json.loads((job / "config.json").read_text())["agents"][0]["env"].get("OPENAI_API_KEY")
fixed = 0
for path in list(job.glob("*__*/config.json")) + list(job.glob("*__*/lock.json")):
    data = json.loads(path.read_text())
    env = data.get("agent", {}).get("env", {})
    if env.get("OPENAI_API_KEY") == "[REDACTED]" and planned is not None:
        env["OPENAI_API_KEY"] = planned
        path.write_text(json.dumps(data, indent=4 if path.name == "config.json" else None))
        fixed += 1
print(f"[fix_redacted_key] {job.name}: {fixed} trial files set to the planned OPENAI_API_KEY")
