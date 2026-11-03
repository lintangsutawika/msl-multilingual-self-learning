import mlebench.data as d, pathlib, re
# Replace the whole interactive Kaggle rule-acceptance function with a no-op so
# neither input() prompt (the "y/n" one nor the "Press Enter..." one) can hit
# EOF in a headless verifier / validate server. Without this, `mlebench prepare`
# exits non-zero and `set -euo pipefail` aborts the verifier before reward.txt.
p = pathlib.Path(d.__file__)
t = p.read_text()
new = re.sub(
    r"def _prompt_user_to_accept_rules\(competition_id: str\) -> None:.*?(?=\ndef |\Z)",
    "def _prompt_user_to_accept_rules(competition_id: str) -> None:\n    pass\n",
    t, flags=re.S,
)
open(p, "w").write(new)
print("neutralized kaggle rule-acceptance prompts")
