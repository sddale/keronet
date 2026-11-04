"""Check every HPO study in the manifest against hpo/search_spaces.py (parameters, choices, suggestion order).

Also prints each study's convergence status; exits 1 on any search-space mismatch.
Usage: uv run tools/verify_search_spaces.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hpo.search_spaces import SEARCH_SPACES  # noqa: E402
from tools._common import REPO, base_arch, convergence, db_space, load_manifest  # noqa: E402

bad = 0
for e in load_manifest():
    if "hpo_dir" not in e:
        continue
    arch = base_arch(e["arch"])
    db = REPO / e["hpo_dir"] / "study.db"
    want = [(n, list(c)) for n, c in SEARCH_SPACES[arch][e["space"]]]
    got = db_space(db)
    ok = got == want
    bad += not ok
    n, best, since = convergence(db)
    conv = "converged" if since >= 2 else "NOT converged"
    print(f"{'OK ' if ok else 'BAD'} {e['cell']:32s} {e['arch']:20s} space={e['space']:8s} "
          f"completed={n:2d} best=#{best} ({conv})  evaluated=#{e.get('evaluated_trial')}")
    if not ok:
        print(f"    declared: {want}\n    in DB:    {got}")
print(f"\n{bad} search-space mismatch(es)")
sys.exit(1 if bad else 0)
