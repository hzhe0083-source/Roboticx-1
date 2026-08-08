"""Build a synthetic genuine-pair smoke dataset from mw_subset_smoke3.pt.

Each pair shares the exact same state (vision/proprio/previous_action) but
uses the language + action chunk of two different tasks.  Code-path smoke
only; the real fork data comes from mw fork collection (todo #5).
"""
import torch

d = torch.load("data/mw_subset_smoke3.pt", map_location="cpu", weights_only=True)
B = d["vision_tokens"].shape[0]
iid = d["instruction_id"]

out = {}
for k in ("vision_tokens", "proprio", "previous_action", "actions",
          "language_hidden", "language_mask", "instruction_id", "episode_id"):
    out[k] = []
pairs = []
for i in range(B):
    j = (i + 1) % B
    while iid[j].item() == iid[i].item():
        j = (j + 1) % B
    for src, dst in ((i, i), (i, j)):
        out["vision_tokens"].append(d["vision_tokens"][src])
        out["proprio"].append(d["proprio"][src])
        out["previous_action"].append(d["previous_action"][src])
        out["actions"].append(d["actions"][dst])
        out["language_hidden"].append(d["language_hidden"][dst])
        out["language_mask"].append(d["language_mask"][dst])
        out["instruction_id"].append(d["instruction_id"][dst])
        out["episode_id"].append(d["episode_id"][dst])
    pairs += [i, i]
out["pair_id"] = torch.tensor(pairs)

out = {k: torch.stack(v) for k, v in out.items() if isinstance(v, list)}
out["pair_id"] = torch.tensor(pairs)
out["normalization"] = d["normalization"]
out["metadata"] = d["metadata"]
torch.save(out, "/tmp/smoke_pairs.pt")
print(f"built /tmp/smoke_pairs.pt: {len(pairs)} samples, {B} pairs; "
      f"same-state ok: {torch.allclose(out['vision_tokens'][0], out['vision_tokens'][1])}, "
      f"diff-task ok: {out['instruction_id'][0].item() != out['instruction_id'][1].item()}")
