import torch, re
from pathlib import Path

IN_PTH = Path("assets/models/mobilenetv2_finetuned.pth")
obj = torch.load(IN_PTH, map_location="cpu")
sd = obj.state_dict() if isinstance(obj, torch.nn.Module) else obj
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]

keys = list(sd.keys())
print(f"[INFO] Keys: {len(keys)} total")
for k in keys[:30]:
    print(" ", k)

# poszukaj czegokolwiek z 'classifier' w nazwie (różne prefiksy)
cands = [k for k in keys if re.search(r"(classifier|fc|head)", k)]
print("\n[HINT] Keys that look like head:", *cands[:30], sep="\n  ")
