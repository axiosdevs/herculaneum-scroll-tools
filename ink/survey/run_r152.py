"""Run the canonical 2 um ResNet3D-152 ink model on a layer stack."""
import os, sys, time, numpy as np, torch
# the villa loader, found the same way reproduce.py finds it: VILLA_INFERENCE if set, else
# fetched once into a cache -- so this runs from a fresh clone without a villa checkout
for _p in ("/workspace/ink", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
    if os.path.exists(os.path.join(_p, "inference_env.py")):
        sys.path.insert(0, _p)
        break
import inference_env
inference_env.ensure()
from model_resnet3d_3d_decoder import load_model

TILE = int(os.environ.get("TILE", "256")); STRIDE = int(os.environ.get("STRIDE", "128"))
CLIP = 200
_CACHE = {}
def get_net(ckpt, dev, frames):
    key = (ckpt, str(dev), frames)
    if key not in _CACHE:
        n = load_model(ckpt, dev, num_frames=frames); n.eval(); _CACHE[key] = n
    return _CACHE[key]

def _device(device=None):
    if device: return torch.device(device)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def predict(stack, ckpt="r152.ckpt", device=None, reverse=False, batch=None):
    """stack: (C, H, W) uint8 -> (H, W) float32 ink probability."""
    dev = _device(device)
    batch = batch or int(os.environ.get("BATCH", "8" if dev.type == "cuda" else "1"))
    net = get_net(ckpt, dev, stack.shape[0])
    C, H, W = stack.shape
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H-TILE, 0)+1, STRIDE)) or [0]
    xs = list(range(0, max(W-TILE, 0)+1, STRIDE)) or [0]
    win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32) + 1e-3
    t0 = time.time(); n = 0
    todo = []
    for y in ys:
        for x in xs:
            t = stack[:, y:y+TILE, x:x+TILE]
            if t.shape[1] != TILE or t.shape[2] != TILE: continue
            if (t != 0).any(0).mean() < 0.05: continue
            todo.append((y, x))
    amp = torch.autocast(dev.type, dtype=torch.float16, enabled=(dev.type == "cuda"))
    with torch.no_grad(), amp:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b+batch]
            arr = np.empty((len(chunk), stack.shape[0], TILE, TILE), np.float32)
            for k, (y, x) in enumerate(chunk):
                a = np.clip(stack[:, y:y+TILE, x:x+TILE], 0, CLIP).astype(np.float32)/CLIP
                arr[k] = a[::-1] if reverse else a
            out = net.forward(torch.from_numpy(arr)[:, None].to(dev))
            if isinstance(out, (list, tuple)): out = out[0]
            out = torch.sigmoid(out)
            if out.ndim == 4: out = out[:, 0]
            P = torch.nn.functional.interpolate(out[:, None], size=(TILE, TILE),
                                                mode="bilinear", align_corners=False)[:, 0].float().cpu().numpy()
            for k, (y, x) in enumerate(chunk):
                acc[y:y+TILE, x:x+TILE] += P[k]*win; cnt[y:y+TILE, x:x+TILE] += win
            n += len(chunk)
    print(f"  плиток {n} за {time.time()-t0:.0f}с", flush=True)
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
