"""Export a PPO checkpoint's policy network to the flat file the viewer loads.

    python3 python/export_policy.py checkpoints/ppo-XXXX-latest.pt models/name.pfp
    build/pf_play --p1 agent:models/name.pfp

Format (little-endian): "PFP1", int32 frame_skip, int32 layer count, then per
layer int32 in, int32 out, float32 weights (out x in, row-major), float32 bias.
Last come the logits for an all-0.1 observation, which the loader re-computes
to confirm its forward pass matches PyTorch's.
"""
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import Agent  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu")
    agent = Agent(ckpt["obs_size"], ckpt["hidden"])
    agent.load_state_dict(ckpt["model"])
    linears = [m for m in agent.policy if isinstance(m, torch.nn.Linear)]
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(b"PFP1")
        f.write(struct.pack("<ii", ckpt["args"]["frame_skip"], len(linears)))
        for layer in linears:
            f.write(struct.pack("<ii", layer.in_features, layer.out_features))
            f.write(layer.weight.detach().numpy().astype("<f4").tobytes())
            f.write(layer.bias.detach().numpy().astype("<f4").tobytes())
        with torch.no_grad():
            check = agent.policy(torch.full((1, ckpt["obs_size"]), 0.1))[0]
        f.write(check.numpy().astype("<f4").tobytes())
    print(f"wrote {dst}: {len(linears)} layers, obs {ckpt['obs_size']}, "
          f"frame skip {ckpt['args']['frame_skip']}, trained {ckpt['steps']:,} steps")


if __name__ == "__main__":
    main()
