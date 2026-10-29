#!/usr/bin/env python3
"""Convert a ComfyUI int8_convrot checkpoint into plain bf16 safetensors.

Run it with ComfyUI's own interpreter, because the dequantisation is done by
ComfyUI's own layout kernels (comfy_kitchen, reached through comfy.quant_ops).
The result is therefore what ComfyUI computes at runtime, not an approximation
of it.

  /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python \
      convert_text_encoder.py --self-test     # one tensor, timed
  ... convert_text_encoder.py                 # the whole file
"""
import argparse
import inspect
import json
import os
import sys
import time

COMFY = "/mnt/data_2t_3/github_repos/ComfyUI"
DEFAULT_SRC = os.path.join(COMFY, "models/text_encoders/moss_image_sft/qwen3.5_9b_int8_convrot.safetensors")
LAYOUT = "TensorWiseINT8Layout"
# positional/rotary buffers must stay float32; everything a module owns becomes bf16
KEEP_F32 = ("inv_freq", "rotary")


def build_params(scale, shape, conf, device):
    from comfy.quant_ops import TensorWiseINT8Layout
    fields = set(inspect.signature(TensorWiseINT8Layout.Params.__init__).parameters)
    kwargs = {"scale": scale.to(device), "orig_dtype": torch.bfloat16, "orig_shape": tuple(shape)}
    for key, value in (("convrot", bool(conf.get("convrot", False))),
                       ("convrot_groupsize", int(conf.get("convrot_groupsize", 256)))):
        if key in fields:
            kwargs[key] = value
    return TensorWiseINT8Layout.Params(**kwargs)


def dequantise(name, weight, scale, conf, device):
    """One quantised Linear weight -> dense bf16, using ComfyUI's kernel."""
    from comfy.quant_ops import QuantizedTensor
    params = build_params(scale, weight.shape, conf, device)
    qt = QuantizedTensor(weight.to(device=device, dtype=torch.int8), LAYOUT, params)
    out = qt.dequantize()
    keep = any(k in name for k in KEEP_F32)
    return out.to("cpu", torch.float32 if keep else torch.bfloat16).contiguous()


def read_conf(tensor):
    return json.loads(tensor.detach().to("cpu").numpy().tobytes())


def self_test(sd, device):
    from safetensors.torch import safe_open
    module = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    key = module + ".weight"
    conf = read_conf(sd[module + ".comfy_quant"])
    print("tensor:", key, tuple(sd[key].shape), sd[key].dtype)
    print("descriptor:", conf)
    for dev in (["cpu", device] if device != "cpu" else ["cpu"]):
        t0 = time.time()
        out = dequantise(key, sd[key], sd[module + ".weight_scale"], conf, dev)
        print(f"  dequant on {dev}: {tuple(out.shape)} {out.dtype} in {time.time() - t0:.2f}s "
              f"| absmean {out.float().abs().mean():.5f} std {out.float().std():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=None, help="default: <src dir>/qwen3.5_9b_bf16.safetensors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, COMFY)
    global torch
    import torch
    from safetensors.torch import load_file, save_file

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {device} | reading {args.src}")
    t0 = time.time()
    sd = load_file(args.src)
    print(f"  {len(sd)} tensors in {time.time() - t0:.1f}s")

    if args.self_test:
        self_test(sd, device)
        return

    dst = args.dst or os.path.join(os.path.dirname(args.src), "qwen3.5_9b_bf16.safetensors")
    out, consumed, done = {}, set(), 0
    t0 = time.time()
    for key, value in list(sd.items()):
        if not key.endswith(".comfy_quant"):
            continue
        base = key[: -len(".comfy_quant")]
        conf = read_conf(value)
        if conf.get("format") != "int8_tensorwise":
            raise SystemExit(f"{base}: unexpected quantisation format {conf.get('format')}")
        weight_key, scale_key = base + ".weight", base + ".weight_scale"
        out[weight_key] = dequantise(weight_key, sd[weight_key], sd[scale_key], conf, device)
        consumed.update([key, weight_key, scale_key])
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done} layers dequantised, {el:.0f}s elapsed", flush=True)

    for key, value in sd.items():
        if key in consumed:
            continue
        if value.is_floating_point() and not any(k in key for k in KEEP_F32):
            value = value.to(torch.bfloat16)
        out[key] = value.contiguous() if value.is_contiguous() is False else value

    print(f"dequantised {done} layers in {time.time() - t0:.0f}s; writing {len(out)} tensors -> {dst}")
    t0 = time.time()
    save_file(out, dst, metadata={"format": "pt", "converted_from": os.path.basename(args.src),
                                 "converted_by": "comfy_kitchen TensorWiseINT8Layout"})
    print(f"wrote {os.path.getsize(dst) / 1e9:.2f} GB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
