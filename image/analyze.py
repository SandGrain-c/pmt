"""Analysis for the PMT course project (run from the repo's image/ folder).

1. Complexity: parameters (total / frozen encoder / trainable decoder), GFLOPs and
   FPS (batch 1, 512x512, fp16) for every variant.
2. Accuracy details on ADE20K val for PMT-S (ours, 1 epoch) and PMT-L (official):
   per-class IoU, per-image mIoU, and figures of the worst / best images.

Usage:
  cd pmt/image
  python3 analyze.py --data /path/to/dir_with_zip --ckpt_s /path/to/runpmts.ckpt \
      --ckpt_l /path/to/pmt_l_ade20k.bin --out /kaggle/working/analysis
"""

import argparse, copy, csv, importlib, json, os, time, warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex

warnings.filterwarnings("ignore")

ADE_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]
assert len(ADE_CLASSES) == 150
IGNORE = 255
CFG_S = "configs/ade20k/semantic/pmt_s_ade20k_1ep.yaml"
CFG_L = "configs/ade20k/semantic/pmt_l.yaml"


def _cls(path):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def build(cfg_path, net_overrides=None, ckpt=None, device="cuda"):
    cfg = yaml.safe_load(open(cfg_path))
    img_size = tuple(cfg["data"]["init_args"].get("img_size", (512, 512)))
    num_classes = cfg["data"]["init_args"]["num_classes"]
    enc_cfg = cfg["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder = _cls(enc_cfg["class_path"])(img_size=img_size, **enc_cfg.get("init_args", {}))
    net_cfg = copy.deepcopy(cfg["model"]["init_args"]["network"])
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    net_kwargs.update(net_overrides or {})
    net_kwargs["masked_attn_enabled"] = False  # inference mode, as in the paper
    network = _cls(net_cfg["class_path"])(num_classes=num_classes, encoder=encoder, **net_kwargs)
    model_kwargs = {k: v for k, v in cfg["model"]["init_args"].items() if k != "network"}
    model = _cls(cfg["model"]["class_path"])(
        img_size=img_size, num_classes=num_classes, network=network, ckpt_path=ckpt, **model_kwargs
    )
    return model.eval().to(device), cfg


def complexity(model, device="cuda", iters=100):
    net = model.network
    enc = sum(p.numel() for p in net.encoder.parameters())
    total = sum(p.numel() for p in net.parameters())
    x = torch.randint(0, 255, (1, 3, *model.img_size), device=device).float()
    gflops = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 1e9
    except Exception as e:  # FLOP counting is best effort
        print("flop counter failed:", e)
    fps = float("nan")
    if device == "cuda":
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            for _ in range(20):
                model(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            fps = iters / (time.time() - t)
    return dict(params_total_M=total / 1e6, params_encoder_M=enc / 1e6,
                params_decoder_M=(total - enc) / 1e6, gflops=gflops, fps_bs1_fp16=fps)


@torch.no_grad()
def predict(model, img, device="cuda"):
    with autocast(dtype=torch.float16, device_type="cuda", enabled=device == "cuda"):
        imgs = [img.to(device)]
        sizes = [img.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_l, class_l = model(crops)
        mask = F.interpolate(mask_l[-1], model.img_size, mode="bilinear")
        crop_logits = model.to_per_pixel_logits_semantic(mask, class_l[-1])
        logits = model.revert_window_logits_semantic(crop_logits, origins, sizes)
    return logits[0].argmax(0).cpu()


def image_miou(pred, tgt, n=150):
    valid = tgt != IGNORE
    p, t = pred[valid], tgt[valid]
    ious = []
    for c in torch.unique(torch.cat([p, t])):
        inter = ((p == c) & (t == c)).sum().item()
        union = ((p == c) | (t == c)).sum().item()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def colorize(a):
    rng = np.random.RandomState(3)
    pal = rng.randint(0, 255, (256, 3)).astype(np.uint8)
    pal[IGNORE] = 0
    return pal[a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_s", required=True)
    ap.add_argument("--ckpt_l", required=True)
    ap.add_argument("--out", default="/kaggle/working/analysis")
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device

    # ---------- 1. complexity ----------
    variants = [
        ("baseline_nb0", CFG_S, {"num_blocks": 0}),
        ("pmt_s_nb2", CFG_S, {"num_blocks": 2}),
        ("pmt_s_nb6 (= no-mask at inference)", CFG_S, {}),
        ("pmt_l_official", CFG_L, {}),
    ]
    rows = []
    for name, cfg, ov in variants:
        m, _ = build(cfg, ov, device=dev)
        r = {"model": name, **complexity(m, dev)}
        print(json.dumps(r))
        rows.append(r)
        del m
        torch.cuda.empty_cache() if dev == "cuda" else None
    with open(f"{a.out}/complexity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # ---------- 2. per-class / per-image accuracy ----------
    data_cls = _cls("datasets.ade20k_semantic.ADE20KSemantic")
    data = data_cls(path=a.data, batch_size=1, num_workers=0, check_empty_targets=False).setup()
    ds = data.val_dataset
    n = min(a.max_val, len(ds))

    results = {}
    for tag, cfg, ck in [("pmt_s", CFG_S, a.ckpt_s), ("pmt_l", CFG_L, a.ckpt_l)]:
        model, _ = build(cfg, ckpt=ck, device=dev)
        metric = MulticlassJaccardIndex(num_classes=150, ignore_index=IGNORE, average=None)
        per_img = []
        for i in range(n):
            img, target = ds[i]
            tgt = model.to_per_pixel_targets_semantic([target], IGNORE)[0].cpu()
            pred = predict(model, img, dev)
            metric.update(pred[None], tgt[None])
            per_img.append(image_miou(pred, tgt))
            if i % 200 == 0:
                print(tag, i, "/", n, flush=True)
        iou = metric.compute().numpy()
        results[tag] = dict(iou=iou, per_img=np.array(per_img))
        print(f"{tag}: mIoU over {n} images = {np.nanmean(iou) * 100:.2f}")
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None

    with open(f"{a.out}/per_class_iou.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "iou_pmt_s", "iou_pmt_l", "gap"])
        for c in range(150):
            s, l = results["pmt_s"]["iou"][c], results["pmt_l"]["iou"][c]
            w.writerow([c, ADE_CLASSES[c], f"{s:.4f}", f"{l:.4f}", f"{l - s:.4f}"])
    np.savetxt(f"{a.out}/per_image_miou_pmt_s.txt", results["pmt_s"]["per_img"], fmt="%.4f")

    order = np.argsort(results["pmt_s"]["iou"])
    print("\nWorst 10 classes (PMT-S):", [(ADE_CLASSES[c], round(float(results['pmt_s']['iou'][c]) * 100, 1)) for c in order[:10]])
    print("Best 10 classes (PMT-S):", [(ADE_CLASSES[c], round(float(results['pmt_s']['iou'][c]) * 100, 1)) for c in order[::-1][:10]])

    # ---------- 3. qualitative figures: 4 worst + 2 best images for PMT-S ----------
    pi = results["pmt_s"]["per_img"]
    idx = list(np.argsort(np.nan_to_num(pi, nan=1.0))[:4]) + list(np.argsort(np.nan_to_num(pi, nan=0.0))[::-1][:2])
    ms, _ = build(CFG_S, ckpt=a.ckpt_s, device=dev)
    ml, _ = build(CFG_L, ckpt=a.ckpt_l, device=dev)
    fig, axes = plt.subplots(len(idx), 4, figsize=(16, 4 * len(idx)))
    for r, i in enumerate(idx):
        img, target = ds[int(i)]
        tgt = ms.to_per_pixel_targets_semantic([target], IGNORE)[0].cpu().numpy()
        ps, pl = predict(ms, img, dev).numpy(), predict(ml, img, dev).numpy()
        panels = [img.permute(1, 2, 0).numpy(), colorize(tgt), colorize(ps), colorize(pl)]
        titles = [f"val #{i}", "Ground truth", f"PMT-S 1ep (img mIoU {pi[i]*100:.1f})", "PMT-L official"]
        for c in range(4):
            axes[r, c].imshow(panels[c]); axes[r, c].set_title(titles[c]); axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(f"{a.out}/qualitative_worst4_best2.png", dpi=80)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
