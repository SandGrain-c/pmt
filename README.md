# PMT trên ADE20K: tái lập và thực nghiệm (Bài tập lớn môn Học máy nâng cao)

Repo này là fork của [tue-mps/pmt](https://github.com/tue-mps/pmt), mã nguồn chính thức của bài báo
**PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders** (CVPR Workshops 2026, [arXiv:2603.25398](https://arxiv.org/abs/2603.25398)).
README gốc của tác giả được giữ ở [README_ORIGINAL.md](README_ORIGINAL.md).

Trong bài tập lớn này, chúng tôi:
1. Tái lập kết quả của checkpoint PMT-L chính thức trên ADE20K.
2. Tự huấn luyện PMT-S (encoder DINOv3 ViT-S/16 đóng băng) với lịch rút gọn trên GPU T4.
3. So sánh với một baseline và 2 thí nghiệm thay đổi thành phần.
4. Phân tích độ phức tạp, IoU theo lớp và các trường hợp lỗi.

## Kết quả (ADE20K val, 2 000 ảnh, mIoU %)

| Mô hình | Thay đổi | Tham số decoder | GFLOPs | FPS (T4, bs1, fp16) | mIoU |
|---|---|---|---|---|---|
| PMT-L (checkpoint chính thức) | ViT-L, lịch train đầy đủ | 53.9M (+303M encoder) | 899.7 | 12.3 | **58.51** (paper: 58.5) |
| PMT-S | 6 block, masked attention | 7.66M | 96.3 | 54.1 | **24.79** |
| Thí nghiệm A | 2 block | 4.10M | 80.5 | 67.7 | 19.70 |
| Thí nghiệm B | không masked attention / annealing | 7.66M | 96.3 | 54.1 | 21.52 |
| Baseline | 0 block (bỏ PMD) | 2.33M | 72.6 | 77.0 | 6.16 |

Mọi mô hình PMT-S được huấn luyện với cùng ngân sách: 1 epoch (2 524 bước), batch 8, một GPU T4, seed 0.
Vì lịch train ngắn hơn khoảng 16 lần so với paper, giá trị tuyệt đối thấp, nhưng phép so sánh giữa các cấu hình là công bằng.
Phân tích chi tiết nằm trong báo cáo.

## Các file đã thêm hoặc thay đổi

| File | Nội dung |
|---|---|
| `image/configs/ade20k/semantic/pmt_s_ade20k_1ep.yaml` | Config PMT-S cho ADE20K (1 epoch, lịch mask annealing thu nhỏ theo tỉ lệ 2524/20208) |
| `run_experiments.py` | Chạy toàn bộ 4 lần train và 5 lần đánh giá trên 2 GPU song song |
| `image/analyze.py` | Tính tham số, GFLOPs, FPS; IoU theo lớp; hình failure cases |
| `results/` | Log, `complexity.csv`, `per_class_iou.csv`, `qualitative_worst4_best2.png` |

## Hướng dẫn chạy lại

**1. Môi trường** (đã thử trên Kaggle với 2× Tesla T4, Python 3.13):
```bash
git clone https://github.com/SandGrain-c/pmt.git && cd pmt
pip install -r requirements.txt
export HF_TOKEN=<token HuggingFace đã được duyệt quyền truy cập facebook/dinov3-vits16-pretrain-lvd1689m và dinov3-vitl16>
export WANDB_MODE=offline
```

**2. Dữ liệu và checkpoint** (không cần giải nén file zip):
```bash
mkdir -p data ckpt
wget -c -P data http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget -O ckpt/pmt_l_ade20k.bin https://huggingface.co/tue-mps/ade20k_semantic_pmt_large_512_dinov3/resolve/main/pytorch_model.bin
```

**3. Tái lập PMT-L** (khoảng 6 phút trên T4):
```bash
python3 image/main.py validate -c image/configs/ade20k/semantic/pmt_l.yaml \
  --model.network.masked_attn_enabled False --trainer.devices 1 --data.batch_size 1 \
  --data.path data --model.ckpt_path ckpt/pmt_l_ade20k.bin
```

**4. Toàn bộ thí nghiệm** (khoảng 70 phút trên 2× T4):
```bash
python3 run_experiments.py --data $(pwd)/data --ckpt_l $(pwd)/ckpt/pmt_l_ade20k.bin --logs logs
```
Có thể chạy riêng từng thí nghiệm như sau:
```bash
python3 image/main.py fit -c image/configs/ade20k/semantic/pmt_s_ade20k_1ep.yaml --compile_disabled \
  --trainer.devices 1 --data.batch_size 8 --data.path data \
  [--model.network.num_blocks 0|2] \
  [--model.network.masked_attn_enabled False --model.attn_mask_annealing_enabled False]
```

**5. Phân tích** (khoảng 20 phút):
```bash
cd image
python3 analyze.py --data ../data --ckpt_s <đường dẫn .ckpt của PMT-S> --ckpt_l ../ckpt/pmt_l_ade20k.bin --out ../results
```

**Lưu ý khi chạy trên Kaggle:**
- Bật Internet và GPU T4 x2.
- Đặt `HF_TOKEN` trong *Add-ons → Secrets*.
- Không đặt `WANDB_API_KEY` khi dùng chế độ offline.
- Dùng *Save & Run All (Commit)* để notebook chạy nền mà không bị ngắt.

## Trích dẫn
```bibtex
@inproceedings{cavagnero2026pmt,
  author    = {Cavagnero, Niccol\`{o} and Norouzi, Narges and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders}},
  booktitle = {CVPR Workshops},
  year      = {2026},
}
```
Mã nguồn gốc dùng giấy phép MIT, © Mobile Perception Systems Lab, TU/e.
