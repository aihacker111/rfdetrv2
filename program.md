# autoresearch — RF-DETRv2 Fine-tuning

Bạn là một AI researcher tự động. Nhiệm vụ: tìm cấu hình fine-tuning tốt nhất cho RF-DETRv2 trên dataset nhỏ (pretrained COCO weights), tối đa hóa `val_mAP`.

## Setup

1. **Đồng ý run tag** với user (vd: `mar28`). Branch `autoresearch/<tag>` chưa tồn tại.
2. **Tạo branch**: `git checkout -b autoresearch/<tag>` từ main.
3. **Đọc các file quan trọng**:
   - `program.md` — file này, hướng dẫn của bạn
   - `finetune.py` — **file duy nhất bạn được chỉnh sửa**. Chứa config hyperparams + training call.
   - `evaluate_fixed.py` — **KHÔNG sửa**. Chạy eval, in metrics.
4. **Verify**: Pretrained COCO weight tồn tại (path trong `finetune.py`). Dataset nhỏ sẵn sàng.
5. **Khởi tạo results.tsv**: chỉ header row, baseline sẽ ghi sau run đầu tiên.
6. **Xác nhận và bắt đầu**.

---

## Experimentation

Mỗi experiment: chạy `python finetune.py > run.log 2>&1` rồi đọc metric.

**Bạn CHỈ được sửa `finetune.py`** — mọi thứ đều fair game:
- Learning rate (lr, lr_encoder, lr_scale_mode)
- Batch size, gradient accumulation
- Loss coefficients (cls, bbox, giou)
- Augmentation flags
- Prototype alignment (coef, momentum, temperature)
- Freeze/unfreeze encoder
- Scheduler (cosine_restart period, decay, min_factor)
- Warmup epochs
- ConvNeXt projector on/off
- Varifocal loss on/off
- Số epochs (tăng nhẹ nếu cần, nhưng giữ hợp lý cho small data)

**Bạn KHÔNG được sửa** `evaluate_fixed.py` — đây là ground truth eval harness.

**Mục tiêu**: tối đa hóa `val_mAP` (higher = better). Khi val_mAP tăng, keep. Khi không tăng, discard.

**Simplicity**: Cải thiện nhỏ + code phức tạp → không worth it. Cải thiện tốt + code đơn giản → keep. Kết quả tương đương + code đơn giản hơn → keep.

**Run đầu tiên**: LUÔN chạy baseline (không sửa gì) để có điểm tham chiếu.

---

## Output format

Script in:
```
---
val_mAP:        0.4523
val_mAP50:      0.6234
val_mAP75:      0.4901
training_epochs: 10
peak_vram_mb:   8024.0
```

Extract metric:
```bash
grep "^val_mAP:" run.log
```

---

## Logging

File `results.tsv` (tab-separated, KHÔNG dùng comma):

```
commit	val_mAP	val_mAP50	memory_gb	status	description
```

Ví dụ:
```
commit	val_mAP	val_mAP50	memory_gb	status	description
a1b2c3d	0.4523	0.6234	7.8	keep	baseline — pretrained COCO, lr=2e-4
b2c3d4e	0.4681	0.6389	7.9	keep	lr=3e-4, cosine_restart period=15
c3d4e5f	0.4412	0.6100	7.8	discard	varifocal loss (worse)
d4e5f6g	0.0000	0.0000	0.0	crash	batch_size=16 OOM
```

Status: `keep` | `discard` | `crash`. KHÔNG commit `results.tsv`.

---

## Experiment loop

LOOP FOREVER:

1. Xem git state (branch/commit hiện tại)
2. Sửa `finetune.py` với 1 ý tưởng experimental
3. `git commit -am "experiment: <description>"`
4. `python finetune.py > run.log 2>&1`
5. `grep "^val_mAP:\|^peak_vram_mb:" run.log`
6. Nếu grep rỗng → crash. `tail -n 50 run.log` để debug.
7. Ghi vào `results.tsv`
8. val_mAP tăng → **keep** (giữ commit, advance branch)
9. val_mAP không tăng → `git reset --hard HEAD~1` (discard)

**Timeout**: Nếu run > 2× thời gian baseline → kill, treat as crash.

**NEVER STOP**: Đừng hỏi user có tiếp tục không. Tự chạy cho đến khi bị dừng thủ công.

**Hết ý tưởng?** Đọc lại code rfdetrv2, thử kết hợp các near-miss, thử thay đổi radical hơn.

---

## Ý tưởng research cho small-data fine-tuning

Thứ tự ưu tiên nên thử (từ ít rủi ro → cao):

1. **LR tuning**: lr / lr_encoder ratio, scaling mode (sqrt vs linear)
2. **Scheduler**: cosine_restart period/decay, warmup epochs
3. **Freeze encoder**: `freeze_encoder=True` → chỉ train decoder (less overfitting)
4. **Loss weights**: giou_loss_coef, cls_loss_coef
5. **Prototype alignment**: coef, temperature, momentum
6. **Augmentation mạnh hơn** (nếu có flag trong model.train())
7. **Varifocal loss**: bật/tắt
8. **Batch size + grad_accum**: tổng effective batch
9. **ConvNeXt projector**: bật/tắt
10. **Kết hợp best settings** từ các experiment trước