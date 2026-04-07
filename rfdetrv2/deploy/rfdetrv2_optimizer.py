# ------------------------------------------------------------------------
# Zero-shot Inference Optimization for RF-DETR
# Method: Dynamic Bi-Modal Query Partitioning (DBQP) via Otsu's Thresholding
# ------------------------------------------------------------------------

import math
import types
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Helpers & Math Functions
# ---------------------------------------------------------------------------

def compute_dynamic_keep_k(scores: torch.Tensor, min_queries: int = 50) -> int:
    """
    Thuật toán Otsu 1D: Tìm điểm cắt k tối đa hóa phương sai liên lớp (Inter-class variance).
    Phân tách chính xác nhóm Background (Nhiễu) và Foreground (Vật thể).
    """
    B, N = scores.shape
    
    # Sắp xếp điểm số tăng dần
    sorted_scores, _ = torch.sort(scores, dim=-1)
    
    # Tạo tensor k [1, 2, ..., N-1]
    k_tensor = torch.arange(1, N, device=scores.device).unsqueeze(0).expand(B, -1)
    
    # Trọng số của 2 class (background và foreground)
    w1 = k_tensor.float() / N
    w2 = 1.0 - w1
    
    # Tính tổng tích lũy để tìm giá trị trung bình nhanh chóng
    cum_sum = torch.cumsum(sorted_scores, dim=-1)
    total_sum = cum_sum[:, -1:]
    
    # Trung bình cộng (Mean) của 2 nhóm
    mu1 = cum_sum[:, :-1] / k_tensor.float()
    
    # Tránh chia cho 0 ở phần tử cuối cùng
    divisor2 = (N - k_tensor).float()
    divisor2 = torch.clamp(divisor2, min=1e-6)
    mu2 = (total_sum - cum_sum[:, :-1]) / divisor2
    
    # Phương sai liên lớp: w1 * w2 * (mu1 - mu2)^2
    sigma_b_sq = w1 * w2 * (mu1 - mu2) ** 2
    
    # Điểm k cắt tối ưu là nơi phương sai đạt max
    max_idx = torch.argmax(sigma_b_sq, dim=-1) # [B]
    
    # Số query cần giữ lại là phần bên phải của điểm cắt
    kept_counts = N - max_idx
    
    # Đảm bảo giữ lại số lượng tối thiểu để an toàn
    kept_counts = torch.clamp(kept_counts, min=min_queries)
    
    # Lấy K lớn nhất trong batch để giữ tensor vuông vức
    K_optimal = kept_counts.max().item()
    
    return K_optimal

def gather_dynamic(tensor: torch.Tensor, topk_idx: torch.Tensor):
    """ Hàm hỗ trợ để rút gọn Tensor theo Index động """
    if tensor is None:
        return None
    B, K = topk_idx.shape
    
    # Đảm bảo index có cùng số chiều với tensor cần gather
    expand_shape = [-1, -1] + list(tensor.shape[2:])
    idx_expanded = topk_idx.view(B, K, *([1]*(tensor.dim()-2))).expand(*expand_shape)
    
    return torch.gather(tensor, 1, idx_expanded)

def gen_sineembed_for_position(pos_tensor, dim=128):
    """ Copy nguyên gốc từ transformer helper để dùng nội bộ trong monkey-patch """
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    return pos


# ---------------------------------------------------------------------------
# 2. Core Injection: Monkey Patching Decoder Forward
# ---------------------------------------------------------------------------

def optimized_forward(self, tgt, memory,
                      tgt_mask=None, memory_mask=None,
                      tgt_key_padding_mask=None, memory_key_padding_mask=None,
                      pos=None, refpoints_unsigmoid=None, level_start_index=None,
                      spatial_shapes=None, valid_ratios=None):
    
    output = tgt
    intermediate = []
    hs_refpoints_unsigmoid = [refpoints_unsigmoid]

    def get_reference(refpoints):
        obj_center = refpoints[..., :4]
        if self._export:
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
            refpoints_input  = obj_center[:, :, None]
        else:
            refpoints_input  = obj_center[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            query_sine_embed = gen_sineembed_for_position(
                refpoints_input[:, :, 0, :], self.d_model / 2)
        query_pos = self.ref_point_head(query_sine_embed)
        return obj_center, refpoints_input, query_pos, query_sine_embed

    if self.lite_refpoint_refine:
        if self.bbox_reparam:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
        else:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

    for layer_id, layer in enumerate(self.layers):
        
        # ====================================================================
        # [DBQP LOGIC]: Dynamic Pruning during Inference
        # ====================================================================
        if not self.training and layer_id == self.start_pruning_layer and output.shape[1] > self.min_queries:
            # 1. Dự đoán logit bằng class_embed layer (vay mượn từ mô hình gốc)
            temp_logits = self.class_embed_ref(output) # [B, N, num_classes]
            
            # 2. Lấy Objectness score cao nhất (bỏ qua background class ở index cuối nếu có)
            scores = temp_logits.max(dim=-1)[0] # [B, N]
            
            # 3. Tính ngưỡng K tối ưu bằng phương sai Otsu
            K_keep = compute_dynamic_keep_k(scores, min_queries=self.min_queries)
            
            if K_keep < output.shape[1]: # Chỉ cắt nếu có queries để cắt
                # Lấy Top-K index
                _, topk_idx = torch.topk(scores, K_keep, dim=-1) # [B, K_keep]
                
                # --- PRUNE TENSORS HIỆN TẠI ---
                output = gather_dynamic(output, topk_idx)
                refpoints_unsigmoid = gather_dynamic(refpoints_unsigmoid, topk_idx)
                
                if self.lite_refpoint_refine:
                    obj_center = gather_dynamic(obj_center, topk_idx)
                    refpoints_input = gather_dynamic(refpoints_input, topk_idx)
                    query_pos = gather_dynamic(query_pos, topk_idx)
                    query_sine_embed = gather_dynamic(query_sine_embed, topk_idx)
                
                # --- ĐỒNG BỘ HÓA HISTORY (CỰC KỲ QUAN TRỌNG) ---
                # Vì 'intermediate' và 'hs_refpoints_unsigmoid' chứa các output của layer trước,
                # Nếu không cắt chúng theo index hiện tại, lệnh `torch.stack` cuối hàm sẽ crash.
                intermediate = [gather_dynamic(t, topk_idx) for t in intermediate]
                hs_refpoints_unsigmoid = [gather_dynamic(t, topk_idx) for t in hs_refpoints_unsigmoid]
        # ====================================================================

        if not self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        query_pos = query_pos * 1

        output = layer(
            output, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=(layer_id == 0),
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            ref_boxes=obj_center,
        )

        if not self.lite_refpoint_refine:
            new_refpoints_delta     = self.bbox_embed(output)
            new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
            if layer_id != self.num_layers - 1:
                hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
            refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

        if self.return_intermediate:
            intermediate.append(self.norm(output))

    if self.norm is not None:
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

    if self.return_intermediate:
        if self._export:
            hs  = intermediate[-1]
            ref = hs_refpoints_unsigmoid[-1] if hasattr(self, 'bbox_embed') and self.bbox_embed is not None else refpoints_unsigmoid
            return hs, ref
        if hasattr(self, 'bbox_embed') and self.bbox_embed is not None:
            return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)]
        else:
            return [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

    return output.unsqueeze(0)


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------

def optimize_model_for_inference(model: nn.Module, start_pruning_layer: int = 2, min_queries: int = 50):
    """
    Tiêm (monkey-patch) thuật toán DBQP vào mô hình RF-DETR.
    Chỉ dùng cho INFERENCE. KHÔNG DÙNG KHI TRAINING.
    
    Args:
        model: Đối tượng LWDETR / RF-DETR của bạn.
        start_pruning_layer: Layer bắt đầu thực hiện tỉa query. Mặc định là 2 (tức là sau layer 0 và 1).
                             Để càng sớm (ví dụ 1) tốc độ càng nhanh nhưng có thể rủi ro giảm mAP nhẹ.
        min_queries: Số lượng query tối thiểu cần giữ lại (Safe guard).
    """
    print(f"🚀 [DBQP Optimizer] Đang tối ưu hóa mô hình RF-DETR cho Inference...")
    
    decoder = model.transformer.decoder
    
    # Truyền tham số và reference của linear layer xuống cho decoder
    decoder.class_embed_ref = model.class_embed
    decoder.start_pruning_layer = start_pruning_layer
    decoder.min_queries = min_queries
    
    # Ghi đè phương thức forward (Monkey Patching)
    decoder.forward = types.MethodType(optimized_forward, decoder)
    
    print(f"✅ [DBQP Optimizer] Hoàn tất! Sẽ loại bỏ query thừa từ Decoder Layer thứ {start_pruning_layer}.")
    print(f"   (Đảm bảo model.eval() đã được gọi khi chạy inference).")