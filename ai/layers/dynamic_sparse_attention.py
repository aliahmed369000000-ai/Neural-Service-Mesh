
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSparseAttention(nn.Module):
    """🚀 خوارزمية الانتباه الانتقائي الديناميكي: ابتكار سيادي لتقليل التعقيد الحسابي."""
    
    def __init__(self, d_model, n_heads, sparsity_k=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_k = sparsity_k
        self.d_head = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # حساب الانتباه الكثيف الأولي
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # تطبيق التفرقة الديناميكية (Dynamic Sparsity)
        # نختار فقط أفضل k من الروابط لكل رمز
        k_elements = max(1, int(seq_len * self.sparsity_k))
        topk_values, _ = torch.topk(scores, k_elements, dim=-1)
        min_score = topk_values[..., -1].unsqueeze(-1)
        
        # حجب الروابط الضعيفة
        sparse_scores = torch.where(scores >= min_score, scores, torch.tensor(-1e9).to(x.device))
        
        attn = F.softmax(sparse_scores, dim=-1)
        context = torch.matmul(attn, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(context)
