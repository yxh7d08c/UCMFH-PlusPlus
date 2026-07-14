import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F

class UnimodalTransformer(nn.Module):
    """单模态Transformer编码器，用于学习图像或文本的内部特征关系"""
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super(UnimodalTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model] - 添加序列维度
        x = self.transformer(x)  # Transformer编码
        x = x.squeeze(1)  # [batch_size, d_model] - 移除序列维度
        return self.norm(x)


class CrossAttentionFusion(nn.Module):
    """双向Cross-Attention融合模块"""
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # Image-to-Text Cross-Attention (图像作为Query，从文本中获取信息)
        self.img2text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Text-to-Image Cross-Attention (文本作为Query，从图像中获取信息)
        self.text2img_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization (用于残差连接后的归一化)
        self.norm_img = LayerNorm(d_model)
        self.norm_text = LayerNorm(d_model)
        
        # Dropout for residual connection
        self.dropout = Dropout(dropout)
        
    def forward(self, img_feat, text_feat):
        """
        双向Cross-Attention交互
        Args:
            img_feat: [batch_size, d_model] 图像特征
            text_feat: [batch_size, d_model] 文本特征
        Returns:
            img_fused: [batch_size, d_model] 融合后的图像特征
            text_fused: [batch_size, d_model] 融合后的文本特征
        """
        # 添加序列维度以适配MultiheadAttention
        img = img_feat.unsqueeze(1)   # [batch, 1, d_model]
        text = text_feat.unsqueeze(1) # [batch, 1, d_model]
        
        # 第一个方向：图像作为Query，文本作为Key/Value
        # 让图像特征从文本中获取互补信息
        img_attended, _ = self.img2text_attn(
            query=img,
            key=text,
            value=text
        )  # [batch, 1, d_model]
        # 残差连接 + LayerNorm
        img_fused = self.norm_img(img + self.dropout(img_attended))
        
        # 第二个方向：文本作为Query，图像作为Key/Value
        # 让文本特征从图像中获取互补信息
        text_attended, _ = self.text2img_attn(
            query=text,
            key=img,
            value=img
        )  # [batch, 1, d_model]
        # 残差连接 + LayerNorm
        text_fused = self.norm_text(text + self.dropout(text_attended))
        
        # 移除序列维度
        img_fused = img_fused.squeeze(1)   # [batch, d_model]
        text_fused = text_fused.squeeze(1) # [batch, d_model]
        
        return img_fused, text_fused


class ImageMlp(nn.Module):
    """图像哈希映射网络"""
    def __init__(self, input_dim=512, hash_lens=64):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return normalize(x, p=2, dim=1)

class TextMlp(nn.Module):
    """文本哈希映射网络"""
    def __init__(self, input_dim=512, hash_lens=64):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return normalize(x, p=2, dim=1)


