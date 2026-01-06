# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()
        # --- small-object heatmap + guided token (default off) ---
        self.so_enable = False
        self.so_topk = 64  # K: 64/100 常用
        self.so_heatmap_level = 1  # 用 P4 做 heatmap（x=[P3,P4,P5] -> index 1）
        self.so_sample_level = 0  # 从 P3 采样 token（index 0）
        self.so_small_area = 0.02  # 小目标阈值(归一化面积 w*h)，你后面要调
        self.so_hm_loss_gain = 0.2  # heatmap loss 权重
        self.so_anchor_grid = 0.05  # 对齐 _generate_anchors 的 grid_size :contentReference[oaicite:4]{index=4}

        # heatmap head: (B, hd, H, W) -> (B,1,H,W) logits
        self.so_hm_head = nn.Sequential(
            nn.Conv2d(hd, hd, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hd, 1, 1, bias=True),
        )

        def enable_small_object_guidance(self, enable=True, topk=64, heatmap_level=1, sample_level=0,
                                         small_area=0.02, hm_loss_gain=0.2):
            self.so_enable = bool(enable)
            self.so_topk = int(topk)
            self.so_heatmap_level = int(heatmap_level)
            self.so_sample_level = int(sample_level)
            self.so_small_area = float(small_area)
            self.so_hm_loss_gain = float(hm_loss_gain)

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        if self.so_enable and (not self.export):  # 导出时建议关掉（topk + grid_sample 常见不友好）
            feats, shapes, proj = self._get_encoder_input(x, return_proj=True)
            hm_logits = self.so_hm_head(proj[self.so_heatmap_level])
            so_tokens, so_anchors = self._so_sample_tokens(
                hm_logits=hm_logits,
                feat_hi=proj[self.so_sample_level],
                topk=self.so_topk,
                level_index=self.so_sample_level
            )
        else:
            feats, shapes = self._get_encoder_input(x)
            hm_logits, so_tokens, so_anchors = None, None, None
        # feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        #修改
        if self.training and (hm_logits is not None) and (dn_meta is not None) and (batch is not None):
            H, W = hm_logits.shape[-2:]
            hm_tgt = self._build_so_heatmap_target(batch, H, W, device=hm_logits.device, dtype=hm_logits.dtype)
            dn_meta['so_hm_logits'] = hm_logits
            dn_meta['so_hm_targets'] = hm_tgt
            dn_meta['so_hm_loss_gain'] = self.so_hm_loss_gain

        # embed, refer_bbox, enc_bboxes, enc_scores = \
        #     self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox, so_tokens=so_tokens, so_anchors=so_anchors)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _build_so_heatmap_target(self, batch, H, W, device, dtype):
        # batch comes from tasks.py targets: {'cls','bboxes','batch_idx','gt_groups'} :contentReference[oaicite:7]{index=7}
        hm = torch.zeros((len(batch['gt_groups']), 1, H, W), device=device, dtype=dtype)
        if batch is None or 'bboxes' not in batch or batch['bboxes'].numel() == 0:
            return hm

        bboxes = batch['bboxes']  # (N,4) xywh normalized
        bidx = batch['batch_idx'].long()  # (N,)
        area = bboxes[:, 2] * bboxes[:, 3]
        keep = area < self.so_small_area
        if keep.sum() == 0:
            return hm

        bboxes = bboxes[keep]
        bidx = bidx[keep]
        # center to feature coords
        px = (bboxes[:, 0] * W).long().clamp_(0, W - 1)
        py = (bboxes[:, 1] * H).long().clamp_(0, H - 1)
        hm[bidx, 0, py, px] = 1.0
        return hm

    def _so_sample_tokens(self, hm_logits, feat_hi, topk, level_index):
        # hm_logits: (B,1,Hlo,Wlo) ; feat_hi: (B,hd,Hhi,Whi)
        B, _, Hlo, Wlo = hm_logits.shape
        _, C, Hhi, Whi = feat_hi.shape
        K = min(int(topk), Hlo * Wlo)
        if K <= 0:
            return None, None

        hm_prob = hm_logits.sigmoid().flatten(1)  # (B, Hlo*Wlo)
        topk_idx = torch.topk(hm_prob, K, dim=1).indices  # (B,K)

        ys = (topk_idx // Wlo).to(torch.float32)  # (B,K)
        xs = (topk_idx % Wlo).to(torch.float32)  # (B,K)

        # normalized [0,1] at P4 pixel centers
        u = (xs + 0.5) / float(Wlo)
        v = (ys + 0.5) / float(Hlo)

        # grid_sample expects [-1,1]
        grid_x = u * 2.0 - 1.0
        grid_y = v * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(B, K, 1, 2)  # (B,K,1,2)

        sampled = F.grid_sample(feat_hi, grid, mode='bilinear', align_corners=False)  # (B,C,K,1)
        tokens = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B,K,C)

        # anchors for these guided queries (logit space like _generate_anchors) :contentReference[oaicite:8]{index=8}
        wh = torch.full((B, K, 2), self.so_anchor_grid * (2.0 ** level_index),
                        device=feat_hi.device, dtype=feat_hi.dtype)
        a = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1), wh], dim=-1).clamp_(1e-4, 1 - 1e-4)  # (B,K,4)
        anchors_logit = torch.log(a / (1 - a))
        return tokens, anchors_logit

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x, return_proj=False):
        x = [self.input_proj[i](feat) for i, feat in
             enumerate(x)]  # projected maps :contentReference[oaicite:6]{index=6}
        feats, shapes = [], []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return (feats, shapes, x) if return_proj else (feats, shapes)

    # def _get_encoder_input(self, x):
    #     """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
    #     # Get projection features
    #     x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
    #     # Get encoder inputs
    #     feats = []
    #     shapes = []
    #     for feat in x:
    #         h, w = feat.shape[2:]
    #         # [b, c, h, w] -> [b, h*w, c]
    #         feats.append(feat.flatten(2).permute(0, 2, 1))
    #         # [nl, 2]
    #         shapes.append([h, w])
    #
    #     # [b, h*w, c]
    #     feats = torch.cat(feats, 1)
    #     return feats, shapes

    # def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, so_tokens=None, so_anchors=None):

        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # # (bs, num_queries)
        # topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # # (bs, num_queries)
        # batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        #
        # # (bs, num_queries, 256)
        # top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # # (bs, num_queries, 4)
        # top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)
        #
        # # Dynamic anchors + static content
        # refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        #
        # enc_bboxes = refer_bbox.sigmoid()
        # if dn_bbox is not None:
        #     refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        # enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        #
        # embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        # if self.training:
        #     refer_bbox = refer_bbox.detach()
        #     if not self.learnt_init_query:
        #         embeddings = embeddings.detach()
        # if dn_embed is not None:
        #     embeddings = torch.cat([dn_embed, embeddings], 1)
        #
        # return embeddings, refer_bbox, enc_bboxes, enc_scores
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc) :contentReference[oaicite:13]{index=13}

        bs = len(feats)
        K = 0 if (so_tokens is None or so_anchors is None) else so_tokens.shape[1]
        K = min(K, self.num_queries)
        main_nq = self.num_queries - K

        # (bs, main_nq)
        if main_nq > 0:
            topk_ind = torch.topk(enc_outputs_scores.max(-1).values, main_nq, dim=1).indices.view(-1)
            batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=feats.device).unsqueeze(-1).repeat(1,
                                                                                                             main_nq).view(
                -1)

            top_k_features = features[batch_ind, topk_ind].view(bs, main_nq, -1)
            top_k_anchors = anchors[:, topk_ind].view(bs, main_nq, -1)
            enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, main_nq, -1)
        else:
            top_k_features = features.new_zeros((bs, 0, features.shape[-1]))
            top_k_anchors = anchors.new_zeros((bs, 0, 4))
            enc_scores = enc_outputs_scores.new_zeros((bs, 0, enc_outputs_scores.shape[-1]))

        # append guided K queries
        if K > 0:
            so_features = self.enc_output(
                so_tokens)  # (bs,K,hd) uses same encoder head :contentReference[oaicite:14]{index=14}
            so_scores = self.enc_score_head(so_features)  # (bs,K,nc)

            top_k_features = torch.cat([top_k_features, so_features], dim=1)  # -> (bs,300,hd)
            top_k_anchors = torch.cat([top_k_anchors, so_anchors], dim=1)  # -> (bs,300,4)
            enc_scores = torch.cat([enc_scores, so_scores], dim=1)  # -> (bs,300,nc)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)

        # embeddings: if learnt_init_query=True, replace tail K with guided features
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.learnt_init_query and K > 0:
            embeddings = embeddings.clone()
            embeddings[:, -K:] = top_k_features[:, -K:]

        # keep original detach behavior :contentReference[oaicite:15]{index=15}
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
