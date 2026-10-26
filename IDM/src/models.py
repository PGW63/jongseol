"""IDM 모델 정의.

- DreamGenIDM: NVIDIA GR00T-Dreams(DreamGen)의 IDM 코드를 그대로 import 해서 사용.
  SigLIP2-large 비전 인코더 + Self-Attn(4층) + Cross-Attn DiT(8층) flow-matching 헤드.
  입력 전처리(GR00TIDMTransform)만 GPU 배치 연산으로 다시 구현했다.
- ResNetIDM: 주력 모델(경량). 프레임별 ResNet18 특징 + Transformer 디코더로 액션 청크 회귀.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
GR00T_ROOT = ROOT / "third_party/GR00T-Dreams"

SIGLIP_MEAN = 0.5
SIGLIP_STD = 0.5
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _import_gr00t_idm():
    """gr00t/__init__ 들이 GR00T-N1 전체(tensorflow, eagle 등)를 끌어오므로
    패키지 껍데기만 등록하고 IDM에 필요한 하위 모듈만 로드한다.
    DreamGen 의존성(transformers 등)은 `./install.sh --dreamgen` 으로만 설치되므로 여기서 import 한다."""
    if "gr00t.model.idm" in sys.modules:
        return sys.modules["gr00t.model.idm"]
    if not GR00T_ROOT.exists():
        raise RuntimeError("GR00T-Dreams 가 없습니다. DreamGen IDM 은 `./install.sh --dreamgen` 으로 설치하세요.")
    import transformers

    transformers.logging.set_verbosity_error()
    for name, sub in [
        ("gr00t", "gr00t"),
        ("gr00t.model", "gr00t/model"),
        ("gr00t.model.backbone", "gr00t/model/backbone"),
    ]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(GR00T_ROOT / sub)]
        sys.modules[name] = pkg
    import importlib

    ident = importlib.import_module("gr00t.model.backbone.identity")
    sys.modules["gr00t.model.backbone"].IdentityBackbone = ident.IdentityBackbone
    return importlib.import_module("gr00t.model.idm")


def augment(frames, train, out_size, crop_scale=0.95):
    """frames uint8 [B, T, S, S, 3] -> float [B*T, 3, out, out] in [0,1].
    GR00T so100 transform: VideoCrop(0.95) -> Resize -> ColorJitter. 같은 샘플의 두 프레임엔 같은 증강."""
    B, T, S = frames.shape[:3]
    x = frames.permute(0, 1, 4, 2, 3).float().div(255)  # B,T,3,S,S
    c = int(S * crop_scale)
    if train:
        oy = torch.randint(0, S - c + 1, (B,)).tolist()
        ox = torch.randint(0, S - c + 1, (B,)).tolist()
    else:
        oy = ox = [(S - c) // 2] * B
    x = torch.stack([x[b, :, :, oy[b] : oy[b] + c, ox[b] : ox[b] + c] for b in range(B)])
    x = F.interpolate(x.flatten(0, 1), size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.view(B, T, 3, out_size, out_size)
    if train:
        dev = x.device
        bright = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * 0.3
        contrast = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * 0.4
        sat = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * 0.5
        x = x * bright
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        x = (x - mean) * contrast + mean
        gray = x.mean(dim=2, keepdim=True)
        x = (x - gray) * sat + gray
        x = x.clamp(0, 1)
    return x.flatten(0, 1)


class DreamGenIDM(nn.Module):
    EMBODIMENT_ID = 26  # GR00TIDMTransform 의 "so100" projector 슬롯
    TOKENS_PER_FRAME = 16
    MAX_SEQ = 112
    MAX_ACTION_DIM = 32

    def __init__(self, action_dim=6, horizon=16, pretrained=None, siglip_path=None,
                 freeze_vision=False, grad_ckpt=False, single_embodiment=False, image_size=256):
        super().__init__()
        idm_mod = _import_gr00t_idm()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(GR00T_ROOT / "IDM_dump/base.yaml")
        siglip_cfg = cfg.config.action_head_cfg.config.siglip_model_cfg
        siglip_cfg.pretrained_model_name_or_path = str(siglip_path)
        # 공개 IDM_so100 체크포인트와 같은 구조 (base.yaml 은 6, 체크포인트는 3)
        cfg.config.action_head_cfg.config.max_num_views = 3
        self.idm = instantiate(cfg)
        if pretrained is not None:
            from safetensors.torch import load_file

            sd = load_file(str(Path(pretrained) / "model.safetensors"))
            missing, unexpected = self.idm.load_state_dict(sd, strict=False)
            print(f"[DreamGenIDM] loaded {pretrained}: missing={len(missing)} unexpected={len(unexpected)}")
        head = self.idm.action_head
        if single_embodiment:
            # CategorySpecificLinear 는 32개 embodiment 가중치를 들고 있지만 so100 슬롯 하나만 쓴다.
            # 해당 슬롯만 남겨도 forward 결과는 동일하고 파라미터/옵티마이저 상태가 ~135M 줄어든다.
            for mlp in [head.action_encoder.W1, head.action_encoder.W2, head.action_encoder.W3,
                        head.action_decoder.layer1, head.action_decoder.layer2]:
                e = self.EMBODIMENT_ID
                mlp.W = nn.Parameter(mlp.W.data[e : e + 1].clone())
                mlp.b = nn.Parameter(mlp.b.data[e : e + 1].clone())
                mlp.num_categories = 1
            self.EMBODIMENT_ID = 0
        if freeze_vision:
            head.siglip_model.vision_model.requires_grad_(False)
        if grad_ckpt:
            head.siglip_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.idm.compute_dtype = "bfloat16"
        self.action_dim, self.horizon, self.image_size = action_dim, horizon, image_size

    def _inputs(self, frames, train):
        B, T = frames.shape[:2]
        dev = frames.device
        img = augment(frames, train, self.image_size)
        img = (img - SIGLIP_MEAN) / SIGLIP_STD
        n_img = T * self.TOKENS_PER_FRAME
        vl = torch.zeros(B, self.MAX_SEQ, dtype=torch.long, device=dev)
        vl[:, -n_img:] = 1  # _IMG_TOKEN, left padding
        return {
            "images": img,
            "view_ids": torch.arange(T, device=dev).repeat(B),
            "vl_token_ids": vl,
            "vl_attn_mask": vl != 0,
            "sa_token_ids": torch.full((B, self.horizon), 4, dtype=torch.long, device=dev),
            "embodiment_id": torch.full((B,), self.EMBODIMENT_ID, dtype=torch.long, device=dev),
        }

    def _pad(self, action):
        B, H, D = action.shape
        a = F.pad(action, (0, self.MAX_ACTION_DIM - D))
        m = torch.zeros_like(a, dtype=torch.bool)
        m[..., :D] = True
        return a, m

    def loss(self, frames, action):
        from transformers.feature_extraction_utils import BatchFeature

        inp = self._inputs(frames, train=True)
        inp["actions"], inp["actions_mask"] = self._pad(action)
        out = self.idm.action_head(None, BatchFeature(data=inp))
        return out["loss"]

    @torch.no_grad()
    def predict(self, frames):
        from transformers.feature_extraction_utils import BatchFeature

        inp = self._inputs(frames, train=False)
        out = self.idm.action_head.get_action(None, BatchFeature(data=inp))
        return out["action_pred"][..., : self.action_dim].float()


class ResNetIDM(nn.Module):
    """경량 baseline: 프레임별 ResNet18(ImageNet) 7x7 특징맵 -> Transformer 인코더 ->
    액션 쿼리 디코더 -> [H, D] 회귀 (L1)."""

    def __init__(self, action_dim=6, horizon=16, d_model=256, image_size=224):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        self.proj = nn.Conv2d(512, d_model, 1)
        n_tok = (image_size // 32) ** 2
        self.pos = nn.Parameter(torch.randn(1, 2 * n_tok, d_model) * 0.02)
        self.frame_emb = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, 4 * d_model, 0.1, batch_first=True, norm_first=True), 4)
        self.queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 8, 4 * d_model, 0.1, batch_first=True, norm_first=True), 4)
        self.head = nn.Linear(d_model, action_dim)
        self.image_size, self.n_tok = image_size, n_tok

    def forward(self, frames, train):
        B = frames.shape[0]
        x = augment(frames, train, self.image_size)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        f = self.proj(self.backbone(x))  # [B*2, d, 7, 7]
        f = f.flatten(2).transpose(1, 2).view(B, 2, self.n_tok, -1)
        f = (f + self.frame_emb.unsqueeze(0)).flatten(1, 2) + self.pos
        mem = self.enc(f)
        return self.head(self.dec(self.queries.expand(B, -1, -1), mem))

    def loss(self, frames, action):
        return F.l1_loss(self.forward(frames, train=True).float(), action)

    @torch.no_grad()
    def predict(self, frames):
        return self.forward(frames, train=False).float()


def build_model(name, **kw):
    ckpt = ROOT / "checkpoints"
    if name == "dreamgen_pretrained":  # IDM_so100 에서 파인튜닝 (전체 파라미터)
        return DreamGenIDM(pretrained=ckpt / "IDM_so100", siglip_path=ckpt / "siglip2-large-vision", **kw)
    if name == "dreamgen_scratch":  # SigLIP2만 사전학습, 헤드는 랜덤 초기화 (전체 파라미터)
        return DreamGenIDM(pretrained=None, siglip_path=ckpt / "siglip2-large-vision", **kw)
    if name == "resnet":
        return ResNetIDM(**kw)
    raise ValueError(name)
