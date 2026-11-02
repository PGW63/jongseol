"""TripoSG 로 사진 한 장 → mesh(GLB, 텍스처 없음). RTX 4060 Ti 8GB 에서 돌도록 메모리를 아낀다.

  env -u PYTHONPATH .venv-triposg/bin/python src/image_to_3d/run_triposg.py assets/sf3d_inputs/blueCup.jpeg
    -> third_party/triposg_out/blueCup.glb

원본 scripts/inference_triposg.py 는 배경 제거 모델, 이미지 인코더, transformer 를 모두 GPU 에 올린 채로
마지막 mesh 추출(505^3 격자)까지 가서 8GB 를 넘긴다 (CUDA illegal memory access).
여기서는 배경 제거 모델은 쓰고 바로 내리고, 나머지는 diffusers CPU offload 로 필요한 것만 GPU 에 올린다.
simulation/ 에서 실행. ROS 의 PYTHONPATH 에 있는 `scripts` 패키지와 이름이 겹치므로 PYTHONPATH 를 비운다.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download

SIM = Path(__file__).resolve().parents[2]
TRIPOSG = SIM / "third_party" / "TripoSG"
sys.path[:0] = [str(TRIPOSG), str(TRIPOSG / "scripts")]

from briarmbg import BriaRMBG  # noqa: E402
from image_process import prepare_image  # noqa: E402
from inference_triposg import simplify_mesh  # noqa: E402
from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--out", help="기본: third_party/triposg_out/<이미지 이름>.glb")
parser.add_argument("--faces", type=int, default=10000, help="면 개수 상한 (-1 이면 줄이지 않음)")
parser.add_argument("--octree-depth", type=int, default=9, help="mesh 추출 해상도 2^depth. 8 로 낮추면 메모리/시간 절약")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--guidance", type=float, default=7.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
out = Path(args.out) if args.out else SIM / "third_party" / "triposg_out" / f"{Path(args.image).stem}.glb"
out.parent.mkdir(parents=True, exist_ok=True)

weights = TRIPOSG / "pretrained_weights"
snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=weights / "TripoSG")
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=weights / "RMBG-1.4")

rmbg = BriaRMBG.from_pretrained(weights / "RMBG-1.4").to("cuda").eval()
with torch.no_grad():
    image = prepare_image(args.image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg)
del rmbg
gc.collect()
torch.cuda.empty_cache()

pipe = TripoSGPipeline.from_pretrained(weights / "TripoSG").to(dtype=torch.float16)
pipe.model_cpu_offload_seq = "image_encoder_dinov2->transformer->vae"  # 원본 pipeline 에 순서가 없어서 지정
pipe.enable_model_cpu_offload()

start = time.time()
with torch.no_grad():
    vertices, faces = pipe(
        image=image,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        flash_octree_depth=args.octree_depth,
    ).samples[0]
mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
if args.faces > 0:
    mesh = simplify_mesh(mesh, args.faces)
mesh.export(out)
print(f"{out}: {len(mesh.faces)} faces, {time.time() - start:.0f}s, "
      f"GPU peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
