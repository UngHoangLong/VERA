"""
mllm_client.py — Module 5: run an MLLM over Module 4 packages.

Supports two self-hosted backends (both ~8B, sharded across GPUs via device_map):
    --backend qwen      -> Qwen/Qwen3-VL-8B-Instruct
    --backend internvl  -> OpenGVLab/InternVL2_5-8B

For each <video_id>/prompt_package.json from Module 4:
  - build the 3-block prompt (prompt_eng.py)
  - attach the sampled frames (labeled per chunk)
  - run the chosen MLLM
  - parse the <verdict> JSON and save <video_id>_verdict.json

Usage:
    cd src/module_5_agent
    python mllm_client.py --backend qwen
    python mllm_client.py --backend internvl

Requires (on the GPU box):
    pip install "transformers>=4.57" accelerate torchvision timm einops
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.module_5_agent.prompt_eng import build_system_prompt, build_user_prompt

QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
INTERNVL_MODEL = "OpenGVLab/InternVL2_5-8B"


# ---------------------------------------------------------------------------
# Shared: gather labeled frames + parse output
# ---------------------------------------------------------------------------

def collect_frames(package: Dict[str, Any], package_dir: Path) -> List[Tuple[str, List[Path]]]:
    """Return [(chunk_label, [abs frame paths]), ...] in chunk order."""
    groups: List[Tuple[str, List[Path]]] = []
    for chunk in package.get("top_chunks", []):
        frame_files = chunk.get("frame_files", [])
        if not frame_files:
            continue
        tm = chunk.get("time_metadata", {})
        label = (f"Frames for {chunk.get('chunk_id','?')} "
                 f"[{tm.get('start_sec',0):.1f}s-{tm.get('end_sec',0):.1f}s]")
        paths = [(package_dir / fn).resolve() for fn in frame_files]
        groups.append((label, paths))
    return groups


def parse_output(text: str) -> Dict[str, Any]:
    """Extract <think> reasoning and the <verdict> JSON block from model output."""
    cleaned = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", text).strip()

    think_match = re.search(r"<think>(.*?)</think>", cleaned, re.DOTALL)
    verdict_match = re.search(r"<verdict>(.*?)</verdict>", cleaned, re.DOTALL)

    reasoning = think_match.group(1).strip() if think_match else ""
    verdict = None
    if verdict_match:
        raw = verdict_match.group(1).strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            verdict = None

    return {"reasoning": reasoning, "verdict": verdict, "raw_output": cleaned}


# ---------------------------------------------------------------------------
# Backend: Qwen3-VL
# ---------------------------------------------------------------------------

class QwenVLClient:
    def __init__(self, model_name: str = QWEN_MODEL, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _build_messages(self, package, frame_groups):
        content: List[Dict[str, Any]] = [{"type": "text", "text": build_user_prompt(package)}]
        for label, paths in frame_groups:
            content.append({"type": "text", "text": label + ":"})
            for p in paths:
                content.append({"type": "image", "image": str(p)})
        return [
            {"role": "system", "content": [{"type": "text", "text": build_system_prompt()}]},
            {"role": "user", "content": content},
        ]

    def analyze(self, package: Dict[str, Any], package_dir: Path) -> str:
        frame_groups = collect_frames(package, package_dir)
        messages = self._build_messages(package, frame_groups)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=True,
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]


# ---------------------------------------------------------------------------
# Backend: InternVL2.5 (custom preprocessing + model.chat)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_load_image(image_file, input_size=448):
    """Load one frame as a single 448x448 tile (max_num=1 keeps memory low for many frames)."""
    import torch
    from PIL import Image
    image = Image.open(image_file).convert("RGB")
    transform = _internvl_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return pixel_values


class InternVLClient:
    def __init__(self, model_name: str = INTERNVL_MODEL, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

    def analyze(self, package: Dict[str, Any], package_dir: Path) -> str:
        frame_groups = collect_frames(package, package_dir)

        # Build pixel_values (one tile per frame) + num_patches_list, in order.
        pv_list = []
        num_patches_list: List[int] = []
        placeholder_lines: List[str] = []
        for label, paths in frame_groups:
            placeholder_lines.append(label + ":")
            for i, p in enumerate(paths):
                pv = _internvl_load_image(str(p)).to(self.torch.float16)
                pv_list.append(pv)
                num_patches_list.append(pv.size(0))
                placeholder_lines.append(f"Frame-{i+1}: <image>")

        question = build_system_prompt() + "\n\n" + build_user_prompt(package)
        if pv_list:
            pixel_values = self.torch.cat(pv_list, dim=0).to(self.model.device)
            question = question + "\n\nFRAMES:\n" + "\n".join(placeholder_lines)
        else:
            pixel_values = None

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config,
            num_patches_list=num_patches_list if pv_list else None,
            history=None, return_history=False,
        )
        return response


# ---------------------------------------------------------------------------
# Factory + main
# ---------------------------------------------------------------------------

def build_client(backend: str, model: str, max_new_tokens: int):
    if backend == "qwen":
        return QwenVLClient(model or QWEN_MODEL, max_new_tokens)
    if backend == "internvl":
        return InternVLClient(model or INTERNVL_MODEL, max_new_tokens)
    raise ValueError(f"Unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: MLLM deepfake reasoning.")
    parser.add_argument("--backend", required=True, choices=["qwen", "internvl"])
    parser.add_argument("--packages_dir", type=str,
                        default="../module_4_retrieval/module4_packages")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: ./verdicts_<backend>")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the model id for the chosen backend.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    packages_dir = Path(args.packages_dir)
    package_dirs = sorted(d for d in packages_dir.iterdir()
                          if d.is_dir() and (d / "prompt_package.json").exists())
    if not package_dirs:
        raise FileNotFoundError(f"No prompt_package.json found under {packages_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./verdicts_{args.backend}")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.backend, args.model, args.max_new_tokens)

    skipped = 0
    for pdir in tqdm(package_dirs, desc=f"Module 5 ({args.backend})", unit="video"):
        out_path = output_dir / f"{pdir.name}_verdict.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            with open(pdir / "prompt_package.json", encoding="utf-8") as f:
                package = json.load(f)
            raw = client.analyze(package, pdir)
            result = parse_output(raw)
            result["video_id"] = package.get("video_id", pdir.name)
            result["backend"] = args.backend
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"Lỗi khi xử lý {pdir.name}: {e}")

    if skipped:
        tqdm.write(f"Bỏ qua {skipped}/{len(package_dirs)} video đã có verdict.")
    print(f"[DONE] Verdicts saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
