"""
mllm_client.py — Module 5: run Qwen3-VL reasoning over Module 4 packages.

For each <video_id>/prompt_package.json produced by Module 4:
  - build the 3-block prompt (prompt_eng.py)
  - attach the sampled frames (labeled per chunk)
  - run Qwen3-VL-8B-Instruct
  - parse the <verdict> JSON and save <video_id>_verdict.json

Usage:
    cd src/module_5_agent
    python mllm_client.py --packages_dir ../module_4_retrieval/module4_packages

Requires (on the GPU box):
    pip install "transformers>=4.57" accelerate torchvision
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.module_5_agent.prompt_eng import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(package: Dict[str, Any], package_dir: Path) -> List[Dict[str, Any]]:
    """Build chat messages: system + user(text prompt followed by labeled frames)."""
    user_text = build_user_prompt(package)

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for chunk in package.get("top_chunks", []):
        frame_files = chunk.get("frame_files", [])
        if not frame_files:
            continue
        tm = chunk.get("time_metadata", {})
        label = (f"Frames for {chunk.get('chunk_id','?')} "
                 f"[{tm.get('start_sec',0):.1f}s-{tm.get('end_sec',0):.1f}s]:")
        content.append({"type": "text", "text": label})
        for fname in frame_files:
            abs_path = (package_dir / fname).resolve()
            content.append({"type": "image", "image": f"file://{abs_path}"})

    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def parse_output(text: str) -> Dict[str, Any]:
    """Extract <think> reasoning and the <verdict> JSON block from model output."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    verdict_match = re.search(r"<verdict>(.*?)</verdict>", text, re.DOTALL)

    reasoning = think_match.group(1).strip() if think_match else ""
    verdict: Optional[Dict[str, Any]] = None
    if verdict_match:
        raw = verdict_match.group(1).strip()
        # tolerate ```json fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            verdict = None

    return {"reasoning": reasoning, "verdict": verdict, "raw_output": text}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class QwenVLClient:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 2048):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def analyze(self, package_dir: Path) -> Dict[str, Any]:
        with open(package_dir / "prompt_package.json", encoding="utf-8") as f:
            package = json.load(f)
        messages = build_messages(package, package_dir)
        output_text = self.generate(messages)
        result = parse_output(output_text)
        result["video_id"] = package.get("video_id", package_dir.name)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: Qwen3-VL deepfake reasoning.")
    parser.add_argument("--packages_dir", type=str,
                        default="../module_4_retrieval/module4_packages",
                        help="Directory of <video_id>/prompt_package.json from Module 4.")
    parser.add_argument("--output_dir", type=str, default="./verdicts")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    packages_dir = Path(args.packages_dir)
    package_dirs = sorted(d for d in packages_dir.iterdir()
                          if d.is_dir() and (d / "prompt_package.json").exists())
    if not package_dirs:
        raise FileNotFoundError(f"No prompt_package.json found under {packages_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = QwenVLClient(args.model, args.max_new_tokens)

    skipped = 0
    for pdir in tqdm(package_dirs, desc="Module 5", unit="video"):
        out_path = output_dir / f"{pdir.name}_verdict.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            result = client.analyze(pdir)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"Lỗi khi xử lý {pdir.name}: {e}")

    if skipped:
        tqdm.write(f"Bỏ qua {skipped}/{len(package_dirs)} video đã có verdict.")
    print(f"[DONE] Verdicts saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
