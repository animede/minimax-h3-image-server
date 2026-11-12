"""プロンプト拡張: 本家 MiniMax-H3-Image-PE 同梱の公式ルール(task_prompt)を OpenAI 互換 LLM に送り、
短い指示(日本語可)を公式ルール準拠の英語プロンプト + 推奨アスペクト比に変換する。

- t2i: PE の `prompt/t2i/system_prompt.txt` + `task_prompt.txt`(`{}` に入力を差し込む)。
  出力 JSON: {"expanded_caption", "aspect_ratio"}(比率は 10 種の列挙)。
- edit: PE の `prompt/edit/task_prompt.txt` を system に置き、user は
  [Reference Media テキスト, 画像]×N + 最終テキスト(User Editing Instruction / User Aspect Ratio Selection)。
  出力 JSON: {"expanded_caption", "target_aspect_ratio"}。マルチモーダル対応の LLM が必要。
- PE 本体(Qwen 系 FP4 ~19GB)ではなく手元の LLM を使うため、精度は本家 PE に劣りうる。
  公式ルール文書は `models/prompts/` に無ければ PE リポジトリから取得する(ライセンスは本家に従う)。
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from fractions import Fraction
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .config import Settings

logger = logging.getLogger("h3image.enhance")

PE_REPO = "MiniMaxAI/MiniMax-H3-Image-PE"
# 本家 GitHub の prompt-writing skill(2026-11-11 公開)。t2i / edit のガイド本文を PE ルールの補足として system に足す
SKILL_RAW_BASE = "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3-Image/main/skills/h3-image-prompt-writing/"
SKILL_FILES = {
    "skill_t2i": "references/t2i-en.txt",
    "skill_edit": "references/edit-en.txt",
}
PE_FILES = {
    "t2i_system": "prompt/t2i/system_prompt.txt",
    "t2i_task": "prompt/t2i/task_prompt.txt",
    "t2i_params": "prompt/t2i/request_params.json",
    "edit_task": "prompt/edit/task_prompt.txt",
}
ALLOWED_RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16"]
# 本家推奨サイズ(約 16MP)を 4MP(2048² 相当)に縮めた、比率ごとの推奨 width×height(32 の倍数)
TARGET_AREA = 2048 * 2048


class EnhanceError(RuntimeError):
    pass


class EnhanceUnavailable(EnhanceError):
    """LLM が設定されていない(501)。"""


class EnhanceConnectionError(EnhanceError):
    """LLM に接続できない / 応答が不正(502)。"""


def suggested_size(ratio: str) -> tuple[int, int] | None:
    try:
        w, h = (int(x) for x in ratio.split(":"))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    scale = math.sqrt(TARGET_AREA / (w * h))
    return (int(round(w * scale / 32)) * 32, int(round(h * scale / 32)) * 32)


class PromptEnhancer:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.prompt_dir = settings.models_dir / "prompts"
        self._texts: dict[str, str] = {}

    @property
    def available(self) -> bool:
        return bool(self.s.llm_base_url and self.s.llm_model)

    # -- 公式ルール文書 ----------------------------------------------------------

    def _text(self, key: str) -> str:
        if key in self._texts:
            return self._texts[key]
        if key in SKILL_FILES:
            rel = SKILL_FILES[key]
            local = self.prompt_dir / "skill" / rel
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                res = httpx.get(SKILL_RAW_BASE + rel, timeout=30.0, follow_redirects=True)
                res.raise_for_status()
                local.write_text(res.text, encoding="utf-8")
                logger.info("skill guide fetched: %s", rel)
        else:
            rel = PE_FILES[key]
            local = self.prompt_dir / rel
            if not local.exists():
                from huggingface_hub import hf_hub_download

                local.parent.mkdir(parents=True, exist_ok=True)
                src = hf_hub_download(PE_REPO, rel)
                local.write_bytes(Path(src).read_bytes())
                logger.info("PE prompt file fetched: %s", rel)
        self._texts[key] = local.read_text(encoding="utf-8")
        return self._texts[key]

    def _skill_guide(self, key: str) -> str | None:
        """公式 skill のガイド(任意)。取得できなければ PE ルールだけで続行する。"""
        if not self.s.enhance_skill_guide:
            return None
        try:
            return self._text(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skill guide %s を取得できません(PE ルールのみで続行): %s", key, exc)
            return None

    # -- リクエスト組み立て ------------------------------------------------------

    def build_t2i(self, prompt: str) -> tuple[list[dict], list[str]]:
        system = self._text("t2i_system").rstrip("\n")
        template = self._text("t2i_task")
        messages = [{"role": "system", "content": system}]
        guide = self._skill_guide("skill_t2i")
        if guide:
            messages.append({"role": "system", "content":
                             "Additional writing guide from the official MiniMax-H3-Image prompt-writing skill. "
                             "Follow it when writing expanded_caption; the output format above is unchanged.\n\n" + guide})
        messages.append({"role": "user", "content": template.format(prompt)})
        return messages, ["expanded_caption", "aspect_ratio"]

    def build_edit(self, prompt: str, image_paths: list[Path],
                   aspect_ratio: str | None) -> tuple[list[dict], list[str]]:
        system = self._text("edit_task")
        parts: list[dict] = []
        for i, path in enumerate(image_paths, start=1):
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                frac = Fraction(im.width, im.height)
                ratio = f"{frac.numerator}:{frac.denominator}"
                # LLM へ送るプレビューは長辺 1024 に縮小(比率は上のメタデータで渡す)
                im.thumbnail((1024, 1024))
                buf = __import__("io").BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=90)
            parts.append({"type": "text", "text": f"Reference Media: <Picture {i}>, aspect_ratio {ratio}"})
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        parts.append({"type": "text", "text": (
            f"User Editing Instruction (JSON string): {json.dumps(prompt, ensure_ascii=False)}\n"
            f"User Aspect Ratio Selection: {aspect_ratio or 'adaptive'}")})
        messages = [{"role": "system", "content": system}]
        guide = self._skill_guide("skill_edit")
        if guide:
            messages.append({"role": "system", "content":
                             "Additional writing guide from the official MiniMax-H3-Image prompt-writing skill. "
                             "Follow it when writing expanded_caption; the output format above is unchanged.\n\n" + guide})
        messages.append({"role": "user", "content": parts})
        return messages, ["expanded_caption", "target_aspect_ratio"]

    # -- 実行 ------------------------------------------------------------------

    def enhance(self, prompt: str, mode: str, image_paths: list[Path] | None = None,
                aspect_ratio: str | None = None) -> dict:
        if not self.available:
            raise EnhanceUnavailable(
                "プロンプト拡張には OpenAI 互換の LLM が必要です(H3IMG_LLM_BASE_URL / H3IMG_LLM_MODEL を設定)")
        prompt = (prompt or "").strip()
        if not prompt:
            raise EnhanceError("prompt が空です")
        if mode == "edit":
            if not image_paths:
                raise EnhanceError("edit の拡張には参照画像(conditions)が必要です")
            messages, keys = self.build_edit(prompt, image_paths, aspect_ratio)
        else:
            messages, keys = self.build_t2i(prompt)
        body = {"model": self.s.llm_model, "messages": messages, "stream": False,
                "temperature": 0.7, "max_tokens": 4096}
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=self.s.llm_timeout_s) as client:
                res = client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise EnhanceConnectionError(f"LLM に接続できません({url}): {exc}") from exc
        if res.status_code != 200:
            raise EnhanceConnectionError(f"LLM が {res.status_code} を返しました: {res.text[:300]}")
        try:
            content = res.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise EnhanceConnectionError(f"LLM の応答形式が不正です: {res.text[:300]}") from exc
        result = _parse_json_object(content, keys)
        caption = str(result[keys[0]]).strip()
        ratio = str(result[keys[1]]).strip()
        out = {"prompt": caption, "aspect_ratio": ratio, "model": self.s.llm_model, "mode": mode}
        size = suggested_size(ratio)
        if size:
            out["suggested_width"], out["suggested_height"] = size
        return out


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _parse_json_object(text: str, keys: list[str]) -> dict:
    """思考タグや Markdown フェンスを剥がし、必要キーを持つ最初の JSON オブジェクトを取り出す。"""
    cleaned = _THINK_RE.sub("", text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    candidates = [cleaned]
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start:i + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in keys):
            return obj
    raise EnhanceConnectionError(f"LLM の応答から {keys} を持つ JSON を取り出せませんでした: {cleaned[:300]}")
