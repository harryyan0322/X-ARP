#!/usr/bin/env python3
"""
X-ARP main entry (Generate–Detect–Explain–Refine loop).

This script follows the Methodology section of the paper:
1) Generate: LLM agents synthesize user profiles, content, and relations.
2) Detect: an explainable bot detector evaluates realism on the mixed network.
3) Explain: extract evidence and produce human-readable explanations.
4) Refine: convert explanations to re-prompts and update agent memory.

Design goals:
- Runnable by executing this file directly.
- Graceful degradation if data/model artifacts are missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root and src are importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class PipelinePaths:
    root: Path
    config_path: Path
    output_dir: Path
    explanations_dir: Path
    logs_dir: Path


def _setup_logging(logs_dir: Path, level: str = "INFO") -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xarp.pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if re-imported
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(
            logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        from yaml import safe_load
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return safe_load(f) or {}


def _build_paths() -> PipelinePaths:
    root = PROJECT_ROOT
    return PipelinePaths(
        root=root,
        config_path=root / "config.yaml",
        output_dir=root / "output",
        explanations_dir=root / "explanations",
        logs_dir=root / "logs",
    )


def _get_user_path(cfg: Dict) -> Optional[Path]:
    env_path = os.getenv("PIPELINE_USER_PATH")
    if env_path:
        return Path(env_path)
    data_cfg = cfg.get("data", {})
    user_path = data_cfg.get("user_path")
    return Path(user_path) if user_path else None


def _get_detector_paths(cfg: Dict) -> Dict[str, Path]:
    env_model = os.getenv("DETECTOR_MODEL_PATH")
    env_data = os.getenv("DETECTOR_DATA_DIR")

    model_path = Path(env_model) if env_model else PROJECT_ROOT / "src" / "detector" / "models_quick" / "AllInOne1_rgcn_rgt_gcn_20250719" / "best_seed642414.pth"
    data_dir = Path(env_data) if env_data else PROJECT_ROOT / "src" / "detector" / "processed_data" / "merged_cleaned_profiles"
    return {"model_path": model_path, "data_dir": data_dir}


def _output_paths(cfg: Dict, output_dir: Path, cycle_num: int) -> Dict[str, Path]:
    data_cfg = cfg.get("data", {})
    mapping_suffix = data_cfg.get("mapping_suffix", "_agent_user_mapping")
    full_result_suffix = data_cfg.get("full_result_suffix", "_full_result_with_metadata")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bot_data_complete_cycle{cycle_num}.json"

    mapping_path = output_path.with_name(output_path.stem + mapping_suffix + ".json")
    full_result_path = output_path.with_name(output_path.stem + full_result_suffix + ".json")

    return {
        "output_path": output_path,
        "mapping_path": mapping_path,
        "full_result_path": full_result_path,
    }


def _run_async(coro):
    return asyncio.run(coro)


# --------------------------- Pipeline Steps ---------------------------

def step_generate(
    cfg: Dict,
    paths: PipelinePaths,
    cycle_num: int,
    explanations_path: Optional[Path] = None,
    agent_ids_to_adjust: Optional[List[int]] = None,
) -> Dict:
    """Generate synthetic network using LLM agents (paper §3.2)."""
    logger = logging.getLogger("xarp.pipeline")
    user_path = _get_user_path(cfg)

    if not paths.config_path.exists():
        logger.warning("[Generate] config.yaml not found; skipping generation.")
        return {"status": "skipped", "reason": "config_missing"}

    if not user_path or not user_path.exists():
        logger.warning("[Generate] user_path not found; skipping generation.")
        return {"status": "skipped", "reason": "user_path_missing"}

    io_paths = _output_paths(cfg, paths.output_dir, cycle_num)
    output_path = io_paths["output_path"]

    logger.info("[Generate] user_path=%s", user_path)
    logger.info("[Generate] output_path=%s", output_path)
    if explanations_path:
        logger.info("[Generate] explanations_path=%s", explanations_path)

    try:
        from src.generator.graph_llm_simulation import main as generator_main

        result = _run_async(
            generator_main(
                config_path=str(paths.config_path),
                user_path=str(user_path),
                explanations_path=str(explanations_path) if explanations_path else None,
                agent_ids_to_adjust=agent_ids_to_adjust,
                output_path=str(output_path),
            )
        )

        return {
            "status": "success",
            "result": result,
            **io_paths,
        }
    except Exception as e:
        logger.exception("[Generate] failed: %s", e)
        return {"status": "error", "error": str(e)}


def step_detect(cfg: Dict, paths: PipelinePaths, generated_output: Optional[Path], cycle_num: int) -> Dict:
    """Run detector on mixed network (paper §3.3)."""
    logger = logging.getLogger("xarp.pipeline")
    detector_paths = _get_detector_paths(cfg)
    model_path = detector_paths["model_path"]
    train_each_cycle = cfg.get("detector", {}).get("train_each_cycle", True)

    user_path = _get_user_path(cfg)
    if not user_path or not user_path.exists():
        logger.warning("[Detect] user_path missing; skipping.")
        return {"status": "skipped", "reason": "user_path_missing"}

    if not generated_output or not generated_output.exists():
        logger.warning("[Detect] generated output missing; skipping.")
        return {"status": "skipped", "reason": "generated_output_missing"}

    if not model_path.exists():
        logger.warning("[Detect] model missing; skipping: %s", model_path)
        return {"status": "skipped", "reason": "model_missing"}

    try:
        from src.detector.pipeline import (
            build_mixed_dataset,
            prepare_graph_data,
            run_inference,
            save_predictions,
            train_detector_model,
        )

        mixed_path = paths.output_dir / f"mixed_cycle{cycle_num}.json"
        mixed, mapping = build_mixed_dataset(user_path, generated_output, mixed_path)

        processed_dir = paths.output_dir / f"processed_data_cycle{cycle_num}"
        prepare_graph_data(
            mixed_json_path=mixed_path,
            output_dir=processed_dir,
            bert_model_path=PROJECT_ROOT / "bert-base-uncased",
        )

        mapping_path = processed_dir / "node_id_mapping.json"
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        trained_model_path = model_path
        training_info = None
        if train_each_cycle:
            model_dir = paths.output_dir / f"detector_models_cycle{cycle_num}"
            try:
                training_info = train_detector_model(
                    data_dir=processed_dir,
                    save_root=model_dir,
                    device=cfg.get("detector", {}).get("device"),
                    exp_name=f"xarp_cycle{cycle_num}",
                )
                trained_model_path = Path(training_info.get("model_path", model_path))
                logger.info("[Detect] trained detector model: %s", trained_model_path)
            except Exception as train_err:
                logger.exception("[Detect] training failed; fallback to existing model: %s", train_err)

        predictions = run_inference(model_path=trained_model_path, data_dir=processed_dir)
        pred_path = paths.output_dir / f"detector_predictions_cycle{cycle_num}.json"
        save_predictions(predictions, pred_path)

        return {
            "status": "success",
            "model_path": trained_model_path,
            "processed_dir": processed_dir,
            "mapping_path": mapping_path,
            "predictions_path": pred_path,
            "mixed_path": mixed_path,
            "training_info": training_info,
        }
    except Exception as e:
        logger.exception("[Detect] failed: %s", e)
        return {"status": "error", "error": str(e)}


def step_explain(cfg: Dict, paths: PipelinePaths, cycle_num: int, detector_info: Optional[Dict] = None) -> Dict:
    """Generate explainability reports (paper §3.3/§3.4)."""
    logger = logging.getLogger("xarp.pipeline")
    detector_paths = _get_detector_paths(cfg)
    model_path = detector_paths["model_path"]
    data_dir = detector_paths["data_dir"]
    mapping_path = None

    if detector_info and detector_info.get("status") == "success":
        model_path = detector_info.get("model_path", model_path)
        data_dir = detector_info.get("processed_dir", data_dir)
        mapping_path = detector_info.get("mapping_path")

    if not Path(model_path).exists() or not Path(data_dir).exists():
        logger.warning("[Explain] model or data missing; skipping.")
        return {"status": "skipped", "reason": "model_or_data_missing"}

    try:
        from src.explainer.main_explainer import main as explainer_main

        # Pass paths via environment for flexible pipeline integration
        os.environ["EXPLAINER_MODEL_PATH"] = str(model_path)
        os.environ["EXPLAINER_DATA_DIR"] = str(data_dir)
        if mapping_path:
            os.environ["EXPLAINER_MAPPING_PATH"] = str(mapping_path)

        explainer_main()

        # Snapshot outputs with cycle suffix
        paths.explanations_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "natural_language": paths.explanations_dir / "natural_language_explanations.json",
            "all_users": paths.explanations_dir / "all_users_explanations.json",
            "summary": paths.explanations_dir / "summary_report.json",
            "re_prompt": paths.explanations_dir / "re_prompt_explanations.json",
        }

        cycle_outputs = {
            "natural_language": paths.explanations_dir / f"natural_language_explanations_cycle{cycle_num}.json",
            "all_users": paths.explanations_dir / f"all_users_explanations_cycle{cycle_num}.json",
            "summary": paths.explanations_dir / f"summary_report_cycle{cycle_num}.json",
            "re_prompt": paths.explanations_dir / f"re_prompt_explanations_cycle{cycle_num}.json",
        }

        for key, src in outputs.items():
            if src.exists():
                cycle_outputs[key].write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                logger.warning("[Explain] missing output: %s", src)

        return {"status": "success", "outputs": cycle_outputs}

    except Exception as e:
        logger.exception("[Explain] failed: %s", e)
        return {"status": "error", "error": str(e)}


def _load_mapping(mapping_path: Path) -> Dict[str, str]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if "agent_id_to_user_id" in data:
        return {str(k): str(v) for k, v in data["agent_id_to_user_id"].items()}
    return {str(k): str(v) for k, v in data.items()}


def _replace_user_ids(explanations: List[str], user_to_agent: Dict[str, str]) -> Dict[str, List[str]]:
    updated_explanations: List[str] = []
    agent_ids_to_adjust = set()

    user_id_pattern = re.compile(r"User ([^\\s,.;:]+)")

    def _replace(match):
        user_id = match.group(1)
        if user_id in user_to_agent:
            agent_id = user_to_agent[user_id]
            agent_ids_to_adjust.add(agent_id)
            return f"Agent {agent_id}"
        return match.group(0)

    for explanation in explanations:
        updated = user_id_pattern.sub(_replace, explanation)
        updated_explanations.append(updated)

    return {
        "updated_explanations": updated_explanations,
        "agent_ids_to_adjust": sorted(agent_ids_to_adjust),
    }


def _load_agents_to_adjust(adjusted_path: Path) -> List[int]:
    try:
        data = json.loads(adjusted_path.read_text(encoding="utf-8"))
        raw_ids = data.get("agents_to_adjust", [])
        cleaned: List[int] = []
        for item in raw_ids:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                continue
        return cleaned
    except Exception:
        return []


def step_refine(
    paths: PipelinePaths,
    cycle_num: int,
    mapping_path: Optional[Path],
    explanations_path: Optional[Path],
) -> Dict:
    """Convert explanations to re-prompts and prepare refinement input (paper §3.4/§3.5)."""
    logger = logging.getLogger("xarp.pipeline")

    if not mapping_path or not mapping_path.exists():
        logger.warning("[Refine] mapping file missing; skipping.")
        return {"status": "skipped", "reason": "mapping_missing"}

    if not explanations_path or not explanations_path.exists():
        logger.warning("[Refine] explanations missing; skipping.")
        return {"status": "skipped", "reason": "explanations_missing"}

    try:
        mapping = _load_mapping(mapping_path)
        user_to_agent = {str(v): str(k) for k, v in mapping.items()}

        explanations = json.loads(explanations_path.read_text(encoding="utf-8"))
        if not isinstance(explanations, list):
            logger.warning("[Refine] explanations is not a list; skipping.")
            return {"status": "skipped", "reason": "invalid_explanations_format"}

        replaced = _replace_user_ids(explanations, user_to_agent)

        adjusted_file = paths.explanations_dir / f"adjusted_natural_language_explanations_cycle{cycle_num}.json"
        adjusted_payload = {
            "cycle_num": cycle_num,
            "adjustment_timestamp": datetime.now().isoformat(),
            "total_explanations": len(replaced["updated_explanations"]),
            "total_agents_to_adjust": len(replaced["agent_ids_to_adjust"]),
            "agents_to_adjust": replaced["agent_ids_to_adjust"],
            "explanations": replaced["updated_explanations"],
        }
        adjusted_file.write_text(json.dumps(adjusted_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "status": "success",
            "adjusted_file": adjusted_file,
            "agents_to_adjust": replaced["agent_ids_to_adjust"],
        }

    except Exception as e:
        logger.exception("[Refine] failed: %s", e)
        return {"status": "error", "error": str(e)}


# --------------------------- Main Loop ---------------------------

def run_cycle(cfg: Dict, paths: PipelinePaths, cycle_num: int, enable_adjustment: bool, prev_adjusted: Optional[Path]) -> Dict:
    logger = logging.getLogger("xarp.pipeline")
    logger.info("[Cycle %s] start", cycle_num)

    explanations_path = prev_adjusted if enable_adjustment and prev_adjusted else None
    agent_ids_to_adjust: Optional[List[int]] = None
    if explanations_path and explanations_path.exists():
        agent_ids_to_adjust = _load_agents_to_adjust(explanations_path)
        if agent_ids_to_adjust:
            logger.info("[Cycle %s] loaded %s agents to adjust", cycle_num, len(agent_ids_to_adjust))

    # Step 1: Generate
    gen = step_generate(cfg, paths, cycle_num, explanations_path=explanations_path, agent_ids_to_adjust=agent_ids_to_adjust)

    # Step 2: Detect
    gen_output = gen.get("output_path") if isinstance(gen, dict) else None
    det = step_detect(cfg, paths, gen_output, cycle_num)

    # Step 3: Explain
    exp = step_explain(cfg, paths, cycle_num, det)

    # Step 4: Refine
    mapping_path = gen.get("mapping_path") if gen.get("status") == "success" else None
    reprompt_path = paths.explanations_dir / f"re_prompt_explanations_cycle{cycle_num}.json"
    natural_expl_path = paths.explanations_dir / f"natural_language_explanations_cycle{cycle_num}.json"
    expl_path = reprompt_path if reprompt_path.exists() else natural_expl_path
    ref = step_refine(paths, cycle_num, mapping_path, expl_path)

    cycle_result = {
        "cycle_number": cycle_num,
        "generation": gen,
        "detection": det,
        "explanation": exp,
        "refinement": ref,
        "status": "completed",
    }

    logger.info("[Cycle %s] end", cycle_num)
    return cycle_result


def main():
    paths = _build_paths()
    cfg = _load_yaml(paths.config_path)
    pipeline_cfg = cfg.get("pipeline", {})

    logger = _setup_logging(paths.logs_dir, pipeline_cfg.get("log_level", "INFO"))

    max_cycles = int(pipeline_cfg.get("max_cycles", 1))
    cycle_delay = int(pipeline_cfg.get("cycle_delay", 0))
    enable_adjustment = bool(pipeline_cfg.get("enable_adjustment_mode", True))

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.explanations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("X-ARP Pipeline start")
    logger.info("config_path=%s", paths.config_path)
    logger.info("max_cycles=%s, cycle_delay=%s, adjustment=%s", max_cycles, cycle_delay, enable_adjustment)

    all_results = {
        "pipeline_start": datetime.now().isoformat(),
        "config_path": str(paths.config_path),
        "max_cycles": max_cycles,
        "cycle_delay": cycle_delay,
        "enable_adjustment": enable_adjustment,
        "cycles": [],
    }

    prev_adjusted: Optional[Path] = None

    for cycle_num in range(1, max_cycles + 1):
        cycle_result = run_cycle(cfg, paths, cycle_num, enable_adjustment, prev_adjusted)
        all_results["cycles"].append(cycle_result)

        # Prepare for next cycle
        ref = cycle_result.get("refinement", {})
        if ref.get("status") == "success":
            prev_adjusted = Path(ref["adjusted_file"])
        else:
            prev_adjusted = None

        if cycle_num < max_cycles and cycle_delay > 0:
            logger.info("Waiting %s seconds before next cycle...", cycle_delay)
            import time
            time.sleep(cycle_delay)

    results_file = paths.logs_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Pipeline results saved: %s", results_file)


if __name__ == "__main__":
    main()
