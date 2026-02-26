from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SocialRecommender:
    """
    Social recommender aligned with the paper:
    Score(u, c) = cos(E_u, E_c) * log(eta * (C_c + 1)) * exp(-lambda * (t_n - t_c)).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rec_cfg = config.get("recsys", {}) if isinstance(config, dict) else {}
        sim_cfg = config.get("simulation", {}) if isinstance(config, dict) else {}

        self.top_k = int(rec_cfg.get("top_k", sim_cfg.get("recsys_top_k", 10)))
        self.score_eta = float(rec_cfg.get("score_eta", sim_cfg.get("score_eta", 100)))
        self.score_lambda = float(rec_cfg.get("score_lambda", sim_cfg.get("score_lambda", 0.017)))
        self.show_score = bool(rec_cfg.get("show_score", sim_cfg.get("show_score", False)))

        model_path = (
            rec_cfg.get("twhin_bert_path")
            or config.get("paths", {}).get("twhin_bert")
            or config.get("paths", {}).get("twhin_bert_path")
            or os.getenv("TWHIN_BERT_PATH")
            or None
        )
        if not model_path:
            raise FileNotFoundError("TwHIN-BERT path not configured (recsys.twhin_bert_path or paths.twhin_bert).")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return cls

    def _user_text(self, user: Dict[str, Any]) -> str:
        profile = user.get("profile", user)
        name = profile.get("name", "") or profile.get("screen_name", "")
        desc = profile.get("description", "")
        location = profile.get("location", "")
        domains = user.get("domains", []) or []
        domains_text = ", ".join(domains) if isinstance(domains, list) else str(domains)
        return f"{name}. {desc}. Location: {location}. Interests: {domains_text}."

    def _post_text(self, post: Dict[str, Any]) -> str:
        return str(post.get("content", ""))

    def _interaction_count(self, post: Dict[str, Any]) -> float:
        likes = float(post.get("num_likes", 0))
        comments = float(post.get("num_comments", 0))
        reposts = float(post.get("num_reposts", 0))
        dislikes = float(post.get("num_dislikes", 0))
        return likes + comments + reposts + dislikes

    def _time_delta_seconds(self, now: datetime, created_time: Any) -> float:
        if isinstance(created_time, datetime):
            delta = now - created_time
            return max(delta.total_seconds(), 0.0)
        return 0.0

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def score_posts(self, user: Dict[str, Any], posts: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
        if not posts:
            return []
        user_text = self._user_text(user)
        post_texts = [self._post_text(p) for p in posts]
        embeddings = self._encode_texts([user_text] + post_texts)
        user_emb = embeddings[0]
        post_embs = embeddings[1:]

        scored = []
        for i, post in enumerate(posts):
            sim = self._cosine(user_emb, post_embs[i])
            c_c = self._interaction_count(post)
            delta_t = self._time_delta_seconds(now, post.get("created_time"))
            score = sim * math.log(self.score_eta * (c_c + 1.0) + 1e-6) * math.exp(-self.score_lambda * delta_t)
            scored.append({**post, "score": score})
        return scored

    def recommend(self, user: Dict[str, Any], posts: List[Dict[str, Any]], now: Optional[datetime] = None, top_k: Optional[int] = None):
        now = now or datetime.utcnow()
        top_k = int(top_k or self.top_k)
        scored = self.score_posts(user, posts, now)
        ranked = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
        if not self.show_score:
            for p in ranked:
                p.pop("score", None)
        return ranked
