from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampledCondition:
    training_lane: str
    condition: dict[str, Any]
    used_fallback: bool = False


class ExpertCurriculumSampler:
    def __init__(
        self,
        conditions: list[dict[str, Any]],
        sampler_config: dict[str, Any],
        seed: int,
    ) -> None:
        self.conditions = conditions
        self.sampler_config = sampler_config
        self.rng = random.Random(seed)

    def sample(self, success_counts: dict[str, int]) -> SampledCondition:
        requested_lane = self._sample_lane()
        candidates = self._lane_candidates(requested_lane)
        used_fallback = False

        if not candidates and requested_lane != "core_syndrome_reasoning":
            requested_lane = "core_syndrome_reasoning"
            candidates = self._lane_candidates(requested_lane)
            used_fallback = True

        if not candidates:
            return SampledCondition(
                training_lane="disease_only_uniform_random_fallback",
                condition=self.rng.choice(self.conditions),
                used_fallback=True,
            )

        selected = self._pick_preferred_condition(
            candidates=candidates,
            lane=requested_lane,
            success_counts=success_counts,
        )
        return SampledCondition(
            training_lane=requested_lane,
            condition=selected,
            used_fallback=used_fallback,
        )

    def _sample_lane(self) -> str:
        lane_weights = self.sampler_config["active_lanes_initial"]
        lanes = list(lane_weights.keys())
        weights = list(lane_weights.values())
        return self.rng.choices(lanes, weights=weights, k=1)[0]

    def _lane_candidates(self, lane: str) -> list[dict[str, Any]]:
        return [
            condition
            for condition in self.conditions
            if lane in condition.get("expert_curriculum", {}).get("curriculum_lanes", [])
        ]

    def _pick_preferred_condition(
        self,
        candidates: list[dict[str, Any]],
        lane: str,
        success_counts: dict[str, int],
    ) -> dict[str, Any]:
        working = list(candidates)

        if lane == "high_consequence_must_not_miss":
            must_not_miss = [
                condition
                for condition in working
                if condition.get("expert_curriculum", {}).get("must_not_miss") is True
            ]
            if must_not_miss:
                working = must_not_miss

        expert_value_preferred = [
            condition
            for condition in working
            if condition.get("expert_curriculum", {}).get("expert_value") in {"high", "medium"}
        ]
        if expert_value_preferred:
            working = expert_value_preferred

        candidate_counts = [
            success_counts.get(condition["condition_id"], 0) for condition in working
        ]
        if all(count == 0 for count in candidate_counts):
            return self.rng.choice(working)

        least_seen_count = min(candidate_counts)
        least_seen = [
            condition
            for condition in working
            if success_counts.get(condition["condition_id"], 0) == least_seen_count
        ]
        return self.rng.choice(least_seen)
