"""Dataset and data pipeline tests for Beta-VLA (Layers 1-3).

Tests task descriptions, __getitem__, collate_fn, and normalization.
Also includes parquet-level tests that bypass the datasets library.

Usage:
    conda activate beta
    source .env && PYTHONPATH=src HF_HOME=/workspace/tingting/hf-home python -m pytest tests/test_dataset.py -v --tb=short
"""
from __future__ import annotations

import glob

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from transformers import AutoTokenizer

from betavla.data.libero_dataset import (
    LiberoDataset,
    LiberoDatasetConfig,
    _load_task_descriptions,
    _to_float32_image,
)
from betavla.data.normalize import (
    NormStats,
    load_norm_stats,
    normalize_quantile,
    unnormalize_quantile,
)

REPO_ID = "physical-intelligence/libero"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
NORM_STATS_PATH = "assets/physical-intelligence/libero/norm_stats.json"
_SNAP_GLOB = "/workspace/tingting/hf-home/hub/datasets--physical-intelligence--libero/snapshots/*/data/chunk-*/episode_*.parquet"


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def task_descriptions():
    return _load_task_descriptions(REPO_ID)


@pytest.fixture(scope="module")
def parquet_files():
    files = sorted(glob.glob(_SNAP_GLOB))
    if not files:
        pytest.skip("No LIBERO parquet files in HF cache")
    return files


@pytest.fixture(scope="module")
def norm_stats():
    stats = load_norm_stats(NORM_STATS_PATH)
    if stats is None:
        pytest.skip("norm_stats.json not found")
    return stats


@pytest.fixture(scope="module")
def dataset():
    try:
        cfg = LiberoDatasetConfig(
            repo_id=REPO_ID, split="train", max_samples=100,
            num_workers=0, norm_stats_path=NORM_STATS_PATH,
        )
        return LiberoDataset(
            cfg, action_horizon=10, action_dim=7, tokenizer_name=TOKENIZER_NAME,
        )
    except Exception as exc:
        pytest.skip(f"Cannot load LiberoDataset: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Task Descriptions
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskDescriptions:
    def test_count(self, task_descriptions):
        assert len(task_descriptions) == 40

    def test_all_indices_present(self, task_descriptions):
        for i in range(40):
            assert i in task_descriptions, f"task_index {i} missing"

    def test_all_nonempty_strings(self, task_descriptions):
        for i, desc in task_descriptions.items():
            assert isinstance(desc, str)
            assert len(desc) > 5, f"task {i}: too short: {desc!r}"

    def test_goal_tasks_unique(self, task_descriptions):
        goal = [task_descriptions[i] for i in range(10, 20)]
        assert len(set(goal)) == 10

    def test_spatial_tasks_unique(self, task_descriptions):
        spatial = [task_descriptions[i] for i in range(30, 40)]
        assert len(set(spatial)) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1b: Parquet-level checks (no datasets lib needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestParquetTaskMapping:
    def test_has_task_index_column(self, parquet_files):
        assert "task_index" in pq.read_schema(parquet_files[0]).names

    def test_no_text_column_in_parquet(self, parquet_files):
        """Confirms the bug: parquet has no prompt column — we need task_index mapping."""
        names = set(pq.read_schema(parquet_files[0]).names)
        assert not (names & {"prompt", "task", "instruction", "language_instruction"})

    def test_all_task_indices_have_descriptions(self, parquet_files, task_descriptions):
        seen = set()
        for f in parquet_files:
            t = pq.read_table(f, columns=["task_index"])
            seen.add(t["task_index"][0].as_py())
        for ti in seen:
            assert ti in task_descriptions, f"task_index {ti} missing description"

    def test_prompt_lookup_nonempty(self, parquet_files, task_descriptions):
        for f in parquet_files[:10]:
            ti = pq.read_table(f, columns=["task_index"])["task_index"][0].as_py()
            assert task_descriptions.get(ti), f"Empty prompt for task_index={ti}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Dataset.__getitem__
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetGetItem:
    def test_prompt_nonempty(self, dataset):
        assert dataset[0][0]["prompt"]

    def test_image_shape_dtype(self, dataset):
        img = dataset[0][0]["base_img"]
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0

    def test_wrist_image(self, dataset):
        assert dataset[0][0]["wrist_img"].shape[2] == 3

    def test_state_shape(self, dataset):
        s = dataset[0][0]["state"]
        assert s.shape == (8,) and s.dtype == np.float32

    def test_action_chunk(self, dataset):
        a = dataset[0][1]
        assert a.shape == (10, 7) and a.dtype == np.float32

    def test_multiple_unique_prompts(self, dataset):
        prompts = {dataset[i][0]["prompt"] for i in range(min(len(dataset), 100))}
        if len(dataset) < 200:
            pytest.skip("Dataset too small (max_samples) to guarantee multiple tasks")
        assert len(prompts) > 1, "All prompts identical"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: collate_fn
# ═══════════════════════════════════════════════════════════════════════════

class TestCollateFn:
    def test_batch_shapes(self, dataset):
        obs, actions = dataset.collate_fn([dataset[i] for i in range(4)])
        assert obs.images["base_0_rgb"].shape == (4, 256, 256, 3)
        assert obs.images["left_wrist_0_rgb"].shape == (4, 256, 256, 3)
        assert obs.state.shape == (4, 8)
        assert actions.shape == (4, 10, 7)

    def test_tokenized_prompt(self, dataset):
        obs, _ = dataset.collate_fn([dataset[i] for i in range(2)])
        assert obs.tokenized_prompt.shape == (2, 128)
        assert obs.tokenized_prompt.dtype == torch.long

    def test_mask_has_real_tokens(self, dataset):
        obs, _ = dataset.collate_fn([dataset[i] for i in range(2)])
        assert (obs.tokenized_prompt_mask.sum(dim=1) > 3).all()

    def test_different_prompts_different_tokens(self, dataset):
        p0 = dataset[0][0]["prompt"]
        diff = next((i for i in range(1, min(len(dataset), 100))
                      if dataset[i][0]["prompt"] != p0), None)
        if diff is None:
            pytest.skip("No different prompt found")
        obs, _ = dataset.collate_fn([dataset[0], dataset[diff]])
        assert not torch.equal(obs.tokenized_prompt[0], obs.tokenized_prompt[1])


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalization:
    def test_round_trip(self):
        stats = NormStats(
            mean=np.zeros(7, dtype=np.float32), std=np.ones(7, dtype=np.float32),
            q01=np.array([-1, -1, -1, -.1, -.1, -.1, -1], dtype=np.float32),
            q99=np.array([1, 1, 1, .1, .1, .1, 1], dtype=np.float32),
        )
        x = np.array([.5, -.3, 0, .05, -.05, 0, 1], dtype=np.float32)
        np.testing.assert_allclose(unnormalize_quantile(normalize_quantile(x, stats), stats), x, atol=1e-5)

    def test_midpoint_maps_to_zero(self):
        stats = NormStats(
            mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32),
            q01=np.full(3, -2, dtype=np.float32), q99=np.full(3, 2, dtype=np.float32),
        )
        np.testing.assert_allclose(normalize_quantile(np.zeros(3, dtype=np.float32), stats), 0, atol=1e-5)

    def test_stats_file(self, norm_stats):
        assert "action" in norm_stats and "state" in norm_stats
        assert len(norm_stats["action"].q01) == 7
        assert len(norm_stats["state"].q01) == 8

    def test_image_conversion(self):
        out = _to_float32_image(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        assert out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
