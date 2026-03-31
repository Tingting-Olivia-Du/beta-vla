"""End-to-end pipeline tests for Beta-VLA.

Verifies data flow from parquet → Dataset → collate → model → loss/sample.
Each layer tests shape, dtype, and semantic invariants.

Usage:
    source .env && PYTHONPATH=src HF_HOME=/workspace/tingting/hf-home python -m pytest tests/test_pipeline.py -v --tb=short
    HF_HOME=/workspace/tingting/hf-home pytest tests/test_pipeline.py -v --tb=short
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from betavla.data.libero_dataset import (
    LiberoDataset,
    LiberoDatasetConfig,
    _load_task_descriptions,
)
from betavla.data.normalize import normalize_quantile, unnormalize_quantile, NormStats
from betavla.data.types import ObservationBatch
from betavla.models.model import BetaVLAConfig, BetaVLAModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REPO_ID = "physical-intelligence/libero"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
NORM_STATS_PATH = "assets/physical-intelligence/libero/norm_stats.json"

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _make_obs(
    batch_size: int = 2,
    image_size: int = 224,
    seq_len: int = 128,
    tokenizer: AutoTokenizer | None = None,
    prompts: list[str] | None = None,
) -> ObservationBatch:
    """Create an ObservationBatch with proper 2-camera layout."""
    if prompts is not None and tokenizer is not None:
        enc = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )
        token_ids = enc["input_ids"]
        token_mask = enc["attention_mask"].bool()
        batch_size = len(prompts)
    else:
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    return ObservationBatch(
        images={
            "base_0_rgb": torch.rand(batch_size, image_size, image_size, 3),
            "left_wrist_0_rgb": torch.rand(batch_size, image_size, image_size, 3),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 8),
        tokenized_prompt=token_ids,
        tokenized_prompt_mask=token_mask,
    )


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped to avoid reloading heavy models)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def task_descriptions():
    return _load_task_descriptions(REPO_ID)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def model():
    cfg = BetaVLAConfig(
        freeze_vision=True,
        freeze_language=False,
        freeze_vggt=False,
        use_lora=False,  # skip LoRA for faster test init
    )
    m = BetaVLAModel(cfg)
    m.eval()
    return m


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
            assert isinstance(desc, str), f"task {i}: expected str, got {type(desc)}"
            assert len(desc) > 5, f"task {i}: description too short: {desc!r}"

    def test_goal_tasks_differ(self, task_descriptions):
        """libero_goal tasks (10-19) should all be unique descriptions."""
        goal_descs = [task_descriptions[i] for i in range(10, 20)]
        assert len(set(goal_descs)) == 10, "Goal task descriptions should be unique"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Dataset.__getitem__
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def dataset():
    try:
        cfg = LiberoDatasetConfig(
            repo_id=REPO_ID,
            split="train",
            max_samples=100,
            num_workers=0,
            norm_stats_path=NORM_STATS_PATH,
        )
        return LiberoDataset(
            cfg, action_horizon=10, action_dim=7, tokenizer_name=TOKENIZER_NAME,
        )
    except Exception as exc:
        pytest.skip(f"Cannot load LIBERO dataset (datasets lib version?): {exc}")


class TestDatasetGetItem:
    def test_prompt_nonempty(self, dataset):
        sample, _ = dataset[0]
        assert sample["prompt"], "prompt should not be empty"

    def test_image_shape_dtype(self, dataset):
        sample, _ = dataset[0]
        img = sample["base_img"]
        assert img.shape == (256, 256, 3), f"unexpected image shape: {img.shape}"
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0

    def test_wrist_image_exists(self, dataset):
        sample, _ = dataset[0]
        assert "wrist_img" in sample
        assert sample["wrist_img"].shape[2] == 3

    def test_state_shape(self, dataset):
        sample, _ = dataset[0]
        assert sample["state"].shape == (8,)
        assert sample["state"].dtype == np.float32

    def test_action_chunk_shape(self, dataset):
        _, actions = dataset[0]
        assert actions.shape == (10, 7)
        assert actions.dtype == np.float32

    def test_different_tasks_have_different_prompts(self, dataset):
        """Different task_index values should produce different prompts."""
        prompts = set()
        for i in range(min(len(dataset), 100)):
            sample, _ = dataset[i]
            prompts.add(sample["prompt"])
        assert len(prompts) > 1, "All prompts are the same — task descriptions not working"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: collate_fn
# ═══════════════════════════════════════════════════════════════════════════

class TestCollateFn:
    def test_batch_shapes(self, dataset):
        batch = [dataset[i] for i in range(4)]
        obs, actions = dataset.collate_fn(batch)

        assert obs.images["base_0_rgb"].shape == (4, 256, 256, 3)
        assert obs.images["left_wrist_0_rgb"].shape == (4, 256, 256, 3)
        assert obs.state.shape == (4, 8)
        assert actions.shape == (4, 10, 7)

    def test_tokenized_prompt_shape(self, dataset):
        batch = [dataset[i] for i in range(2)]
        obs, _ = dataset.collate_fn(batch)

        assert obs.tokenized_prompt.shape == (2, 128)
        assert obs.tokenized_prompt.dtype == torch.long

    def test_prompt_mask_has_real_tokens(self, dataset):
        batch = [dataset[i] for i in range(2)]
        obs, _ = dataset.collate_fn(batch)

        real_tokens = obs.tokenized_prompt_mask.sum(dim=1)  # per sample
        assert (real_tokens > 3).all(), f"Too few real tokens: {real_tokens}"

    def test_different_prompts_have_different_tokens(self, dataset):
        """Samples with different prompts should produce different token IDs."""
        # Find two samples with different prompts
        s0, _ = dataset[0]
        diff_idx = None
        for i in range(1, min(len(dataset), 100)):
            si, _ = dataset[i]
            if si["prompt"] != s0["prompt"]:
                diff_idx = i
                break
        assert diff_idx is not None, "Could not find two different prompts"

        batch = [dataset[0], dataset[diff_idx]]
        obs, _ = dataset.collate_fn(batch)
        assert not torch.equal(obs.tokenized_prompt[0], obs.tokenized_prompt[1])


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Vision Tower
# ═══════════════════════════════════════════════════════════════════════════

class TestVisionTower:
    def test_single_camera_output_shape(self, model):
        D = model.vision_tower.embed_dim
        images = {"base_0_rgb": torch.rand(1, 224, 224, 3)}
        masks = {"base_0_rgb": torch.ones(1, dtype=torch.bool)}
        with torch.no_grad():
            out = model.vision_tower(images, masks)
        assert out.shape == (1, 256, D), f"unexpected shape: {out.shape}"

    def test_two_camera_output_shape(self, model):
        D = model.vision_tower.embed_dim
        images = {
            "base_0_rgb": torch.rand(2, 224, 224, 3),
            "left_wrist_0_rgb": torch.rand(2, 224, 224, 3),
        }
        masks = {
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        }
        with torch.no_grad():
            out = model.vision_tower(images, masks)
        # 2 cameras × 256 patches = 512
        assert out.shape == (2, 512, D), f"unexpected shape: {out.shape}"

    def test_embed_dim_positive(self, model):
        assert model.vision_tower.embed_dim > 0


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: Language Encoder
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageEncoder:
    def test_output_shape(self, model, tokenizer):
        prompts = ["pick up the bowl"]
        enc = tokenizer(prompts, truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model.language_encoder(enc["input_ids"], enc["attention_mask"].bool())
        assert out.shape == (1, 128, model.language_encoder.hidden_size)

    def test_different_prompts_different_embeddings(self, model, tokenizer):
        prompts_a = ["put the bowl on the plate"]
        prompts_b = ["open the middle drawer of the cabinet"]
        enc_a = tokenizer(prompts_a, truncation=True, padding="max_length",
                          max_length=128, return_tensors="pt")
        enc_b = tokenizer(prompts_b, truncation=True, padding="max_length",
                          max_length=128, return_tensors="pt")
        with torch.no_grad():
            out_a = model.language_encoder(enc_a["input_ids"], enc_a["attention_mask"].bool())
            out_b = model.language_encoder(enc_b["input_ids"], enc_b["attention_mask"].bool())

        # Mean-pool over sequence, check cosine similarity
        emb_a = out_a[0, :enc_a["attention_mask"].sum(), :].mean(0)
        emb_b = out_b[0, :enc_b["attention_mask"].sum(), :].mean(0)
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0))
        assert cos_sim.item() < 0.99, f"Embeddings too similar: cosine={cos_sim.item():.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6: VGGT Backbone — Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

class TestVGGTPositionalEncoding:
    def test_pos_dtype_is_long(self, model):
        """2D positions passed to RoPE must be integer (long) tensors."""
        backbone = model.vggt_backbone.model
        if not hasattr(backbone, "position_getter"):
            pytest.skip("Not a VGGT backbone")

        pos = backbone.position_getter(1, 16, 16, device="cpu")
        assert pos.dtype == torch.long or pos.dtype == torch.int64

    def test_2d_grid_not_degenerate(self, model):
        """Camera patches should use proper 2D positions, not 1D (y=0 for all)."""
        backbone = model.vggt_backbone.model
        if not hasattr(backbone, "vision_patch_hw"):
            pytest.skip("Not a VGGT backbone with vision layout")

        # Simulate the forward pos construction
        B, ph, pw = 1, backbone.vision_patch_hw[0], backbone.vision_patch_hw[1]
        cam_pos = backbone.position_getter(B, ph, pw, device="cpu")

        # Y coordinates should span 0..ph-1, not all be 0
        y_coords = cam_pos[0, :, 0].unique()
        assert len(y_coords) == ph, f"Expected {ph} unique y values, got {len(y_coords)}"

    def test_cameras_have_different_y_ranges(self, model):
        """Two cameras should occupy non-overlapping y-coordinate ranges."""
        backbone = model.vggt_backbone.model
        if not hasattr(backbone, "vision_patch_hw"):
            pytest.skip("Not a VGGT backbone with vision layout")

        B, ph, pw = 1, backbone.vision_patch_hw[0], backbone.vision_patch_hw[1]

        cam0_pos = backbone.position_getter(B, ph, pw, device="cpu")
        cam1_pos = backbone.position_getter(B, ph, pw, device="cpu").clone()
        cam1_pos[..., 0] += ph  # same offset logic as forward()

        cam0_y_max = cam0_pos[0, :, 0].max().item()
        cam1_y_min = cam1_pos[0, :, 0].min().item()
        assert cam1_y_min > cam0_y_max, (
            f"Camera y ranges overlap: cam0 max={cam0_y_max}, cam1 min={cam1_y_min}"
        )

    def test_backbone_output_shape(self, model):
        """VGGT backbone should preserve sequence length."""
        H = model.vggt_backbone.hidden_size
        # 512 vision + 128 language = 640 tokens input
        fused = torch.randn(1, 640, H)
        attn_mask = torch.ones(1, 640, dtype=torch.long)
        with torch.no_grad():
            out = model.vggt_backbone(fused, attn_mask)
        assert out.shape == (1, 640, H), f"unexpected shape: {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 7: Action Head
# ═══════════════════════════════════════════════════════════════════════════

class TestActionHead:
    @pytest.fixture(scope="class")
    def action_head(self, model):
        return model.action_head

    def test_compute_loss_shape(self, action_head, model):
        H = model.vggt_backbone.hidden_size
        prefix = torch.randn(2, 100, H)
        state = torch.randn(2, 8)
        actions = torch.randn(2, 10, 7)
        loss = action_head.compute_loss(prefix, state, actions)
        assert loss.ndim == 0, "loss should be scalar"
        assert loss.item() > 0, "loss should be positive"

    def test_sample_shape(self, action_head, model):
        H = model.vggt_backbone.hidden_size
        prefix = torch.randn(1, 100, H)
        state = torch.randn(1, 8)
        out = action_head.sample(prefix, state, num_steps=2)
        assert out.shape == (1, 10, 7)

    def test_gripper_weight_affects_loss(self, action_head, model):
        """gripper_loss_weight > 1 should produce different loss than weight=1."""
        H = model.vggt_backbone.hidden_size
        prefix = torch.randn(2, 50, H)
        state = torch.randn(2, 8)
        actions = torch.randn(2, 10, 7)

        torch.manual_seed(42)
        details = action_head.compute_loss(prefix, state, actions, return_details=True)
        assert "unweighted_loss" in details
        assert "gripper_loss" in details
        if action_head.cfg.gripper_loss_weight != 1.0:
            assert details["loss"].item() != details["unweighted_loss"].item(), (
                "Weighted loss should differ from unweighted when gripper_loss_weight != 1"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 8: End-to-End
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_forward_loss(self, model):
        obs = _make_obs(batch_size=2)
        actions = torch.randn(2, 10, 7)
        with torch.no_grad():
            out = model(obs, actions)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].item() > 0

    def test_forward_inference(self, model):
        obs = _make_obs(batch_size=1)
        with torch.no_grad():
            out = model(obs, actions=None, num_ode_steps=2)
        assert "actions" in out
        assert out["actions"].shape == (1, 10, 7)

    def test_loss_backward_gradients_exist(self, model):
        """Loss should be differentiable — gradients should flow to trainable params."""
        # Need to enable grad for this test
        model.train()
        obs = _make_obs(batch_size=1)
        actions = torch.randn(1, 10, 7)

        out = model(obs, actions)
        out["loss"].backward()

        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received gradients"

        model.eval()
        model.zero_grad()

    def test_different_prompts_different_output(self, model, tokenizer):
        """Different language instructions should produce different losses."""
        prompt_a = ["put the bowl on the plate"]
        prompt_b = ["open the middle drawer of the cabinet"]

        obs_a = _make_obs(batch_size=1, tokenizer=tokenizer, prompts=prompt_a)
        obs_b = _make_obs(batch_size=1, tokenizer=tokenizer, prompts=prompt_b)

        # Use same images/state/actions for both
        obs_b.images = obs_a.images
        obs_b.image_masks = obs_a.image_masks
        obs_b.state = obs_a.state

        actions = torch.randn(1, 10, 7)
        with torch.no_grad():
            loss_a = model(obs_a, actions)["loss"].item()
            loss_b = model(obs_b, actions)["loss"].item()

        assert loss_a != loss_b, (
            f"Same loss for different prompts ({loss_a:.6f}) — language not affecting output"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Normalization round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalization:
    def test_quantile_round_trip(self):
        stats = NormStats(
            mean=np.zeros(7, dtype=np.float32),
            std=np.ones(7, dtype=np.float32),
            q01=np.array([-1, -1, -1, -0.1, -0.1, -0.1, -1], dtype=np.float32),
            q99=np.array([1, 1, 1, 0.1, 0.1, 0.1, 1], dtype=np.float32),
        )
        x = np.array([0.5, -0.3, 0.0, 0.05, -0.05, 0.0, 1.0], dtype=np.float32)
        normed = normalize_quantile(x, stats)
        recovered = unnormalize_quantile(normed, stats)
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_normalized_range(self):
        stats = NormStats(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            q01=np.array([-2, -2, -2], dtype=np.float32),
            q99=np.array([2, 2, 2], dtype=np.float32),
        )
        x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normed = normalize_quantile(x, stats)
        # x=0 is midpoint between q01=-2 and q99=2, so normalized = 0
        np.testing.assert_allclose(normed, [0.0, 0.0, 0.0], atol=1e-5)
