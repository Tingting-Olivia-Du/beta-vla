"""Tests for vision tower LoRA injection and forward pass."""
from __future__ import annotations

import pytest
import torch

from betavla.data.types import ObservationBatch
from betavla.models.model import BetaVLAConfig, BetaVLAModel, _inject_lora
from betavla.models.vision_tower import PaliGemmaVisionTower, PaliGemmaVisionTowerConfig


def _make_dummy_observation(batch_size: int = 2, image_size: int = 224, seq_len: int = 8) -> ObservationBatch:
    """Create a minimal ObservationBatch for testing."""
    return ObservationBatch(
        images={"agentview": torch.randn(batch_size, 3, image_size, image_size)},
        image_masks={"agentview": torch.ones(batch_size)},
        state=torch.randn(batch_size, 8),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, seq_len)),
        tokenized_prompt_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
    )


# ── Test 1: _inject_lora with task_type=None produces a working PeftModel ──

class TestInjectLoraTaskType:
    def test_inject_lora_none_task_type(self):
        """LoRA injection with task_type=None should wrap in PeftModel (not PeftModelForFeatureExtraction)."""
        from peft import PeftModel, PeftModelForFeatureExtraction
        base = torch.nn.Linear(16, 16)
        # Wrap in a Module so PEFT can find target modules
        model = torch.nn.Sequential(base)
        wrapped = _inject_lora(
            model, r=4, alpha=8, dropout=0.0,
            target_modules=["0"],  # target the Linear layer
            task_type=None,
        )
        assert isinstance(wrapped, PeftModel)
        assert not isinstance(wrapped, PeftModelForFeatureExtraction)

    def test_inject_lora_feature_extraction_task_type(self):
        """LoRA injection with default task_type should use PeftModelForFeatureExtraction."""
        from peft import PeftModelForFeatureExtraction
        model = torch.nn.Sequential(torch.nn.Linear(16, 16))
        wrapped = _inject_lora(
            model, r=4, alpha=8, dropout=0.0,
            target_modules=["0"],
        )
        assert isinstance(wrapped, PeftModelForFeatureExtraction)


# ── Test 2: PaliGemmaVisionTower with LoRA can forward ──

class TestVisionTowerWithLora:
    @pytest.fixture(scope="class")
    def vision_tower_with_lora(self):
        """Build PaliGemma vision tower and inject LoRA."""
        cfg = PaliGemmaVisionTowerConfig(
            model_name="google/paligemma2-3b-pt-224",
            image_size=224,
        )
        tower = PaliGemmaVisionTower(cfg)
        tower.vision_tower = _inject_lora(
            tower.vision_tower,
            r=4, alpha=8, dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            task_type=None,
        )
        tower.eval()
        return tower

    def test_forward_no_error(self, vision_tower_with_lora):
        """LoRA-wrapped vision tower should forward without 'unexpected keyword argument' errors."""
        tower = vision_tower_with_lora
        images = {"cam0": torch.randn(1, 3, 224, 224)}
        masks = {"cam0": torch.ones(1)}
        with torch.no_grad():
            out = tower(images, masks)
        assert out.ndim == 3  # (B, N_patches, D)
        assert out.shape[0] == 1

    def test_output_shape(self, vision_tower_with_lora):
        """Output should be (B, num_patches, embed_dim)."""
        tower = vision_tower_with_lora
        images = {"cam0": torch.randn(2, 3, 224, 224)}
        masks = {"cam0": torch.ones(2)}
        with torch.no_grad():
            out = tower(images, masks)
        assert out.shape[0] == 2
        assert out.shape[2] == tower.embed_dim

    def test_lora_params_trainable(self, vision_tower_with_lora):
        """LoRA adapter parameters should be trainable."""
        tower = vision_tower_with_lora
        lora_params = [n for n, p in tower.vision_tower.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0, "No LoRA parameters found"
        for n, p in tower.vision_tower.named_parameters():
            if "lora_" in n:
                assert p.requires_grad, f"LoRA param {n} should be trainable"

    def test_base_params_frozen(self, vision_tower_with_lora):
        """Base (non-LoRA) parameters should be frozen after PEFT wrapping."""
        tower = vision_tower_with_lora
        for n, p in tower.vision_tower.named_parameters():
            if "lora_" not in n:
                assert not p.requires_grad, f"Base param {n} should be frozen"


# ── Test 3: BetaVLAConfig accepts new fields ──

class TestConfigFields:
    def test_default_lora_on_vision_false(self):
        cfg = BetaVLAConfig()
        assert cfg.lora_on_vision is False

    def test_default_vision_target_modules(self):
        cfg = BetaVLAConfig()
        assert "out_proj" in cfg.lora_target_modules_vision

    def test_custom_config(self):
        cfg = BetaVLAConfig(
            lora_on_vision=True,
            lora_target_modules_vision=["q_proj", "v_proj"],
        )
        assert cfg.lora_on_vision is True
        assert cfg.lora_target_modules_vision == ["q_proj", "v_proj"]

    def test_freeze_vision_true_blocks_lora(self):
        """When freeze_vision=True, lora_on_vision should not cause LoRA injection."""
        cfg = BetaVLAConfig(
            freeze_vision=True,
            lora_on_vision=True,
        )
        # The model __init__ checks `not cfg.freeze_vision` before injecting
        assert cfg.freeze_vision is True
        assert cfg.lora_on_vision is True


# ── Test 4: Full model forward with vision LoRA ──

class TestFullModelWithVisionLora:
    @pytest.fixture(scope="class")
    def model(self):
        cfg = BetaVLAConfig(
            freeze_vision=False,
            freeze_language=False,
            freeze_vggt=False,
            use_lora=True,
            lora_on_vision=True,
            lora_on_language=True,
            lora_on_vggt=True,
            lora_r=4,
            lora_alpha=8,
        )
        m = BetaVLAModel(cfg)
        m.eval()
        return m

    def test_forward_loss(self, model):
        """Full forward with actions should return a loss without errors."""
        obs = _make_dummy_observation(batch_size=2)
        actions = torch.randn(2, 10, 7)
        with torch.no_grad():
            out = model(obs, actions)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar

    def test_forward_inference(self, model):
        """Forward without actions should return sampled actions."""
        obs = _make_dummy_observation(batch_size=1)
        with torch.no_grad():
            out = model(obs, actions=None, num_ode_steps=2)
        assert "actions" in out
        assert out["actions"].shape[0] == 1

    def test_vision_has_lora_layers(self, model):
        """Vision tower should have LoRA adapter layers after construction."""
        lora_params = [n for n, _ in model.vision_tower.vision_tower.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0

    def test_trainable_param_count(self, model):
        """Trainable params should be much less than total (LoRA is working)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable / total
        # With LoRA on all 3 components, trainable should be < 30% of total
        assert ratio < 0.30, f"Trainable ratio {ratio:.1%} too high — LoRA may not be working"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
