"""Unit tests for srt/sampling/sampling_observer_pp.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.sampling.sampling_observer_pp import (
    add_auxiliary_output_to_pp_tensors,
    pop_auxiliary_output_from_pp_tensors,
)
from sglang.test.test_utils import CustomTestCase

OUTPUT_PREFIX = "__sampling_observer_output__."


class _DeviceOutput:
    """A device-side auxiliary output that supports PP transport."""

    def __init__(self, tensors=None):
        self.tensors = (
            dict(tensors) if tensors is not None else {"logits": torch.ones(2)}
        )

    def copy_to_host(self, copy_tensor):
        return _HostOutput()

    def to_pp_tensors(self):
        return dict(self.tensors)


class _HostOutput:
    def consume(self, batch, commits):
        pass


class _PlainDeviceOutput:
    """A device-side output without PP transport support."""

    def copy_to_host(self, copy_tensor):
        return _HostOutput()


class _PPObserver:
    def __init__(self):
        self.reconstructed = None

    def from_pp_tensors(self, tensors):
        self.reconstructed = dict(tensors)
        return _DeviceOutput(tensors)


class _NullPPObserver:
    """A PP observer whose reconstruction fails (returns None)."""

    def from_pp_tensors(self, tensors):
        return None


class _PlainObserver:
    """A sampling observer without PP transport support."""

    def is_active(self, sampling_info):
        return True

    def before_grammar(self, logits, sampling_info):
        return None

    def after_sample(self, state, token_ids):
        return None


class TestAddAuxiliaryOutputToPpTensors(CustomTestCase):

    def test_none_output_is_noop(self):
        """A None output must leave the tensor mapping untouched."""
        tensors = {"tokens": torch.ones(1)}
        add_auxiliary_output_to_pp_tensors(tensors, None)
        self.assertEqual(set(tensors), {"tokens"})

    def test_non_pp_output_raises(self):
        """An output without to_pp_tensors must be rejected."""
        tensors = {}
        with self.assertRaisesRegex(
            RuntimeError, "does not support pipeline-parallel transport"
        ):
            add_auxiliary_output_to_pp_tensors(tensors, _PlainDeviceOutput())

    def test_empty_output_raises(self):
        """A PP output with no tensors must be rejected."""
        output = _DeviceOutput(tensors={})
        with self.assertRaisesRegex(RuntimeError, "at least one tensor"):
            add_auxiliary_output_to_pp_tensors({}, output)

    def test_empty_name_raises(self):
        """Tensor names must be non-empty strings."""
        output = _DeviceOutput(tensors={"": torch.ones(1)})
        with self.assertRaisesRegex(RuntimeError, "non-empty strings"):
            add_auxiliary_output_to_pp_tensors({}, output)

    def test_non_string_name_raises(self):
        """Tensor names must be strings."""
        output = _DeviceOutput(tensors={1: torch.ones(1)})
        with self.assertRaisesRegex(RuntimeError, "non-empty strings"):
            add_auxiliary_output_to_pp_tensors({}, output)

    def test_non_tensor_value_raises(self):
        """Values must be torch tensors."""
        output = _DeviceOutput(tensors={"logits": [1.0, 2.0]})
        with self.assertRaisesRegex(RuntimeError, "not a tensor"):
            add_auxiliary_output_to_pp_tensors({}, output)

    def test_duplicate_name_raises(self):
        """Re-adding an already present auxiliary tensor must be rejected."""
        tensors = {f"{OUTPUT_PREFIX}logits": torch.ones(1)}
        output = _DeviceOutput(tensors={"logits": torch.ones(2)})
        with self.assertRaisesRegex(RuntimeError, "duplicate auxiliary PP tensor"):
            add_auxiliary_output_to_pp_tensors(tensors, output)

    def test_happy_path_adds_prefixed_tensors(self):
        """Valid outputs are added under the reserved prefix."""
        tensors = {"tokens": torch.ones(1)}
        output = _DeviceOutput(
            tensors={"logits": torch.ones(2), "topk_idx": torch.ones(3)}
        )
        add_auxiliary_output_to_pp_tensors(tensors, output)
        self.assertEqual(
            set(tensors),
            {"tokens", f"{OUTPUT_PREFIX}logits", f"{OUTPUT_PREFIX}topk_idx"},
        )
        self.assertEqual(tensors[f"{OUTPUT_PREFIX}logits"].shape, (2,))
        self.assertEqual(tensors[f"{OUTPUT_PREFIX}topk_idx"].shape, (3,))


class TestPopAuxiliaryOutputFromPpTensors(CustomTestCase):

    def test_no_prefixed_keys_returns_none(self):
        """A mapping without auxiliary keys must return None."""
        tensors = {"tokens": torch.ones(1)}
        self.assertIsNone(
            pop_auxiliary_output_from_pp_tensors(tensors, observer=_PPObserver())
        )
        self.assertEqual(set(tensors), {"tokens"})

    def test_prefixed_keys_without_observer_raises(self):
        """Auxiliary output without an observer must be rejected."""
        tensors = {f"{OUTPUT_PREFIX}logits": torch.ones(1)}
        with self.assertRaisesRegex(RuntimeError, "without a sampling observer"):
            pop_auxiliary_output_from_pp_tensors(tensors, observer=None)

    def test_non_pp_observer_raises(self):
        """An observer without from_pp_tensors must be rejected."""
        tensors = {f"{OUTPUT_PREFIX}logits": torch.ones(1)}
        with self.assertRaisesRegex(RuntimeError, "does not support pipeline-parallel"):
            pop_auxiliary_output_from_pp_tensors(tensors, observer=_PlainObserver())

    def test_non_tensor_value_raises(self):
        """Non-tensor auxiliary values must be rejected."""
        tensors = {f"{OUTPUT_PREFIX}logits": [1.0, 2.0]}
        with self.assertRaisesRegex(RuntimeError, "non-tensor auxiliary PP output"):
            pop_auxiliary_output_from_pp_tensors(tensors, observer=_PPObserver())

    def test_reconstruction_failure_raises(self):
        """An observer that fails to reconstruct must be rejected."""
        tensors = {f"{OUTPUT_PREFIX}logits": torch.ones(1)}
        with self.assertRaisesRegex(RuntimeError, "did not reconstruct"):
            pop_auxiliary_output_from_pp_tensors(tensors, observer=_NullPPObserver())

    def test_happy_path_reconstructs_and_cleans(self):
        """Valid auxiliary output is reconstructed and its keys removed."""
        tensors = {
            "tokens": torch.ones(1),
            f"{OUTPUT_PREFIX}logits": torch.ones(2),
            f"{OUTPUT_PREFIX}topk_idx": torch.ones(3),
        }
        observer = _PPObserver()
        output = pop_auxiliary_output_from_pp_tensors(tensors, observer=observer)

        self.assertIsNotNone(output)
        self.assertEqual(set(output.to_pp_tensors()), {"logits", "topk_idx"})
        self.assertEqual(observer.reconstructed, output.to_pp_tensors())
        # Auxiliary keys must be removed; unrelated keys must be kept.
        self.assertEqual(set(tensors), {"tokens"})


if __name__ == "__main__":
    unittest.main()
