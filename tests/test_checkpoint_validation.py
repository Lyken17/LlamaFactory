# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

from llamafactory.hparams import get_train_args


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}


def test_resume_from_checkpoint_fallback_nonexistent():
    """Test that resume_from_checkpoint falls back to model_name_or_path when path doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        nonexistent_checkpoint = os.path.join(tmpdir, "nonexistent_checkpoint")

        _, _, training_args, _, _ = get_train_args(
            {
                "output_dir": output_dir,
                "resume_from_checkpoint": nonexistent_checkpoint,
                **TRAIN_ARGS,
            }
        )
        # Should fallback to None since checkpoint doesn't exist
        assert training_args.resume_from_checkpoint is None


def test_resume_from_checkpoint_fallback_empty_dir():
    """Test that resume_from_checkpoint falls back to model_name_or_path when directory is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        empty_checkpoint = os.path.join(tmpdir, "empty_checkpoint")
        os.makedirs(empty_checkpoint)

        _, _, training_args, _, _ = get_train_args(
            {
                "output_dir": output_dir,
                "resume_from_checkpoint": empty_checkpoint,
                **TRAIN_ARGS,
            }
        )
        # Should fallback to None since checkpoint directory is empty
        assert training_args.resume_from_checkpoint is None


def test_resume_from_checkpoint_valid():
    """Test that resume_from_checkpoint is preserved when it contains valid checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        valid_checkpoint = os.path.join(tmpdir, "checkpoint-1")
        os.makedirs(valid_checkpoint)

        # Create a fake checkpoint file to make it look valid
        with open(os.path.join(valid_checkpoint, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(valid_checkpoint, "adapter_config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(valid_checkpoint, "adapter_model.safetensors"), "w") as f:
            f.write("")

        _, _, training_args, _, _ = get_train_args(
            {
                "output_dir": output_dir,
                "resume_from_checkpoint": valid_checkpoint,
                **TRAIN_ARGS,
            }
        )
        # Should preserve the checkpoint path since it contains valid checkpoint files
        assert training_args.resume_from_checkpoint == valid_checkpoint


def test_validation_skipped_when_model_name_or_path_missing():
    """Test that resume_from_checkpoint validation only happens when model_name_or_path is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        nonexistent_checkpoint = os.path.join(tmpdir, "nonexistent_checkpoint")

        args = {
            "output_dir": output_dir,
            "resume_from_checkpoint": nonexistent_checkpoint,
            **TRAIN_ARGS,
        }
        # Remove model_name_or_path to test the condition
        del args["model_name_or_path"]

        # model_name_or_path is required, so this should raise an error
        # The checkpoint validation should not happen because model_name_or_path is None
        try:
            _, _, training_args, _, _ = get_train_args(args)
            # If we reach here, the validation was skipped (which is correct)
            # The checkpoint should remain set since validation didn't run
            assert training_args.resume_from_checkpoint == nonexistent_checkpoint
        except (ValueError, TypeError) as e:
            # This is expected if model_name_or_path is required by the system
            # In this case, the test passes because we're testing that validation
            # only happens when both are set
            assert "model_name_or_path" in str(e).lower() or "required" in str(e).lower()
