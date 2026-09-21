# # coding=utf-8
# # Copyright 2026 The Alibaba Qwen team.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # ============================================================================
# # MODIFIED: dataset.py's collate_fn now builds a 6-token codec preamble
# # (was 5) to carry an explicit lang_id, which shifted the speaker-embedding
# # slot from position 6 to position 7 (see codec_embedding_mask[i, PREFIX-2]
# # in the modified dataset.py). This script's speaker-embedding injection
# # below is updated to match (was `[:, 6, :]`, now `[:, 7, :]`).
# # ============================================================================
# import argparse
# import json
# import os
# import shutil

# import torch
# from accelerate import Accelerator
# from dataset import TTSDataset
# from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
# from safetensors.torch import save_file
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import AutoConfig

# # CHANGED: single source of truth for the speaker-embedding slot position,
# # must always match PREFIX-2 in dataset.py's collate_fn (PREFIX=9 -> slot=7).
# SPEAKER_EMBEDDING_SLOT = 7  # was 6 before language conditioning was added

# target_speaker_embedding = None
# def train():
#     global target_speaker_embedding

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
#     parser.add_argument("--output_model_path", type=str, default="output")
#     parser.add_argument("--train_jsonl", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--lr", type=float, default=2e-5)
#     parser.add_argument("--num_epochs", type=int, default=3)
#     parser.add_argument("--speaker_name", type=str, default="speaker_test")
#     args = parser.parse_args()

#     # accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

#     accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")


#     MODEL_PATH = args.init_model_path

#     qwen3tts = Qwen3TTSModel.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#     )
#     config = AutoConfig.from_pretrained(MODEL_PATH)

#     train_data = open(args.train_jsonl).readlines()
#     train_data = [json.loads(line) for line in train_data]
#     dataset = TTSDataset(train_data, qwen3tts.processor, config)
#     train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

#     optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

#     model, optimizer, train_dataloader = accelerator.prepare(
#         qwen3tts.model, optimizer, train_dataloader
#     )

#     num_epochs = args.num_epochs
#     model.train()

#     for epoch in range(num_epochs):
#         for step, batch in enumerate(train_dataloader):
#             with accelerator.accumulate(model):

#                 input_ids = batch['input_ids']
#                 codec_ids = batch['codec_ids']
#                 ref_mels = batch['ref_mels']
#                 text_embedding_mask = batch['text_embedding_mask']
#                 codec_embedding_mask = batch['codec_embedding_mask']
#                 attention_mask = batch['attention_mask']
#                 codec_0_labels = batch['codec_0_labels']
#                 codec_mask = batch['codec_mask']

#                 speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
#                 if target_speaker_embedding is None:
#                     target_speaker_embedding = speaker_embedding

#                 input_text_ids = input_ids[:, :, 0]
#                 input_codec_ids = input_ids[:, :, 1]

#                 input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
#                 input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
#                 # CHANGED: was `input_codec_embedding[:, 6, :] = speaker_embedding`
#                 # Must match dataset.py's shifted speaker slot (PREFIX-2 = 7).
#                 input_codec_embedding[:, SPEAKER_EMBEDDING_SLOT, :] = speaker_embedding

#                 input_embeddings = input_text_embedding + input_codec_embedding

#                 for i in range(1, 16):
#                     codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
#                     codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
#                     input_embeddings = input_embeddings + codec_i_embedding

#                 outputs = model.talker(
#                     inputs_embeds=input_embeddings[:, :-1, :],
#                     attention_mask=attention_mask[:, :-1],
#                     labels=codec_0_labels[:, 1:],
#                     output_hidden_states=True
#                 )

#                 hidden_states = outputs.hidden_states[0][-1]
#                 talker_hidden_states = hidden_states[codec_mask[:, :-1]]
#                 talker_codec_ids = codec_ids[codec_mask]

#                 sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

#                 loss = outputs.loss + 0.3 * sub_talker_loss

#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(model.parameters(), 1.0)

#                 optimizer.step()
#                 optimizer.zero_grad()

#             if step % 10 == 0:
#                 accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

#         if accelerator.is_main_process:
#             output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
#             shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

#             input_config_file = os.path.join(MODEL_PATH, "config.json")
#             output_config_file = os.path.join(output_dir, "config.json")
#             with open(input_config_file, 'r', encoding='utf-8') as f:
#                 config_dict = json.load(f)
#             config_dict["tts_model_type"] = "custom_voice"
#             talker_config = config_dict.get("talker_config", {})
#             talker_config["spk_id"] = {
#                 args.speaker_name: 3000
#             }
#             talker_config["spk_is_dialect"] = {
#                 args.speaker_name: False
#             }
#             config_dict["talker_config"] = talker_config

#             with open(output_config_file, 'w', encoding='utf-8') as f:
#                 json.dump(config_dict, f, indent=2, ensure_ascii=False)

#             unwrapped_model = accelerator.unwrap_model(model)
#             state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

#             drop_prefix = "speaker_encoder"
#             keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
#             for k in keys_to_drop:
#                 del state_dict[k]

#             weight = state_dict['talker.model.codec_embedding.weight']
#             state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
#             save_path = os.path.join(output_dir, "model.safetensors")
#             save_file(state_dict, save_path)

# if __name__ == "__main__":
#     train()

#--------------------------------------------------

# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
#
# ============================================================================
# MODIFIED (see "# CHANGED" markers):
#   1. Speaker-embedding injection slot: 6 -> 7 (dataset.py now carries an
#      explicit lang_id token, shifting the speaker slot by one position).
#   2. MULTI-SPEAKER SUPPORT: the original script captured exactly ONE
#      reference speaker embedding for the whole run (`target_speaker_embedding`,
#      a single global variable) and always baked it into a hardcoded ID
#      (3000), and always OVERWROTE spk_id / spk_is_dialect in the saved
#      config rather than merging into it. That meant:
#        - training on two speakers in one run only ever "remembered" one
#          of them as a named custom voice
#        - resuming/continuing from a checkpoint that already had a named
#          speaker would erase it
#      Now: one embedding snapshot is captured PER unique speaker name
#      (from the training jsonl's "speaker" field), each gets its own
#      stable ID (starting at 3000, or continuing after whatever IDs
#      already exist in the init model's config.json), and existing
#      spk_id / spk_is_dialect entries are preserved and merged with new
#      ones instead of being replaced.
#
#   Your train_with_codes.jsonl must now include a "speaker" field per line
#   (e.g. "speaker": "mahmoud" or "speaker": "sara") -- see dataset.py.
# ============================================================================

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# CHANGED: must match dataset.py's PREFIX-2 (PREFIX=9 -> slot=7)
SPEAKER_EMBEDDING_SLOT = 7  # was 6 before language conditioning was added

# CHANGED: was a single Tensor (`target_speaker_embedding = None`).
# Now a dict: {speaker_name: embedding_tensor}, one entry per unique
# speaker encountered during training, captured the first time each name
# is seen.
target_speaker_embeddings = {}


def assign_speaker_ids(existing_spk_id: dict, speaker_names):
    """
    Returns a merged {speaker_name: id} dict: keeps every ID already
    present in existing_spk_id (so previously-trained speakers are never
    renumbered or dropped), and assigns fresh, unused IDs (starting at
    3000) to any speaker_names not already present.
    """
    merged = dict(existing_spk_id)  # CHANGED: start from existing, don't overwrite
    used_ids = set(merged.values())
    next_id = 3000
    for name in speaker_names:
        if name in merged:
            continue
        while next_id in used_ids:
            next_id += 1
        merged[name] = next_id
        used_ids.add(next_id)
        next_id += 1
    return merged


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    # CHANGED: --speaker_name is no longer used -- speaker identity now
    # comes from the "speaker" field in each training jsonl line, since a
    # single run can (and should, to avoid catastrophic forgetting) train
    # multiple speakers together. Kept as an accepted-but-ignored arg only
    # so old launch commands don't immediately break; remove it once your
    # scripts are updated.
    parser.add_argument("--speaker_name", type=str, default=None,
                         help="DEPRECATED / ignored: put a 'speaker' field "
                              "in each line of --train_jsonl instead.")
    args = parser.parse_args()
    if args.speaker_name:
        print("[warn] --speaker_name is ignored. Speaker identity is now "
              "read per-sample from the 'speaker' field in train_jsonl.")

    # accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")


    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                speaker_names = batch['speaker_names']  # CHANGED: list[str], len == batch size

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                # CHANGED: snapshot ONE embedding per unique speaker name,
                # the first time each name is encountered anywhere in
                # training (instead of only ever remembering the very
                # first sample of the very first batch).
                for i, name in enumerate(speaker_names):
                    if name not in target_speaker_embeddings:
                        target_speaker_embeddings[name] = speaker_embedding[i:i+1].clone()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                # CHANGED: was `input_codec_embedding[:, 6, :] = speaker_embedding`
                input_codec_embedding[:, SPEAKER_EMBEDDING_SLOT, :] = speaker_embedding

                # input_embeddings = input_text_embedding + input_codec_embedding
                input_embeddings = model.talker.text_projection(input_text_embedding) + input_codec_embedding


                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | "
                                   f"Speakers seen so far: {list(target_speaker_embeddings.keys())}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})

            # CHANGED: merge with whatever speakers already exist in the
            # init model's config (e.g. from a previous training run),
            # instead of overwriting spk_id / spk_is_dialect wholesale.
            existing_spk_id = talker_config.get("spk_id", {})
            speaker_id_map = assign_speaker_ids(existing_spk_id, target_speaker_embeddings.keys())

            existing_spk_is_dialect = talker_config.get("spk_is_dialect", {})
            spk_is_dialect = dict(existing_spk_is_dialect)
            for name in speaker_id_map:
                spk_is_dialect.setdefault(name, False)

            talker_config["spk_id"] = speaker_id_map
            talker_config["spk_is_dialect"] = spk_is_dialect
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            # CHANGED: inject EVERY captured speaker embedding at its own
            # assigned ID (was: only ID 3000, only one speaker).
            weight = state_dict['talker.model.codec_embedding.weight']
            for name, embedding in target_speaker_embeddings.items():
                spk_id = speaker_id_map[name]
                if spk_id >= weight.shape[0]:
                    raise ValueError(
                        f"Speaker ID {spk_id} for '{name}' is out of range for "
                        f"codec_embedding of size {weight.shape[0]}. You'll need "
                        f"to resize the embedding table (same approach as "
                        f"prepare_lang.py) before assigning IDs this high."
                    )
                state_dict['talker.model.codec_embedding.weight'][spk_id] = (
                    embedding[0].detach().to(weight.device).to(weight.dtype)
                )

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

            accelerator.print(f"[info] Saved checkpoint with speakers: {speaker_id_map}")

if __name__ == "__main__":
    train()