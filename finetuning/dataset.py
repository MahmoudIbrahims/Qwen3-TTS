from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        # CHANGED: build a case-insensitive lookup for language -> codec ID
        self._lang_map_lower = {
            k.lower(): v for k, v in self.config.talker_config.codec_language_id.items()
        }

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    # CHANGED: new helper -- resolves a language string to its codec ID.
    # Raises a clear error instead of silently ignoring the language, which
    # is what the original code effectively did.
    def _resolve_lang_id(self, language: str) -> int:
        if language is None or language.lower() == "auto":
            raise ValueError(
                "Every training sample must have an explicit 'language' "
                "field matching a key in config.talker_config.codec_language_id "
                f"(available: {list(self.config.talker_config.codec_language_id.keys())}). "
                "'Auto' / missing language is no longer silently accepted, "
                "because the model must see an explicit language token to "
                "learn it -- add a 'language' field to every line of your "
                "training jsonl."
            )
        lang_id = self._lang_map_lower.get(language.lower())
        if lang_id is None:
            raise ValueError(
                f"Language '{language}' not found in "
                f"config.talker_config.codec_language_id. "
                f"Available: {list(self.config.talker_config.codec_language_id.keys())}"
            )
        return lang_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels



    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']

        # CHANGED: resolve language -> codec ID here, so it's ready for collate_fn
        lang_id = self._resolve_lang_id(language)

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:,:-5],    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "ref_mel":ref_mel,
            "lang_id": lang_id,             # CHANGED: now carried through to collate_fn
        }
        
    def collate_fn(self, batch):
        assert self.lag_num == -1

        # CHANGED: overhead grew from 8 to 9 positions because the codec
        # preamble now carries an explicit lang_id token (6 tokens instead
        # of 5): [think_id, think_bos_id, lang_id, think_eos_id, speaker(0), pad_id]
        PREFIX = 9  # was 8

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + PREFIX
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']
            lang_id         = data['lang_id']  # CHANGED

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:PREFIX-1, 0] = self.config.tts_pad_token_id
            input_ids[i,   PREFIX-1, 0] = self.config.tts_bos_token_id
            input_ids[i, PREFIX:PREFIX+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   PREFIX+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, PREFIX+text_ids_len-2:PREFIX+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :PREFIX+text_ids_len+codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            # CHANGED: 6-token preamble (was 5) carrying the explicit lang_id.
            # Uses codec_think_id (was codec_nothink_id) since a language IS
            # being specified now.
            input_ids[i,    3:PREFIX ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_think_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            lang_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id
                                        ]
                                    )
            input_ids[i,    PREFIX:PREFIX+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    PREFIX+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    PREFIX+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    PREFIX+text_ids_len-1:PREFIX+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    PREFIX+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    PREFIX+text_ids_len-1:PREFIX+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    PREFIX+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, PREFIX+text_ids_len-1:PREFIX+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:PREFIX+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, PREFIX-2] = False       # CHANGED: speaker slot shifted from 6 to PREFIX-2 (=7)

            codec_mask[i,   PREFIX+text_ids_len-1:PREFIX+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :PREFIX+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels,dim=0)

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask
        }