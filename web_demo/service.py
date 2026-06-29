from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import librosa
import numpy as np
import scipy.io.wavfile as wav
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
AMN_CODE_DIR = ROOT_DIR / "amn_code"
if str(AMN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AMN_CODE_DIR))

from exp_setup import get_speakers_and_wm
from models import ModelTrainer
from models.WatermarkNet import WatermarkNet
from my_utils import utils
from my_utils.datasets import MyNumpyAudioData


class WatermarkService:
    def __init__(self) -> None:
        self.root_dir = ROOT_DIR
        self.storage_dir = self.root_dir / "web_demo" / "storage"
        self.upload_dir = self.storage_dir / "uploads"
        self.output_dir = self.storage_dir / "outputs"
        self.model_packages_dir = self.storage_dir / "model_packages"
        self.ckpt_dir_candidates = [
            self.root_dir / "save" / "ExpEmbedWatermark" / "ckpt",
            self.root_dir / "out" / "ExpEmbedWatermark" / "ckpt",
        ]
        self.wav2vec2_dir = self.root_dir / "pretrained_models" / "wav2vec2"

        self.sr = 16000
        self.audio_sec_len = 16000
        self.wm_length = 16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._load_lock = threading.Lock()
        self._model_cache: dict[str, WatermarkNet] = {}
        self._benign_encoded_wm = self._init_benign_wm()

        utils.device = self.device
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_packages_dir.mkdir(parents=True, exist_ok=True)

    def _init_benign_wm(self):
        cfg = SimpleNamespace(wm_length=self.wm_length)
        _, _, benign_encoded_wm = get_speakers_and_wm(cfg, self.wm_length)
        return benign_encoded_wm

    def _find_base_ckpt_path(self) -> Path:
        for candidate in self.ckpt_dir_candidates:
            if not candidate.exists():
                continue
            ckpt_path, _ = ModelTrainer.ModelTrainer.get_latest_ckpt_path(candidate)
            if ckpt_path is not None:
                return ckpt_path
        raise FileNotFoundError(
            "找不到 ExpEmbedWatermark 的 checkpoint，请确认 save/ExpEmbedWatermark/ckpt 或 out/ExpEmbedWatermark/ckpt 存在"
        )

    def _resolve_ckpt_path(self, model_rel_path: str | None = None) -> Path:
        if model_rel_path:
            ckpt_path = (self.storage_dir / model_rel_path).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")
            return ckpt_path
        return self._find_base_ckpt_path()

    def _load_model_from_ckpt(self, ckpt_path: Path) -> WatermarkNet:
        cache_key = str(ckpt_path.resolve())
        with self._load_lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached

            model = WatermarkNet(
                self._benign_encoded_wm,
                self.sr,
                self.audio_sec_len,
                wav2vec2_dir=str(self.wav2vec2_dir),
                aug_normal_prob=0.0,
                aug_normal_scale=0.0,
            ).to(self.device)

            saved = ModelTrainer.ModelTrainer.load_ckpt(ckpt_path)
            model.load_state_dict(saved["model_state"], strict=False)
            model.eval()
            self._model_cache[cache_key] = model
            return model

    def _get_model(self, model_rel_path: str | None = None) -> WatermarkNet:
        ckpt_path = self._resolve_ckpt_path(model_rel_path)
        return self._load_model_from_ckpt(ckpt_path)

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        try:
            audio, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
            return audio.astype(np.float32)
        except Exception:
            audio_np, sr = utils.read_audio(audio_path, None)
            if sr != self.sr:
                audio_np = utils.resample_wav(audio_np, sr, self.sr)
            return np.asarray(audio_np, dtype=np.float32)

    def _parse_code(self, code: str) -> np.ndarray:
        compact = "".join(ch for ch in code.strip() if ch in "01")
        if len(compact) != self.wm_length:
            raise ValueError(f"水印编码长度必须是 {self.wm_length} 位 0/1")
        return np.fromiter((int(ch) for ch in compact), dtype=np.int64)

    @staticmethod
    def _safe_name(name: str) -> str:
        compact = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
        compact = compact.strip("_")
        return compact or f"watermark_{uuid4().hex[:8]}"

    @staticmethod
    def _save_wav(audio: np.ndarray, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        wav.write(out_path, 16000, (audio * 32767.0).astype(np.int16))

    @staticmethod
    def _chunk_audio(waveform: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        if len(waveform) <= chunk_size:
            return [waveform]
        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + chunk_size, len(waveform))
            chunk = waveform[start:end]
            if len(chunk) < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            chunks.append(chunk.astype(np.float32))
            start += chunk_size
        return chunks

    def _build_training_pairs(self, waveform: np.ndarray, code: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        chunks = self._chunk_audio(waveform, self.audio_sec_len)
        wav_lst = [chunk.astype(np.float32) for chunk in chunks]
        wm_lst = [code.astype(np.int64) for _ in chunks]
        return wav_lst, wm_lst

    def _decode_audio_with_model(self, model: WatermarkNet, waveform: np.ndarray) -> dict:
        wav_secs = model.split_waveform(waveform)
        with torch.inference_mode():
            logits = model.decoder(
                model.spec_for_classificiation(wav_secs)[..., model.min_freq_idx:model.max_freq_idx, :]
            )

        bit_matrix = (logits > 0.0).long().cpu().numpy().astype(int)
        vote_ratio = bit_matrix.mean(axis=0)
        vote_bits = (vote_ratio >= 0.5).astype(int)
        vote_code = ''.join(str(int(bit)) for bit in vote_bits.tolist())
        agreement = float((bit_matrix == vote_bits).mean())

        return {
            'segment_count': int(bit_matrix.shape[0]),
            'detected_code': vote_code,
            'confidence': round(float(vote_ratio.mean()), 4),
            'bit_agreement': round(agreement, 4),
            'section_codes': [''.join(str(bit) for bit in row.tolist()) for row in bit_matrix],
        }

    def train_and_save_model_package(self, name: str, code: str, waveform: np.ndarray) -> dict:
        package_dir = self.model_packages_dir / self._safe_name(name)
        if package_dir.exists():
            shutil.rmtree(package_dir)
        ckpt_dir = package_dir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        code_arr = self._parse_code(code)
        wav_lst, wm_lst = self._build_training_pairs(waveform, code_arr)
        dset = MyNumpyAudioData(wav_lst, wm_lst, self.audio_sec_len)

        wm_net = WatermarkNet(
            benign_wm=self._benign_encoded_wm,
            sr=self.sr,
            audio_sec_len=self.audio_sec_len,
            wav2vec2_dir=str(self.wav2vec2_dir),
            aug_normal_prob=0.4,
            aug_normal_scale=0.04,
        ).to(self.device)
        wm_net.MIN_SPEECH_SPEC_POWER = 0

        trainer_cfg = ModelTrainer.ModelTrainerConfig(
            batch_size=8,
            test_batch_size=8,
            num_workers=0,
            loss_func=WatermarkNet.wm_loss,
            loss_other_data=SimpleNamespace(config=SimpleNamespace(tts_recon_loss_weight=0.5)),
            is_classifier=False,
            max_epochs=50,
            lr=1e-4,
            lr_gamma=1.0,
            lr_step_size=[999],
            ckpt_dir=ckpt_dir,
            save_every=50,
            best_ckpt_dir=ckpt_dir,
            best_skip_epochs=0,
        )

        progress_steps = []

        def train_progress(epoch, train_loss, test_loss, _cb_data):
            progress_steps.append({
                "epoch": int(epoch),
                "total": int(trainer_cfg.max_epochs),
                "progress": round(epoch / trainer_cfg.max_epochs, 4),
                "train_loss": round(float(train_loss), 6) if train_loss is not None else None,
                "test_loss": round(float(test_loss), 6) if test_loss is not None else None,
            })

        def _save_compact_ckpt(self, cur_epoch, optimizer, scheduler, train_status_dic, ckpt_dir=None):
            target_dir = Path(ckpt_dir or self.config.ckpt_dir or self.config.best_ckpt_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            state_dict = self.model.state_dict()
            compact_state = {k: v for k, v in state_dict.items() if not k.startswith('encoder.wav2vec2.')}
            payload = {
                "cur_epoch": cur_epoch,
                "model_state": compact_state,
                "optimizer_state": {},
                "scheduler_state": {},
                "train_status_dic": train_status_dic,
            }
            save_path = target_dir / f"saved_epoch-{cur_epoch}.tar"
            torch.save(payload, save_path)

        trainer_cfg.train_epoch_callback = train_progress
        trainer = ModelTrainer.ModelTrainer(model=wm_net, train_set=dset, test_set=dset, config=trainer_cfg)
        trainer.save_ckpt = _save_compact_ckpt.__get__(trainer, ModelTrainer.ModelTrainer)
        trainer.run()

        final_ckpt, epoch_saved = ModelTrainer.ModelTrainer.get_latest_ckpt_path(ckpt_dir)
        if final_ckpt is None:
            epoch_saved = trainer_cfg.max_epochs
            final_ckpt = ckpt_dir / f"saved_epoch-{epoch_saved}.tar"
            torch.save(
                {
                    "cur_epoch": epoch_saved,
                    "model_state": {k: v for k, v in wm_net.state_dict().items() if not k.startswith('encoder.wav2vec2.')},
                    "optimizer_state": {},
                    "scheduler_state": {},
                    "train_status_dic": {},
                },
                final_ckpt,
            )

        metadata = {
            "name": name,
            "code": code,
            "saved_ckpt": str(final_ckpt),
            "wm_length": self.wm_length,
            "sr": self.sr,
            "train_samples": len(wav_lst),
        }
        metadata_path = package_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "package_dir": package_dir,
            "model_rel_path": str(final_ckpt.relative_to(self.storage_dir)),
            "metadata_rel_path": str(metadata_path.relative_to(self.storage_dir)),
            "progress": progress_steps,
        }

    def clear_model_packages(self) -> None:
        if self.model_packages_dir.exists():
            shutil.rmtree(self.model_packages_dir)
        self.model_packages_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache.clear()

    def embed(self, audio_path: Path, code: str, model_rel_path: str | None = None) -> dict:
        model = self._get_model(model_rel_path)
        waveform = self._load_audio(audio_path)

        tgt_wm = self._parse_code(code)
        with torch.inference_mode():
            acc_wm, acc_org, wm_waveform = model.test_wm_capability(waveform, tgt_wm)

        best_audio = wm_waveform.detach().cpu().numpy().reshape(-1)[: len(waveform)]
        verification = self._decode_audio_with_model(model, best_audio)
        return {
            'input_duration': round(len(waveform) / self.sr, 3),
            'segment_count': int(np.ceil(len(waveform) / self.audio_sec_len)),
            'watermark_code': ''.join(str(int(x)) for x in tgt_wm.tolist()),
            'bit_accuracy': round(float(acc_wm), 4),
            'clean_accuracy': round(float(acc_org), 4),
            'verification_code': verification['detected_code'],
            'verification_match': verification['detected_code'] == code,
            'audio': best_audio,
        }

    def _decode_waveform(self, waveform: np.ndarray, model_rel_path: str | None = None) -> dict:
        model = self._get_model(model_rel_path)
        wav_secs = model.split_waveform(waveform)
        with torch.inference_mode():
            logits = model.decoder(
                model.spec_for_classificiation(wav_secs)[..., model.min_freq_idx:model.max_freq_idx, :]
            )

        bit_matrix = (logits > 0.0).long().cpu().numpy().astype(int)
        codes = ["".join(str(bit) for bit in row.tolist()) for row in bit_matrix]
        vote_ratio = bit_matrix.mean(axis=0)
        vote_bits = (vote_ratio >= 0.5).astype(int)
        vote_code = "".join(str(int(bit)) for bit in vote_bits.tolist())
        agreement = float((bit_matrix == vote_bits).mean())

        return {
            "segment_count": int(len(codes)),
            "detected_code": vote_code,
            "confidence": round(float(vote_ratio.mean()), 4),
            "bit_agreement": round(agreement, 4),
            "section_codes": codes,
        }

    def detect_against_saved_models(self, audio_path: Path, registrations: list) -> dict:
        waveform = self._load_audio(audio_path)
        if len(waveform) < self.audio_sec_len:
            raise ValueError("音频至少需要 1 秒")

        best_match = None
        best_result = None
        evaluations = []
        fallback_result = None

        for registration in registrations:
            model_rel_path = getattr(registration, "model_rel_path", None)
            if not model_rel_path:
                continue

            result = self._decode_waveform(waveform, model_rel_path=model_rel_path)
            if fallback_result is None:
                fallback_result = result

            is_match = result["detected_code"] == registration.code
            evaluations.append({
                "name": registration.name,
                "code": registration.code,
                "model_rel_path": model_rel_path,
                "decoded_code": result["detected_code"],
                "match": is_match,
                "confidence": result["confidence"],
                "bit_agreement": result["bit_agreement"],
            })

            if is_match:
                if best_result is None or (result["bit_agreement"], result["confidence"]) > (
                    best_result["bit_agreement"],
                    best_result["confidence"],
                ):
                    best_match = registration
                    best_result = result

        if best_match is not None and best_result is not None:
            return {
                "matched": True,
                "detected_name": best_match.name,
                "detected_code": best_match.code,
                "model_rel_path": best_match.model_rel_path,
                "confidence": best_result["confidence"],
                "bit_agreement": best_result["bit_agreement"],
                "input_duration": round(len(waveform) / self.sr, 3),
                "segment_count": best_result["segment_count"],
                "evaluations": evaluations,
            }

        return {
            "matched": False,
            "detected_name": "无水印",
            "detected_code": None,
            "raw_detected_code": fallback_result["detected_code"] if fallback_result else None,
            "confidence": fallback_result["confidence"] if fallback_result else 0.0,
            "bit_agreement": fallback_result["bit_agreement"] if fallback_result else 0.0,
            "input_duration": round(len(waveform) / self.sr, 3),
            "segment_count": fallback_result["segment_count"] if fallback_result else 0,
            "evaluations": evaluations,
        }
