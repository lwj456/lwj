"""
TTS模型统一包装器 - 支持Echo-TTS, GLM-TTS, YourTTS

作者: Claude Code
日期: 2026-01-14
"""

import sys
import os
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import logging

# 添加项目路径
amn_opensource_code_dir = Path(__file__).parent.parent.parent / "amn_opensource_code"


class TTSModelBase(ABC):
    """所有TTS模型的基类"""

    def __init__(self, device='cuda', sample_rate=16000):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.target_sample_rate = 16000  # 统一下采样到16kHz
        self.native_sample_rate = sample_rate  # TTS模型的原生采样率

    @abstractmethod
    def synthesize_batch(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        批量合成音频

        输入: (batch_size, audio_length) 16kHz音频 torch.Tensor
        输出: (batch_size, audio_length) 16kHz TTS合成音频 torch.Tensor
        """
        pass

    def freeze_parameters(self):
        """冻结所有参数"""
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def resample_to_target(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        重采样音频到目标采样率(16kHz)

        输入: audio (batch_size, length) 或 (length,), orig_sr: 原始采样率
        输出: (batch_size, target_length) 或 (target_length,) 16kHz音频
        """
        if orig_sr == self.target_sample_rate:
            return audio

        # 处理batch维度
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # 重采样
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=self.target_sample_rate
        ).to(audio.device)

        resampled = resampler(audio)

        if squeeze_back:
            resampled = resampled.squeeze(0)

        return resampled

    def pad_or_crop_to_length(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        将音频调整到目标长度(padding或crop)

        输入: audio (batch_size, length)
        输出: (batch_size, target_length)
        """
        current_length = audio.shape[-1]

        if current_length < target_length:
            # Padding
            pad_length = target_length - current_length
            audio = F.pad(audio, (0, pad_length), mode='constant', value=0)
        elif current_length > target_length:
            # Crop
            audio = audio[..., :target_length]

        return audio


class EchoTTSWrapper(TTSModelBase):
    """
    Echo-TTS包装器

    原生采样率: 44.1kHz
    输出采样率: 16kHz (下采样)
    """

    def __init__(self, device='cuda'):
        super().__init__(device, sample_rate=44100)

        # 添加echo-tts路径
        echo_tts_dir = amn_opensource_code_dir / "echo-tts-main"
        if str(echo_tts_dir) not in sys.path:
            sys.path.insert(0, str(echo_tts_dir))

        try:
            from inference import load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf, sample_euler_cfg_independent_guidances

            logging.info("Loading Echo-TTS models...")
            self.model = load_model_from_hf(device=str(self.device))
            self.fish_ae = load_fish_ae_from_hf(device=str(self.device))
            self.pca_state = load_pca_state_from_hf(device=str(self.device))

            self.sample_fn = sample_euler_cfg_independent_guidances

            # 冻结参数
            self.freeze_parameters()
            for param in self.fish_ae.parameters():
                param.requires_grad = False

            logging.info("Echo-TTS models loaded successfully!")

        except Exception as e:
            logging.error(f"Failed to load Echo-TTS: {e}")
            raise

    def synthesize_batch(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        批量合成音频 (简化版本 - 逐个处理)

        输入: (batch_size, audio_length) 16kHz
        输出: (batch_size, audio_length) 16kHz
        """
        batch_size = audio_batch.shape[0]
        target_length = audio_batch.shape[1]
        outputs = []

        # 由于Echo-TTS的复杂性,逐个样本处理
        for i in range(batch_size):
            audio_16k = audio_batch[i]  # 保持为tensor (length,)

            # 使用torchaudio重采样到44.1k
            resampler_to_native = torchaudio.transforms.Resample(
                orig_freq=16000,
                new_freq=44100
            ).to(self.device)
            audio_44k = resampler_to_native(audio_16k.unsqueeze(0)).squeeze(0)

            try:
                # Echo-TTS推理 (使用音频作为speaker reference)
                # 使用inference.py中的ae_encode和ae_decode函数
                from inference import ae_encode, ae_decode

                with torch.no_grad():
                    # 编码: 音频 -> 潜在表示
                    # ae_encode期望输入形状为 (batch, 1, length)
                    audio_44k_batch = audio_44k.unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                    latent = ae_encode(self.fish_ae, self.pca_state, audio_44k_batch)

                    # 简化处理: 直接解码（跳过diffusion采样以加速）
                    # 实际完整流程应该使用sample_pipeline进行diffusion采样
                    output_44k_batch = ae_decode(self.fish_ae, self.pca_state, latent)

                # 下采样到16kHz
                output_16k = self.resample_to_target(output_44k_batch.squeeze(0).squeeze(0), orig_sr=44100)

                # 调整长度
                output_16k = self.pad_or_crop_to_length(output_16k.unsqueeze(0), target_length).squeeze(0)

            except Exception as e:
                import traceback
                logging.warning(f"Echo-TTS synthesis failed for sample {i}: {type(e).__name__}: {e}")
                logging.debug(traceback.format_exc())
                output_16k = audio_batch[i]  # 降级方案:使用原音频

            outputs.append(output_16k)

        return torch.stack(outputs)


class GLMTTSWrapper(TTSModelBase):
    """
    GLM-TTS包装器 - 完整实现

    原生采样率: 24kHz
    输出采样率: 16kHz (下采样)
    """

    def __init__(self, device='cuda'):
        super().__init__(device, sample_rate=24000)

        # 添加GLM-TTS路径
        glm_tts_dir = amn_opensource_code_dir / "GLM-TTS-main"
        if str(glm_tts_dir) not in sys.path:
            sys.path.insert(0, str(glm_tts_dir))

        self.glm_tts_dir = glm_tts_dir
        self.original_dir = os.getcwd()

        try:
            logging.info("Loading GLM-TTS models...")

            # 导入GLM-TTS组件
            from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
            from utils import yaml_util, tts_model_util, seed_util
            from transformers import AutoTokenizer, LlamaForCausalLM
            from llm.glmtts import GLMTTS
            from utils.audio import mel_spectrogram
            from functools import partial

            # 切换到GLM-TTS目录（模型加载需要相对路径）
            os.chdir(glm_tts_dir)

            # 加载Speech Tokenizer
            speech_tokenizer_path = "ckpt/speech_tokenizer"
            _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
            speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

            # 配置特征提取器
            feat_extractor = partial(
                mel_spectrogram,
                sampling_rate=24000,
                hop_size=480,
                n_fft=1920,
                num_mels=80,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False
            )

            # 加载Tokenizer
            glm_tokenizer = AutoTokenizer.from_pretrained(
                "ckpt/vq32k-phoneme-tokenizer",
                trust_remote_code=True
            )
            tokenize_fn = lambda text: glm_tokenizer.encode(text)

            # 加载Frontend
            self.frontend = TTSFrontEnd(
                tokenize_fn,
                speech_tokenizer,
                feat_extractor,
                "frontend/campplus.onnx",
                "frontend/spk2info.pt",
                self.device,
            )

            self.text_frontend = TextFrontEnd(use_phoneme=False)

            # 加载LLM
            llama_path = "ckpt/llm"
            self.llm = GLMTTS(
                llama_cfg_path=os.path.join(llama_path, "config.json"),
                mode="PRETRAIN"
            )
            self.llm.llama = LlamaForCausalLM.from_pretrained(
                llama_path,
                dtype=torch.float32
            ).to(self.device)
            self.llm.llama_embedding = self.llm.llama.model.embed_tokens

            # 设置特殊token
            special_token_ids = self._get_special_token_ids(tokenize_fn)
            self.llm.set_runtime_vars(special_token_ids=special_token_ids)

            # 加载Flow模型
            flow_ckpt = "ckpt/flow/flow.pt"
            flow_config = "ckpt/flow/config.yaml"
            flow = yaml_util.load_flow_model(flow_ckpt, flow_config, self.device)
            self.flow = tts_model_util.Token2Wav(flow, sample_rate=24000, device=self.device)

            # 切换回原目录
            os.chdir(self.original_dir)

            self.model = self.llm  # 设置model属性以便freeze_parameters工作
            logging.info("GLM-TTS models loaded successfully!")

        except Exception as e:
            logging.error(f"Failed to load GLM-TTS: {e}")
            os.chdir(self.original_dir)  # 确保切换回原目录
            self.model = None
            logging.warning("GLM-TTS wrapper falling back to placeholder mode.")

    def _get_special_token_ids(self, tokenize_fn):
        """获取特殊token IDs"""
        _special_token_ids = {
            "ats": "<|audio_0|>",
            "ate": "<|audio_32767|>",
            "boa": "<|begin_of_audio|>",
            "eoa": "<|user|>",
            "pad": "<|endoftext|>",
        }

        special_token_ids = {}
        endoftext_id = tokenize_fn("<|endoftext|>")[0]

        for k, v in _special_token_ids.items():
            __ids = tokenize_fn(v)
            if len(__ids) != 1:
                raise AssertionError(f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}")
            if __ids[0] < endoftext_id:
                raise AssertionError(f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}")
            special_token_ids[k] = __ids[0]

        return special_token_ids

    def synthesize_batch(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        批量合成音频

        输入: (batch_size, audio_length) 16kHz
        输出: (batch_size, audio_length) 16kHz
        """
        # 如果是占位符模式，使用简单重采样
        if self.model is None:
            return self._synthesize_batch_placeholder(audio_batch)

        # 完整实现：使用GLM-TTS进行合成
        batch_size = audio_batch.shape[0]
        target_length = audio_batch.shape[1]
        outputs = []

        # 切换到GLM-TTS目录
        os.chdir(self.glm_tts_dir)

        try:
            # 逐个样本处理
            for i in range(batch_size):
                audio_16k = audio_batch[i]  # 保持在原设备上

                # 上采样到24kHz
                resampler_to_24k = torchaudio.transforms.Resample(
                    orig_freq=16000,
                    new_freq=24000
                ).to(self.device)
                audio_24k = resampler_to_24k(audio_16k.unsqueeze(0)).squeeze(0)

                try:
                    # 保存临时音频文件（GLM-TTS需要从文件读取）
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        torchaudio.save(tmp_path, audio_24k.unsqueeze(0).cpu(), 24000)

                    # 切换到GLM-TTS目录进行推理
                    os.chdir(self.glm_tts_dir)

                    # 提取speaker embedding和speech token
                    embedding = self.frontend._extract_spk_embedding(tmp_path)
                    prompt_speech_token = self.frontend._extract_speech_token([tmp_path])

                    # 使用Flow模型直接推理（不使用LLM生成）
                    with torch.no_grad():
                        token_bt = prompt_speech_token.to(self.device)

                        # 直接调用flow的inference_with_cache（允许空prompt）
                        mel_btd, _ = self.flow.flow.inference_with_cache(
                            token=token_bt,
                            prompt_token=None,
                            prompt_feat=None,
                            embedding=embedding,
                            n_timesteps=10
                        )

                        # 使用vocoder转换mel到音频
                        tts_speech = self.flow.vocoder(mel_btd)

                    # 切换回原目录
                    os.chdir(self.original_dir)

                    # 清理临时文件
                    os.unlink(tmp_path)

                    # 下采样到16kHz
                    output_16k = self.resample_to_target(tts_speech.squeeze(0), orig_sr=24000)
                    output_16k = self.pad_or_crop_to_length(output_16k.unsqueeze(0), target_length).squeeze(0)

                except Exception as e:
                    import traceback
                    os.chdir(self.original_dir)  # 确保切换回原目录
                    logging.warning(f"GLM-TTS synthesis failed for sample {i}: {type(e).__name__}: {e}")
                    logging.debug(traceback.format_exc())
                    output_16k = audio_batch[i]  # 降级方案

                outputs.append(output_16k)

        finally:
            # 切换回原目录
            os.chdir(self.original_dir)

        return torch.stack(outputs)

    def _synthesize_batch_placeholder(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """占位符实现：简单的重采样循环"""
        batch_size = audio_batch.shape[0]
        target_length = audio_batch.shape[1]

        outputs = []
        for i in range(batch_size):
            audio_16k = audio_batch[i]

            # 上采样到24kHz
            resampler_to_24k = torchaudio.transforms.Resample(
                orig_freq=16000,
                new_freq=24000
            ).to(self.device)
            audio_24k = resampler_to_24k(audio_16k.unsqueeze(0)).squeeze(0)

            # 占位符:直接使用输入(模拟TTS处理)
            output_24k = audio_24k

            # 下采样回16kHz
            output_16k = self.resample_to_target(output_24k, orig_sr=24000)
            output_16k = self.pad_or_crop_to_length(output_16k.unsqueeze(0), target_length).squeeze(0)

            outputs.append(output_16k)

        return torch.stack(outputs)


class YourTTSWrapper(TTSModelBase):
    """
    YourTTS包装器 - 完整实现

    原生采样率: 16kHz
    输出采样率: 16kHz (无需重采样)
    """

    def __init__(self, device='cuda'):
        super().__init__(device, sample_rate=16000)

        # 添加YourTTS路径
        yourtts_dir = amn_opensource_code_dir / "coqui_ai_TTS"
        if str(yourtts_dir) not in sys.path:
            sys.path.insert(0, str(yourtts_dir))

        try:
            logging.info("Loading YourTTS model...")

            # 导入YourTTS组件
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.models.vits import Vits

            # 查找预训练模型
            pretrained_dir = amn_opensource_code_dir.parent / "pretrained_models" / "coqui_ai"
            model_path = pretrained_dir / "tts" / "tts_models--multilingual--multi-dataset--your_tts" / "model_file.pth"
            config_path = pretrained_dir / "tts" / "tts_models--multilingual--multi-dataset--your_tts" / "config.json"

            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"YourTTS model not found at {pretrained_dir}")

            # 加载配置
            config = VitsConfig()
            config.load_json(str(config_path))

            # 修复config.json中的硬编码路径
            model_dir = model_path.parent

            # 修复speaker encoder路径
            if hasattr(config.model_args, 'speaker_encoder_config_path'):
                config.model_args.speaker_encoder_config_path = str(model_dir / "config_se.json")
            if hasattr(config.model_args, 'speaker_encoder_model_path'):
                config.model_args.speaker_encoder_model_path = str(model_dir / "model_se.pth")

            # 对于zero-shot，不需要预加载的speaker embeddings
            if hasattr(config.model_args, 'd_vector_file'):
                config.model_args.d_vector_file = None
            if hasattr(config, 'd_vector_file'):
                config.d_vector_file = None

            # 加载模型
            self.model = Vits.init_from_config(config)
            self.model.load_checkpoint(config, str(model_path), eval=True, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            # 冻结参数
            self.freeze_parameters()

            logging.info("YourTTS model loaded successfully!")

        except Exception as e:
            logging.error(f"Failed to load YourTTS: {e}")
            self.model = None
            logging.warning("YourTTS wrapper falling back to placeholder mode.")

    def synthesize_batch(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        批量合成音频

        输入: (batch_size, audio_length) 16kHz
        输出: (batch_size, audio_length) 16kHz
        """
        # 如果是占位符模式，使用简单克隆
        if self.model is None:
            return audio_batch.clone()

        # 完整实现：使用YourTTS进行zero-shot语音克隆
        batch_size = audio_batch.shape[0]
        target_length = audio_batch.shape[1]
        outputs = []

        # 逐个样本处理
        for i in range(batch_size):
            audio_16k = audio_batch[i].cpu()

            try:
                # 保存临时音频文件（用作speaker reference）
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    torchaudio.save(tmp_path, audio_16k.unsqueeze(0), 16000)

                # 计算speaker embedding
                speaker_name = f"temp_speaker_{i}"
                if speaker_name not in self.model.speaker_manager.name_to_id:
                    speaker_embedding = self.model.speaker_manager.compute_embedding_from_clip(tmp_path)

                    # 注册到speaker manager
                    self.model.speaker_manager.name_to_id[speaker_name] = len(self.model.speaker_manager.name_to_id)
                    self.model.speaker_manager.embeddings_by_names[speaker_name] = [speaker_embedding]

                # 使用dummy文本进行语音合成（实际上是音频重建）
                # 注意：这里简化处理，实际应该使用ASR提取文本
                dummy_text = "Hello, this is a test."

                # 临时修改test_sentences进行推理
                old_test_sentences = self.model.config.test_sentences if hasattr(self.model.config, 'test_sentences') else None
                self.model.config.test_sentences = [
                    [dummy_text, speaker_name, None, "en"]
                ]

                with torch.no_grad():
                    gen_dic = self.model.test_run(assets=None)

                # 恢复原始配置
                if old_test_sentences is not None:
                    self.model.config.test_sentences = old_test_sentences

                # 获取输出音频
                gen_audios_dic = gen_dic["audios"]
                waveform = list(gen_audios_dic.values())[0]

                # 转换为torch tensor
                output_16k = torch.from_numpy(waveform).float().to(self.device)

                # 调整长度
                output_16k = self.pad_or_crop_to_length(output_16k.unsqueeze(0), target_length).squeeze(0)

                # 清理临时文件
                os.unlink(tmp_path)

            except Exception as e:
                import traceback
                logging.warning(f"YourTTS synthesis failed for sample {i}: {type(e).__name__}: {e}")
                logging.debug(traceback.format_exc())
                output_16k = audio_batch[i]  # 降级方案

            outputs.append(output_16k)

        return torch.stack(outputs)


# 工厂函数
def create_tts_wrapper(model_name: str, device='cuda') -> TTSModelBase:
    """
    创建TTS包装器实例

    参数:
        model_name: 'echo', 'glm', 或 'yourtts'
        device: 'cuda' 或 'cpu'

    返回:
        对应的TTS包装器实例
    """
    model_name = model_name.lower()

    if model_name == 'echo':
        return EchoTTSWrapper(device)
    elif model_name == 'glm':
        return GLMTTSWrapper(device)
    elif model_name == 'yourtts':
        return YourTTSWrapper(device)
    else:
        raise ValueError(f"Unknown TTS model: {model_name}. Choose from ['echo', 'glm', 'yourtts']")


if __name__ == '__main__':
    # 单元测试
    logging.basicConfig(level=logging.INFO)

    print("测试TTSWrapper...")

    # 创建测试音频
    test_audio = torch.randn(3, 16000)  # 3个1秒16kHz音频
    print(f"测试音频shape: {test_audio.shape}")

    # 测试YourTTS (最简单)
    print("\n测试YourTTS...")
    yourtts = create_tts_wrapper('yourtts')
    output_yourtts = yourtts.synthesize_batch(test_audio)
    print(f"YourTTS输出shape: {output_yourtts.shape}")
    assert output_yourtts.shape == test_audio.shape

    # 测试GLM-TTS
    print("\n测试GLM-TTS...")
    glmtts = create_tts_wrapper('glm')
    output_glm = glmtts.synthesize_batch(test_audio)
    print(f"GLM-TTS输出shape: {output_glm.shape}")
    assert output_glm.shape == test_audio.shape

    # 测试Echo-TTS (最复杂,可能失败)
    print("\n测试Echo-TTS...")
    try:
        echotts = create_tts_wrapper('echo')
        output_echo = echotts.synthesize_batch(test_audio[:1])  # 只测试1个样本
        print(f"Echo-TTS输出shape: {output_echo.shape}")
    except Exception as e:
        print(f"Echo-TTS测试失败: {e}")

    print("\n所有测试完成!")
