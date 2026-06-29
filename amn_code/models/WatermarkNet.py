import torchaudio
import webrtcvad

from my_utils import utils

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import logging
import torchaudio.pipelines as pipelines


def relu_encoder():
    return nn.LeakyReLU(0.2)


def relu_decoder():
    return nn.LeakyReLU(0.2)


class WatermarkDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=(1, 1)),
            nn.InstanceNorm2d(64),
            relu_decoder(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=(1, 1)),
            nn.InstanceNorm2d(64),
            relu_decoder(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)),
            relu_decoder(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)),
            relu_decoder(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 0)),
            relu_decoder(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 0)),
            relu_decoder(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, ),
            relu_decoder(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, ),
            relu_decoder(),
        )

        linear_dim = 1024
        linear_hidden = 4096 * 2
        self.classifier = nn.Sequential(
            nn.Linear(linear_dim, linear_hidden),
            relu_decoder(),
            nn.Dropout(0.35),  # 降低dropout率从0.70到0.35，减少训练不稳定性

            nn.Linear(linear_hidden, num_classes),
        )

    def forward(self, x, return_features=False):
        features = self.features(x)

        features = features.view(features.size(0), -1)

        out = self.classifier(features)

        if return_features is True:
            return out, features

        return out


class MyWatermarkEncoder(nn.Module):
    def __init__(self, wm_len, freq_len, wav2vec2):
        super().__init__()
        self.freq_len = freq_len

        # we use wav2vec2 for extracting features from input waveforms
        self.wav2vec2 = wav2vec2

        feature_len = 1024
        hidden_dim = 2048

        self.wm_cond_layer = nn.Linear(wm_len, feature_len)

        self.lstm = nn.LSTM(feature_len, hidden_dim)

        self.inv_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            relu_encoder(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            relu_encoder(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            relu_encoder(),

            nn.Conv1d(hidden_dim, freq_len*2, kernel_size=1),

        )

    def forward(self, x, wm):
        # init_spec = x

        x, _ = self.wav2vec2(x)

        # reshape x to (N, H, L)

        x = torch.permute(x, [0, 2, 1])

        # be conditioned on the watermark here
        wm_trans = self.wm_cond_layer(wm * 2.0 - 1)
        x = x + wm_trans.unsqueeze(-1)

        # now x is (N, H, L), we need to change it to (L, N, H)
        # first change x to non-batch first
        x = torch.permute(x, [2, 0, 1])
        x, (hn, cn) = self.lstm(x)
        x = torch.permute(x, [1, 2, 0])

        x = self.inv_block(x)

        x = x.view(x.shape[0], 2, self.freq_len, x.shape[-1])

        return x

    def eval(self):
        super().eval()

        # make lstm in train model for backpropagation
        self.lstm.train()


class WatermarkNet(nn.Module):
    N_FFT = 512                        # number of fft we want to calculate
    HOP_LEN = N_FFT // 4

    DELTA_MAX_FREQ = 1000              # the maximum frequency of delta
    DELTA_MIN_FREQ = 100               # the minimum frequency of delta
    MIN_SPEECH_SPEC_POWER = -1.0       # the minimum speech power. Values less than it will be seen as non-speech
    SKIP_FREQ = 0.0

    aug_normal_prob = None
    aug_normal_scale = None

    def __init__(self, benign_wm, sr, audio_sec_len, wav2vec2_dir,
                 aug_normal_prob=0.0, aug_normal_scale=0.1, use_vad=False,
                 use_tts=False, tts_config=None):
        super().__init__()

        self.sr = sr
        self.audio_sec_len = audio_sec_len  # number of samples in an audio section

        # create wav2vec2 for the encoder
        #  WAV2VEC2_BASE 300MB
        #  WAV2VEC2_LARGE 1.18 GB
        bundle = pipelines.WAV2VEC2_LARGE   # use the large model
        assert bundle.sample_rate == sr, "bundle.sample_rate not equal to our sr"
        wav2vec2 = bundle.get_model(dl_kwargs={"model_dir": wav2vec2_dir})

        # create encoder
        self.min_freq_idx, self.max_freq_idx = self.cal_freq_idx_range(sr)
        freq_len = self.max_freq_idx - self.min_freq_idx
        self.encoder = MyWatermarkEncoder(len(benign_wm), freq_len, wav2vec2)

        # create decoder
        self.decoder = WatermarkDecoder(len(benign_wm))

        self.benign_wm = torch.from_numpy(benign_wm).to(utils.device)

        self.spec_delta = None
        self.x_prime_spec = None
        self.x_prime = None
        self.org_decoded = None

        self.aug_normal_prob = aug_normal_prob
        self.aug_normal_scale = aug_normal_scale

        self.vad = None
        if use_vad is True:
            self.vad = webrtcvad.Vad(0)     # 3-> most aggressive; 0 -> least aggressive

        # ==== 新增: TTS模型初始化 ====
        self.use_tts = use_tts
        self.tts_models = None
        self.x_prime_before_tts = None  # 用于计算TTS重建loss

        if use_tts:
            logging.info("Initializing TTS models for watermark training...")
            from models.TTSWrapper import create_tts_wrapper

            tts_config = tts_config or {}
            tts_device = tts_config.get('device', utils.device)

            self.tts_models = {}
            tts_model_names = ['echo', 'glm', 'yourtts']

            # 逐个尝试加载TTS模型,允许部分失败
            for model_name in tts_model_names:
                try:
                    logging.info(f"Loading {model_name.upper()}-TTS...")
                    model = create_tts_wrapper(model_name, device=tts_device)
                    model.freeze_parameters()
                    self.tts_models[model_name] = model
                    logging.info(f"{model_name.upper()}-TTS loaded successfully!")
                except Exception as e:
                    logging.error(f"Failed to load {model_name.upper()}-TTS: {e}")

            # 检查是否至少有一个TTS模型加载成功
            if len(self.tts_models) == 0:
                logging.warning("No TTS models loaded successfully. TTS will be disabled.")
                self.use_tts = False
                self.tts_models = None
            else:
                logging.info(f"TTS training enabled! {len(self.tts_models)} TTS model(s) loaded: {list(self.tts_models.keys())}")

        # comment this out to use default initialization by pytorch
        # self._initialize_weights()
        pass

    @staticmethod
    def cal_freq_idx_range(sr):
        max_freq = sr // 2
        spec_dim_dec = WatermarkNet.N_FFT // 2 + 1

        min_freq_idx = int(spec_dim_dec * WatermarkNet.DELTA_MIN_FREQ / max_freq)
        max_freq_idx = int(spec_dim_dec * WatermarkNet.DELTA_MAX_FREQ / max_freq)

        return min_freq_idx, max_freq_idx

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def spec_for_classificiation(raw_data):
        # return WatermarkNet.spec_for_reconstruct(raw_data)

        spec = utils.cal_spec(raw_data, n_fft=WatermarkNet.N_FFT, hop_length=WatermarkNet.HOP_LEN).unsqueeze(1)
        return spec

    @staticmethod
    def spec_for_reconstruct(raw_data):
        spec = utils.cal_spec(raw_data, n_fft=WatermarkNet.N_FFT, return_complex=True, hop_length=WatermarkNet.HOP_LEN)

        # stack real and imaginary part
        real_part, img_part = torch.real(spec), torch.imag(spec)

        combined = torch.concat([real_part.unsqueeze(1), img_part.unsqueeze(1)], dim=1)
        return combined

    @staticmethod
    def recover_compelx(combined):
        real_part, img_part = combined[:, 0, :, :], combined[:, 1, :, :]

        rlt = torch.complex(real_part, img_part)
        return rlt

    def aug_waveform(self, waveform):
        aug = waveform
        # may apply some augmentations here
        if self.training:

            # white noise
            if np.random.uniform() < self.aug_normal_prob:
                aug = aug + torch.randn(size=aug.shape).to(utils.device) * self.aug_normal_scale

        return torch.clip(aug, min=-1, max=1)

    def forward(self, x, return_mask=False, wm_strength=1.0, model_ids=None):
        x, wm = x
        x = x.to(utils.device)
        wm = wm.to(utils.device)

        init_spec = self.spec_for_reconstruct(x)
        assert init_spec.shape[-2] == self.N_FFT // 2 + 1

        # only modify interested range of spectrogram
        self.spec_delta = self.encoder(x, wm)

        # make delta the same length as spec
        repeat_factor = int(np.ceil(init_spec.shape[-1] / self.spec_delta.shape[-1]))
        self.spec_delta = torch.repeat_interleave(self.spec_delta, repeat_factor, dim=-1)
        self.spec_delta = self.spec_delta[..., :init_spec.shape[-1]]

        if self.training is False:
            # get rid of non-speech part for no training
            spec_power = torch.sqrt(init_spec[:, 0]**2 + init_spec[:, 1]**2)

            # only consider speech part
            spec_power = spec_power[:, self.min_freq_idx + int((self.max_freq_idx - self.min_freq_idx) * self.SKIP_FREQ): self.max_freq_idx]

            speech_part = (spec_power >= self.MIN_SPEECH_SPEC_POWER).sum(1) > 0
            speech_part = speech_part.unsqueeze(1).unsqueeze(1)

            self.spec_delta = self.spec_delta * speech_part

        self.x_prime_spec = init_spec
        self.x_prime_spec[..., self.min_freq_idx: self.max_freq_idx, :] = (
                self.x_prime_spec[..., self.min_freq_idx: self.max_freq_idx, :] + self.spec_delta * wm_strength)

        self.x_prime = utils.inv_spec(self.recover_compelx(self.x_prime_spec), n_fft=self.N_FFT, hop_length=self.HOP_LEN)
        self.x_prime = torch.clip(self.x_prime, -1.0, 1.0)
        if self.x_prime.shape != x.shape:
            # need to pad x_prime if inverse fft does not preserve the length
            assert self.x_prime.shape[1] < x.shape[1], "only consider the case that the signal is shorter"
            diff = x.shape[1] - self.x_prime.shape[1]
            self.x_prime = F.pad(self.x_prime, (0, diff))
            assert self.x_prime.shape == x.shape

        # ==== 新增: TTS处理阶段 ====
        if self.use_tts and self.training and model_ids is not None and self.tts_models is not None:
            # 保存原始水印音频用于计算TTS重建loss
            self.x_prime_before_tts = self.x_prime.clone()

            # 根据model_ids分batch处理
            batch_size = x.shape[0]
            x_prime_tts = torch.zeros_like(self.x_prime)

            for model_idx, model_name in enumerate(['echo', 'glm', 'yourtts']):
                # 找到属于该模型的样本
                mask = (model_ids == model_idx)
                if mask.sum() == 0:
                    continue

                # 提取该模型的样本
                model_batch = self.x_prime[mask]

                try:
                    # TTS推理(权重frozen,不计算梯度)
                    with torch.no_grad():
                        tts_batch_out = self.tts_models[model_name].synthesize_batch(model_batch)

                    # 允许梯度回传(TTS输出作为常量,但需要梯度流向encoder)
                    tts_batch_out = tts_batch_out.detach().requires_grad_(True)

                except Exception as e:
                    # TTS失败时使用原音频
                    logging.warning(f"TTS {model_name} failed: {e}. Using original audio.")
                    tts_batch_out = model_batch

                # 重组batch(按原始顺序)
                x_prime_tts[mask] = tts_batch_out

            self.x_prime = x_prime_tts

        # do some augmentation here and try to decode watermark
        aug = self.aug_waveform(self.x_prime)
        spec = self.spec_for_classificiation(aug)
        assert spec.shape[-2] == self.N_FFT // 2 + 1

        partial_spec = spec[..., self.min_freq_idx: self.max_freq_idx, :].clone()
        out = self.decoder(partial_spec)

        # also save the decoded original x
        aug_x = self.aug_waveform(x)
        org_spec = self.spec_for_classificiation(aug_x)
        partial_org_spec = org_spec[..., self.min_freq_idx: self.max_freq_idx, :].clone()
        self.org_decoded = self.decoder(partial_org_spec)

        if return_mask:
            return out, None

        return out

    @staticmethod
    def wm_loss(trainer, data, logits, target, cur_epoch, it, other_data):
        x, wm = data
        wm = wm.to(utils.device)
        batch_size = wm.size(0)

        wm_net = trainer.model

        def perturb_loss(_data):
            _loss = (_data ** 2).flatten(1).sum(1)
            _loss = _loss.sum()
            return _loss

        # norm loss
        norm_loss = perturb_loss(wm_net.spec_delta)

        def wm_to_target_vals(_wm, _alpha=5.0):
            # change values from {0, 1} to {-_alpha, _alpha}
            return (_wm * 2.0 - 1.0) * _alpha

        wm_acc_loss = F.mse_loss(logits, wm_to_target_vals(wm), reduction='sum')

        # clean speech are decoded to a fixed value
        benign_wm = wm_net.benign_wm.unsqueeze(0).repeat([batch_size, 1]).float()
        org_wm_acc_loss = F.mse_loss(wm_net.org_decoded, wm_to_target_vals(benign_wm), reduction='sum')

        # ==== 新增: TTS重建loss ====
        tts_recon_loss = torch.tensor(0.0, device=utils.device)
        if wm_net.use_tts and wm_net.training and hasattr(wm_net, 'x_prime_before_tts') and wm_net.x_prime_before_tts is not None:
            # 确保TTS输出与encoder输出相似(保持水印可检测性)
            # 使用MSE loss衡量TTS前后的音频差异
            tts_recon_loss = F.mse_loss(wm_net.x_prime, wm_net.x_prime_before_tts, reduction='sum')

        # 合并loss (权重可调)
        # 调整权重: 减小norm权重，增加wm_acc权重，提高水印嵌入准确率
        # norm=0.05 (大幅减小，允许更多频谱修改)
        # wm_acc=3.0 (增加权重，强制模型学习水印嵌入)
        # org_acc=1.5 (略微增加，确保benign音频不受影响)
        # tts_recon=0.5 (保持不变)
        tts_weight = getattr(other_data.config, 'tts_recon_loss_weight', 0.5) if hasattr(other_data, 'config') else 0.5

        loss = (0.05 * norm_loss +
                3.0 * wm_acc_loss +
                1.5 * org_wm_acc_loss +
                tts_weight * tts_recon_loss)

        return loss

    @staticmethod
    def waveform_to_sections(waveform, sec_len):
        wav_sec_lst = []

        # number of sections
        num_sec = int(np.ceil(len(waveform) / sec_len))
        for sec_idx in range(num_sec):
            start_pos = sec_idx * sec_len
            end_pos = (sec_idx + 1) * sec_len
            end_pos = min(end_pos, len(waveform))

            wav_sec = np.zeros(sec_len)
            wav_sec[0: end_pos - start_pos] = waveform[start_pos: end_pos]
            wav_sec = torch.FloatTensor(wav_sec).unsqueeze(0)

            wav_sec_lst.append(wav_sec)

        wav_secs = torch.concat(wav_sec_lst, dim=0).to(utils.device)

        return wav_secs

    def split_waveform(self, waveform):
        return self.waveform_to_sections(waveform, self.audio_sec_len)

    def inference(self, x, wm_pool, benign_wm_included, wm_decode_func=lambda _k: _k, major_vote_size=None):
        """
        default wm_decode_func is identity function.
        """

        assert self.training is False, "call eval() for inference."

        if hasattr(self, "benign_wm_decoded") is False:
            self.benign_wm_decoded = wm_decode_func(self.benign_wm)

        if benign_wm_included is False:
            # add benign wm to the last
            wm_pool = torch.concat([wm_pool, self.benign_wm_decoded.unsqueeze(0)], dim=0)

        # first remove non-speech part
        if self.vad is not None:
            x = utils.remove_non_speech(x, self.vad, self.sr)

        if len(x) < self.audio_sec_len:
            # x is too short for a section
            return None

        # for each audio section, find the wm
        wav_secs = self.split_waveform(x)

        logits = self.decoder(self.spec_for_classificiation(wav_secs)[..., self.min_freq_idx: self.max_freq_idx, :])
        pred_wm = (logits > 0.0).long()

        pred_wm = wm_decode_func(pred_wm)

        pred_wm = pred_wm.unsqueeze(1)
        wm_pool = wm_pool.unsqueeze(0).repeat(pred_wm.shape[0], 1, 1)

        rlt = (pred_wm == wm_pool).sum(dim=-1)
        rlt = (rlt == wm_pool.shape[-1]).int()
        assert (rlt.sum(dim=1) > 1).sum().item() == 0, "one correct wm for each section at most"

        pred_val, pred_wm_idx = torch.max(rlt, dim=1)
        mask = pred_val < 1     # mask this off if it is not equal to a wm in the pool
        pred_wm_idx[mask] = -1

        if major_vote_size is not None:
            # iterate through all predictions and do max vote
            slide_times = int(np.ceil(pred_wm_idx.size(0) / major_vote_size))
            for i in range(slide_times):
                sub_pred = pred_wm_idx[i * major_vote_size: (i + 1) * major_vote_size]
                unique_out, cnts = torch.unique(sub_pred, return_counts=True)
                major_vote = (cnts > major_vote_size // 2)
                assert major_vote.sum().item() <= 1, "only one element can be the major voted one"
                if major_vote.sum().item() < 1:
                    # do nothing as every element is different
                    continue

                voted_val = unique_out[major_vote].item()
                sub_pred[:] = voted_val

        return pred_wm_idx

    def test_wm_capability(self, benign_wav, tgt_wm):
        """
        test the capability of embedding watermarks
        """
        assert self.training is False, "call eval() for test_wm_acc."
        wav_secs = self.split_waveform(benign_wav)

        tgt_wm = torch.from_numpy(tgt_wm).unsqueeze(0).to(utils.device)
        tgt_wm = tgt_wm.repeat(wav_secs.shape[0], 1)

        benign_wm = self.benign_wm.unsqueeze(0)
        benign_wm = benign_wm.repeat(wav_secs.shape[0], 1)

        logits, mask = self((wav_secs, tgt_wm), return_mask=True)
        preds = (logits > 0.0).long()
        acc_wm = (preds == tgt_wm).sum() / preds.numel()
        acc_wm = acc_wm.item()

        # save current watermarked data
        wm_waveform = self.x_prime.detach()

        # detection for original audio
        org_spec = self.spec_for_classificiation(wav_secs)
        org_logits = self.decoder(org_spec[..., self.min_freq_idx: self.max_freq_idx, :])
        org_preds = (org_logits > 0.0).long()
        acc_org = (org_preds == benign_wm).sum() / org_preds.numel()
        acc_org = acc_org.item()

        return acc_wm, acc_org, wm_waveform







