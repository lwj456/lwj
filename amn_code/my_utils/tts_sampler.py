"""
TTS模型样本均分采样器 - 支持batch内均分策略

功能: 确保每个batch的样本均匀分配给三个TTS模型(Echo-TTS, GLM-TTS, YourTTS)

作者: Claude Code
日期: 2026-01-14
"""

import torch
import numpy as np
from typing import List, Tuple


def triple_tts_collate_fn(batch):
    """
    为batch内的样本分配TTS模型ID

    功能:
        - 将batch平均分成3份,分别分配给echo, glm, yourtts
        - 确保batch_size能被3整除

    参数:
        batch: DataLoader返回的batch数据
               格式: [(data1, label1), (data2, label2), ..., (dataN, labelN)]
               其中data可能是tuple (audio, watermark)

    返回:
        ((audios, wms), labels, model_ids) 的tuple,其中:
        - audios: (batch_size, audio_length) torch.FloatTensor
        - wms: (batch_size, wm_length) torch.LongTensor
        - labels: (batch_size,) torch.LongTensor (dummy, 通常为0)
        - model_ids: (batch_size,) torch.LongTensor
                     值为0/1/2,对应echo/glm/yourtts

    示例:
        batch_size=63时:
        - samples 0-20  (21个) → model_id=0 (echo)
        - samples 21-41 (21个) → model_id=1 (glm)
        - samples 42-62 (21个) → model_id=2 (yourtts)
    """
    batch_size = len(batch)

    # 验证batch_size能被3整除
    if batch_size % 3 != 0:
        raise ValueError(
            f"batch_size ({batch_size}) 必须能被3整除,以便均分给三个TTS模型。"
            f"建议使用batch_size=63或66。当前batch将被截断。"
        )

    samples_per_model = batch_size // 3  # 每个模型分配的样本数

    # 分配模型ID: 0=echo, 1=glm, 2=yourtts
    model_ids = []
    for i in range(batch_size):
        model_id = i // samples_per_model  # 整数除法:0~20→0, 21~41→1, 42~62→2
        if model_id >= 3:  # 防止越界(如果batch_size不能被3整除)
            model_id = 2
        model_ids.append(model_id)

    # 解包batch数据
    # 支持两种格式:
    # 1. ((audio, wm), label)
    # 2. (audio, label)

    audios = []
    wms = []
    labels = []

    for item in batch:
        data, label = item

        if isinstance(data, tuple) and len(data) == 2:
            # 格式1: data = (audio, wm)
            audio, wm = data
            audios.append(audio)
            wms.append(wm)
        else:
            # 格式2: data = audio
            audios.append(data)
            wms.append(torch.zeros(1))  # 占位符

        labels.append(label)

    # 转换为tensor
    try:
        audios = torch.stack(audios)  # (batch_size, audio_length)
    except Exception as e:
        # 如果stack失败,尝试pad到相同长度
        max_len = max(a.shape[-1] for a in audios)
        audios_padded = []
        for a in audios:
            if a.shape[-1] < max_len:
                pad_len = max_len - a.shape[-1]
                a = torch.nn.functional.pad(a, (0, pad_len), mode='constant', value=0)
            audios_padded.append(a)
        audios = torch.stack(audios_padded)

    try:
        if wms[0].numel() > 1:  # 不是占位符
            wms = torch.stack(wms)  # (batch_size, wm_length)
        else:
            wms = torch.zeros(batch_size, 1)  # 占位符
    except:
        wms = torch.zeros(batch_size, 1)

    labels = torch.tensor(labels, dtype=torch.long)  # (batch_size,)
    model_ids = torch.tensor(model_ids, dtype=torch.long)  # (batch_size,)

    # 返回格式: ((audios, wms), labels, model_ids)
    return (audios, wms), labels, model_ids


def verify_model_distribution(model_ids: torch.Tensor, batch_size: int) -> dict:
    """
    验证model_ids的分布是否均匀

    参数:
        model_ids: (batch_size,) torch.LongTensor, 值为0/1/2
        batch_size: batch大小

    返回:
        dict包含统计信息: {
            'echo_count': int,
            'glm_count': int,
            'yourtts_count': int,
            'is_balanced': bool
        }
    """
    unique, counts = torch.unique(model_ids, return_counts=True)

    distribution = {
        'echo_count': 0,
        'glm_count': 0,
        'yourtts_count': 0
    }

    for model_id, count in zip(unique.tolist(), counts.tolist()):
        if model_id == 0:
            distribution['echo_count'] = count
        elif model_id == 1:
            distribution['glm_count'] = count
        elif model_id == 2:
            distribution['yourtts_count'] = count

    # 检查是否均衡(允许±1的误差)
    expected_per_model = batch_size // 3
    is_balanced = all(
        abs(count - expected_per_model) <= 1
        for count in distribution.values()
    )
    distribution['is_balanced'] = is_balanced

    return distribution


# 可选: Custom Sampler (更高级的采样策略)
class TripleModelBatchSampler(torch.utils.data.Sampler):
    """
    自定义BatchSampler,确保每个batch严格均分给三个模型

    用法:
        sampler = TripleModelBatchSampler(dataset, batch_size=63)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, data_source, batch_size: int, drop_last: bool = False):
        """
        参数:
            data_source: Dataset实例
            batch_size: batch大小(必须能被3整除)
            drop_last: 是否丢弃最后不完整的batch
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_size % 3 != 0:
            raise ValueError(f"batch_size ({batch_size}) 必须能被3整除")

        self.num_samples = len(data_source)

    def __iter__(self):
        # 生成随机打乱的索引
        indices = torch.randperm(self.num_samples).tolist()

        # 分割成batch
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i: i + self.batch_size]

            # 检查batch大小
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and len(batch) >= 3:
                # 保留不完整的batch,但截断到能被3整除
                valid_size = (len(batch) // 3) * 3
                batches.append(batch[:valid_size])

        # 打乱batch顺序
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    # 单元测试
    print("测试triple_tts_collate_fn...")

    # 创建测试数据
    batch_size = 63
    audio_length = 16000
    wm_length = 128

    # 格式1: ((audio, wm), label)
    batch = []
    for i in range(batch_size):
        audio = torch.randn(audio_length)
        wm = torch.randint(0, 2, (wm_length,))
        batch.append(((audio, wm), 0))

    # 测试collate_fn
    (audios, wms), labels, model_ids = triple_tts_collate_fn(batch)

    print(f"audios shape: {audios.shape}")  # 应该是 (63, 16000)
    print(f"wms shape: {wms.shape}")  # 应该是 (63, 128)
    print(f"labels shape: {labels.shape}")  # 应该是 (63,)
    print(f"model_ids shape: {model_ids.shape}")  # 应该是 (63,)

    # 验证model_ids分布
    distribution = verify_model_distribution(model_ids, batch_size)
    print(f"\nModel分布:")
    print(f"  Echo-TTS (0): {distribution['echo_count']} 个样本")
    print(f"  GLM-TTS (1): {distribution['glm_count']} 个样本")
    print(f"  YourTTS (2): {distribution['yourtts_count']} 个样本")
    print(f"  是否均衡: {distribution['is_balanced']}")

    # 验证正确性
    assert audios.shape == (batch_size, audio_length)
    assert wms.shape == (batch_size, wm_length)
    assert model_ids.shape == (batch_size,)

    # 验证分配
    samples_per_model = batch_size // 3
    assert (model_ids[:samples_per_model] == 0).all(), "前21个应该是echo"
    assert (model_ids[samples_per_model:2*samples_per_model] == 1).all(), "中21个应该是glm"
    assert (model_ids[2*samples_per_model:] == 2).all(), "后21个应该是yourtts"

    print("\n✓ 所有测试通过!")

    # 测试TripleModelBatchSampler
    print("\n测试TripleModelBatchSampler...")
    from torch.utils.data import TensorDataset

    dummy_dataset = TensorDataset(
        torch.randn(200, audio_length),
        torch.randint(0, 2, (200, wm_length))
    )

    batch_sampler = TripleModelBatchSampler(dummy_dataset, batch_size=63, drop_last=True)
    print(f"生成的batch数量: {len(batch_sampler)}")

    # 生成几个batch
    for i, batch_indices in enumerate(batch_sampler):
        print(f"Batch {i}: {len(batch_indices)} 个样本")
        if i >= 2:  # 只显示前3个
            break

    print("\n✓ Sampler测试通过!")
