#!/usr/bin/env python3
"""
简化版水印检测工具 - 检测文件夹中的所有音频

使用方法：
    python amn_code/detect_folder_audios.py --folder "path/to/your/audio/folder"

可选参数：
    --folder: 音频文件夹路径（必需）
    --output: 结果保存路径（可选，默认保存在out/ExpEvaluate/detect_results/）
    --format: 音频格式（可选，默认*.wav）
    
示例：
    python amn_code/detect_folder_audios.py --folder "data/my_fake_speech"
    python amn_code/detect_folder_audios.py --folder "data/my_fake_speech" --format "*.mp3"
"""

from pathlib import Path
import numpy as np
import logging
import sys
import argparse
from tqdm import tqdm
import json

# 修复numpy版本兼容性问题 - 必须在导入torch之前执行
import sys
import numpy

# 为旧版本numpy pickle兼容性创建模块别名
if not hasattr(numpy, '_core'):
    # 创建 numpy._core 模块
    import types
    numpy._core = types.ModuleType('numpy._core')
    numpy._core.multiarray = numpy.core.multiarray
    numpy._core._multiarray_umath = numpy.core._multiarray_umath
    
# 注册到sys.modules中，这样pickle可以找到
sys.modules['numpy._core'] = numpy._core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from exp_setup import init_config, get_speakers_and_wm
from my_utils import utils
from models.WatermarkNet import WatermarkNet
from models.ModelTrainer import ModelTrainer
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class AudioWatermarkDetector:
    """音频水印检测器"""
    
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        self.wm_net = None
        self.speakers_wm_lst = None
        self.wm_pool = None
        self.speaker_names = []
        
    def load_model(self):
        """加载水印检测模型"""
        logging.info("正在加载水印检测模型...")
        
        # 获取说话人和水印配置
        self.speakers_wm_lst, benign_org_wm, benign_encoded_wm = get_speakers_and_wm(
            self.exp_cfg, self.exp_cfg.wm_length
        )
        
        # 准备水印池（所有说话人的水印）
        wm_list = []
        self.speaker_names = []
        for speaker_data in self.speakers_wm_lst:
            wm_list.append(speaker_data["encoded_wm"])
            self.speaker_names.append(speaker_data["speaker"])
        
        self.wm_pool = torch.from_numpy(np.array(wm_list)).to(utils.device)
        
        # 初始化模型
        audio_sec_len = 16000  # 1秒
        self.wm_net = WatermarkNet(
            benign_encoded_wm,
            audio_sec_len,
            audio_sec_len,
            wav2vec2_dir=self.exp_cfg.wav2vec2_pretrained_dir
        )
        
        # 加载模型权重
        model_dir = Path(self.exp_cfg.out_dir).joinpath("ExpEmbedWatermark")
        ckpt_dir = model_dir.joinpath("ckpt")
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"模型检查点目录未找到: {ckpt_dir}\n"
                f"请先运行 run_wm_speech.py 训练水印模型"
            )
        
        dic_saved = ModelTrainer.load_latest_ckpt(ckpt_dir)
        self.wm_net.load_state_dict(dic_saved["model_state"])
        self.wm_net = self.wm_net.to(utils.device)
        self.wm_net.eval()
        
        logging.info(f"模型加载成功！支持检测 {len(self.speaker_names)} 个说话人的水印")
        
    def detect_audio(self, audio_path):
        """
        检测单个音频文件
        
        返回:
            dict: {
                'file': 文件名,
                'duration': 时长（秒）,
                'has_watermark': 是否检测到水印,
                'detected_speaker': 检测到的说话人（如果有）,
                'confidence': 置信度（检测到水印的段数占比）,
                'details': 详细检测信息
            }
        """
        try:
            # 读取音频
            audio, sr = utils.read_audio(audio_path, None)
            
            # 重采样到16kHz
            if sr != self.exp_cfg.sr:
                audio = utils.resample_wav(audio, sr, self.exp_cfg.sr)
            
            duration = len(audio) / self.exp_cfg.sr
            
            # 如果音频太短，无法检测
            if len(audio) < self.exp_cfg.sr:
                return {
                    'file': audio_path.name,
                    'duration': duration,
                    'has_watermark': False,
                    'detected_speaker': None,
                    'confidence': 0.0,
                    'details': '音频太短，无法检测（需要至少1秒）'
                }
            
            # 使用水印网络进行推理
            with torch.no_grad():
                predictions = self.wm_net.inference(
                    audio, 
                    self.wm_pool, 
                    benign_wm_included=False
                )
            
            if predictions is None:
                return {
                    'file': audio_path.name,
                    'duration': duration,
                    'has_watermark': False,
                    'detected_speaker': None,
                    'confidence': 0.0,
                    'details': '检测失败（音频可能损坏或格式不支持）'
                }
            
            # 分析预测结果
            predictions = predictions.cpu().detach().numpy()
            
            # 统计每个说话人被检测到的次数
            speaker_counts = {}
            for speaker_idx in range(len(self.speaker_names)):
                count = (predictions == speaker_idx).sum()
                if count > 0:
                    speaker_counts[self.speaker_names[speaker_idx]] = count
            
            total_segments = len(predictions)
            
            # 判断是否检测到水印
            if len(speaker_counts) > 0:
                # 找出检测次数最多的说话人
                detected_speaker = max(speaker_counts, key=speaker_counts.get)
                detected_count = speaker_counts[detected_speaker]
                confidence = detected_count / total_segments
                
                # 将numpy类型转换为Python原生类型（用于JSON序列化）
                speaker_counts_serializable = {k: int(v) for k, v in speaker_counts.items()}
                
                return {
                    'file': audio_path.name,
                    'duration': float(duration),
                    'has_watermark': True,
                    'detected_speaker': detected_speaker,
                    'confidence': float(confidence),
                    'detected_segments': int(detected_count),
                    'total_segments': int(total_segments),
                    'all_detections': speaker_counts_serializable,
                    'details': f'检测到 {detected_speaker} 的水印，置信度 {confidence:.2%}'
                }
            else:
                return {
                    'file': audio_path.name,
                    'duration': float(duration),
                    'has_watermark': False,
                    'detected_speaker': None,
                    'confidence': 0.0,
                    'detected_segments': 0,
                    'total_segments': int(total_segments),
                    'all_detections': {},
                    'details': '未检测到水印'
                }
                
        except Exception as e:
            logging.error(f"处理文件 {audio_path.name} 时出错: {str(e)}")
            return {
                'file': audio_path.name,
                'duration': 0,
                'has_watermark': False,
                'detected_speaker': None,
                'confidence': 0.0,
                'details': f'错误: {str(e)}'
            }
    
    def detect_folder(self, folder_path, file_pattern="*.wav"):
        """
        检测文件夹中的所有音频
        
        参数:
            folder_path: 文件夹路径
            file_pattern: 文件匹配模式（如 "*.wav", "*.mp3"）
        
        返回:
            list: 检测结果列表
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        
        # 查找所有音频文件
        audio_files = list(folder.glob(file_pattern))
        
        if len(audio_files) == 0:
            logging.warning(f"在 {folder} 中未找到匹配 {file_pattern} 的文件")
            return []
        
        logging.info(f"找到 {len(audio_files)} 个音频文件，开始检测...")
        
        results = []
        for audio_file in tqdm(audio_files, desc="检测进度"):
            result = self.detect_audio(audio_file)
            results.append(result)
        
        return results


def save_results(results, output_dir):
    """保存检测结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果（JSON格式）
    json_file = output_dir.joinpath("detection_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存汇总报告（文本格式）
    report_file = output_dir.joinpath("detection_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("音频水印检测报告\n")
        f.write("=" * 70 + "\n\n")
        
        total_files = len(results)
        watermarked_files = sum(1 for r in results if r['has_watermark'])
        
        f.write(f"总文件数: {total_files}\n")
        f.write(f"检测到水印: {watermarked_files}\n")
        f.write(f"未检测到水印: {total_files - watermarked_files}\n")
        f.write(f"检测率: {watermarked_files/total_files*100:.2f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("详细结果\n")
        f.write("=" * 70 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. 文件: {result['file']}\n")
            f.write(f"   时长: {result['duration']:.2f}秒\n")
            f.write(f"   检测到水印: {'是' if result['has_watermark'] else '否'}\n")
            
            if result['has_watermark']:
                f.write(f"   检测到的说话人: {result['detected_speaker']}\n")
                f.write(f"   置信度: {result['confidence']:.2%}\n")
                f.write(f"   检测段数: {result.get('detected_segments', 0)}/{result.get('total_segments', 0)}\n")
            
            f.write(f"   详情: {result['details']}\n")
            f.write("\n")
        
        # 按说话人统计
        speaker_stats = {}
        for result in results:
            if result['has_watermark']:
                speaker = result['detected_speaker']
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = 0
                speaker_stats[speaker] += 1
        
        if speaker_stats:
            f.write("=" * 70 + "\n")
            f.write("按说话人统计\n")
            f.write("=" * 70 + "\n\n")
            
            for speaker, count in sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{speaker}: {count} 个文件 ({count/total_files*100:.2f}%)\n")
    
    logging.info(f"结果已保存:")
    logging.info(f"  详细结果: {json_file}")
    logging.info(f"  汇总报告: {report_file}")
    
    return report_file


def print_summary(results):
    """打印检测摘要"""
    total_files = len(results)
    watermarked_files = sum(1 for r in results if r['has_watermark'])
    
    print("\n" + "=" * 70)
    print("检测摘要")
    print("=" * 70)
    print(f"总文件数: {total_files}")
    print(f"检测到水印: {watermarked_files} ({watermarked_files/total_files*100:.2f}%)")
    print(f"未检测到水印: {total_files - watermarked_files}")
    
    # 按说话人统计
    speaker_stats = {}
    for result in results:
        if result['has_watermark']:
            speaker = result['detected_speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = 0
            speaker_stats[speaker] += 1
    
    if speaker_stats:
        print("\n按说话人统计:")
        for speaker, count in sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {speaker}: {count} 个文件")
    
    print("=" * 70)
    
    # 显示前5个检测结果
    print("\n前5个文件的检测结果:")
    for i, result in enumerate(results[:5], 1):
        status = "✓ 检测到水印" if result['has_watermark'] else "✗ 未检测到"
        speaker_info = f" ({result['detected_speaker']})" if result['has_watermark'] else ""
        print(f"  {i}. {result['file']}: {status}{speaker_info}")
    
    if len(results) > 5:
        print(f"  ... 还有 {len(results)-5} 个文件")


def main():
    # 先解析我们自己的参数
    parser = argparse.ArgumentParser(
        description='检测文件夹中所有音频的水印',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python amn_code/detect_folder_audios.py --folder "data/my_fake_speech"
  python amn_code/detect_folder_audios.py --folder "data/test" --format "*.mp3"
  python amn_code/detect_folder_audios.py --folder "save/ExpCommercialGenerated" --output "out/my_results"
        """
    )
    
    parser.add_argument('--folder', type=str, required=True,
                        help='要检测的音频文件夹路径')
    parser.add_argument('--output', type=str, default=None,
                        help='结果保存路径（默认: out/ExpEvaluate/detect_results/）')
    parser.add_argument('--format', type=str, default='*.wav',
                        help='音频文件格式（默认: *.wav）')
    
    # 使用 parse_known_args 避免与 exp_setup.init_config() 的参数冲突
    args, unknown = parser.parse_known_args()
    
    # 保存原始的 sys.argv
    original_argv = sys.argv.copy()
    
    # 临时修改 sys.argv，只保留未知参数（用于 init_config）
    sys.argv = [sys.argv[0]] + unknown
    
    print("\n" + "=" * 70)
    print("AudioMarkNet - 文件夹音频水印检测工具")
    print("=" * 70)
    print(f"检测文件夹: {args.folder}")
    print(f"文件格式: {args.format}")
    print("=" * 70 + "\n")
    
    try:
        # 初始化配置
        exp_cfg = init_config()
        
        # 恢复原始的 sys.argv
        sys.argv = original_argv
        
        # 创建检测器并加载模型
        detector = AudioWatermarkDetector(exp_cfg)
        detector.load_model()
        
        # 检测文件夹中的所有音频
        results = detector.detect_folder(args.folder, args.format)
        
        if len(results) == 0:
            print("未找到任何音频文件！")
            return
        
        # 设置输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(exp_cfg.out_dir).joinpath("ExpEvaluate/detect_results")
        
        # 保存结果
        report_file = save_results(results, output_dir)
        
        # 打印摘要
        print_summary(results)
        
        print(f"\n详细报告已保存至: {report_file}")
        print("\n✓ 检测完成！")
        
    except Exception as e:
        # 恢复原始的 sys.argv（以防异常时没有恢复）
        sys.argv = original_argv
        logging.error(f"检测过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
