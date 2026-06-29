"""
诊断脚本 - 检查 run_evaluate.py 的路径配置

此脚本会检查：
1. exp_cfg.out_dir 的配置
2. 期望的迭代目录路径
3. 每个目录中实际的 .wav 文件数量
4. 是否存在路径配置问题
"""

import exp_setup
from pathlib import Path
from tqdm import tqdm

def diagnose_paths():
    print("=" * 80)
    print("开始诊断 run_evaluate.py 的路径配置...")
    print("=" * 80)
    
    # 初始化配置
    exp_cfg = exp_setup.init_config()
    
    print(f"\n1. 配置信息:")
    print(f"   out_dir = {exp_cfg.out_dir}")
    print(f"   speaker_adapt_iters = {exp_cfg.speaker_adapt_iters}")
    print(f"   wm_length = {exp_cfg.wm_length}")
    
    # 获取说话人列表
    speakers_wm_lst, _, _ = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)
    speaker_lst = [x["speaker"] for x in speakers_wm_lst]
    
    print(f"\n2. 说话人列表 ({len(speaker_lst)} 个):")
    for speaker in speaker_lst:
        print(f"   - {speaker}")
    
    # 检查每个 TTS 实验
    for tts_exp_name in ["ExpSpeakerAdaptYourTTS", "ExpSpeakerAdaptSV2TTS"]:
        print(f"\n{'=' * 80}")
        print(f"检查实验: {tts_exp_name}")
        print(f"{'=' * 80}")
        
        exp_dir = Path(exp_cfg.out_dir).joinpath(tts_exp_name)
        print(f"\n实验目录: {exp_dir}")
        print(f"目录是否存在: {exp_dir.exists()}")
        
        if not exp_dir.exists():
            print(f"❌ 警告: 实验目录不存在!")
            continue
        
        # 检查每个说话人
        for speaker_name in speaker_lst:
            print(f"\n--- 说话人: {speaker_name} ---")
            
            # 检查每个迭代
            has_error = False
            for cur_iter in exp_cfg.speaker_adapt_iters:
                iter_dir = exp_dir.joinpath(f"adapt_to_{speaker_name}/iter_{cur_iter:04d}")
                
                if not iter_dir.exists():
                    print(f"  ❌ iter_{cur_iter:04d}: 目录不存在 - {iter_dir}")
                    has_error = True
                else:
                    # 统计 .wav 文件数量
                    wav_files = list(iter_dir.glob("*.wav"))
                    num_files = len(wav_files)
                    
                    if num_files == 100:
                        print(f"  ✅ iter_{cur_iter:04d}: {num_files} 个文件")
                    else:
                        print(f"  ❌ iter_{cur_iter:04d}: {num_files} 个文件 (期望100个) - {iter_dir}")
                        has_error = True
                        
                        # 显示前5个文件作为示例
                        if num_files > 0:
                            print(f"     示例文件:")
                            for f in wav_files[:5]:
                                print(f"       - {f.name}")
            
            if not has_error:
                print(f"  ✨ {speaker_name} 的所有迭代都正确!")
    
    print("\n" + "=" * 80)
    print("诊断完成!")
    print("=" * 80)
    
    print("\n如果发现问题:")
    print("  1. 路径不存在: 检查 exp_setup.py 中的 out_dir 配置")
    print("  2. 文件数量不对: 检查训练是否正确完成")
    print("  3. 路径正确但仍报错: 可能是大小写或特殊字符问题")

if __name__ == '__main__':
    diagnose_paths()
