"""
自定义评估脚本 - 只评估已生成的迭代

这个脚本允许你评估已经生成的迭代数据，而不需要等待所有中间迭代完成。
默认只评估 iter_0001 和 iter_0101。

如果需要评估其他迭代，可以修改 custom_init_config() 函数中的 cfg.speaker_adapt_iters。
"""

import sys
import exp_setup

# 修改配置，只评估已存在的迭代
original_init = exp_setup.init_config

def custom_init_config():
    cfg = original_init()
    
    # 只评估 iter_0001 和 iter_0101
    # 如果你有其他已完成的迭代，可以添加到这个列表中
    cfg.speaker_adapt_iters = [1, 101]
    
    print("=" * 70)
    print("使用自定义迭代配置:")
    print(f"  speaker_adapt_iters = {cfg.speaker_adapt_iters}")
    print("=" * 70)
    
    return cfg

# 替换原始的配置初始化函数
exp_setup.init_config = custom_init_config

# 导入并运行主程序
from run_evaluate import main

if __name__ == '__main__':
    print("\n正在运行自定义评估脚本...")
    print("此脚本只评估已存在的迭代: [1, 101]\n")
    
    main()
    
    print("\n评估完成!")
    sys.exit()
