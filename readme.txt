=============================================
概述

	- 文件夹 "amn_code" 包含论文实验所需的全部代码。

	- 文件夹 "amn_opensource_code" 包含其他工作的开源代码，例如 YourTTS（coqui_ai_TTS）。

	- 文件夹 "pretrained_models" 包含其他工作的预训练权重，例如 YourTTS（coqui_ai_TTS）。AASIST 的预训练权重存放在 "amn_code/run_compare_aasist/weights" 中。

	- 文件夹 "web_demo" 包含网页演示代码。

	- 大多数评估结果都是通过运行 run_evaluate.py 生成的。该脚本应在完成所有实验之后运行，例如使用 YourTTS 和 SV2TTS 进行说话人自适应之后，因为该脚本会加载已有实验结果，然后计算指标并绘制图表。

=============================================
IDE 设置

	- 所有实验中使用的 Python 版本为 3.11.3，PyTorch 版本为 2.3.0。

	- 开发使用的 IDE 是 PyCharm。为使 IDE 加载正确的脚本，请将以下文件夹标记为 Source Root：
		- "amn_code"
		- "amn_opensource_code"
		- "amn_opensource_code/aasist"
		- "amn_opensource_code/coqui_ai_TTS"
		- "amn_opensource_code/Real-Time-Voice-Cloning"
		- "amn_opensource_code/TimbreWatermarking/watermarking_model"
		- "amn_opensource_code/TimbreWatermarking/watermarking_model/distortions"
		- "amn_opensource_code/Trainer"
		- "amn_opensource_code/voxceleb_trainer"


=============================================
VCTK 数据集

	- 从官方网站下载 VCTK 数据集：https://datashare.ed.ac.uk/handle/10283/3443。

	- 解压 VCTK-Corpus-0.92.zip 并预处理音频文件的代码可以在 ExpSpeakerAdaptYourTTS.py 的 prepare_vctk 函数中找到。使用 YourTTS 运行说话人自适应时，例如运行 run_speaker_adapt_YourTTS.py，如果尚未完成 VCTK 数据的解压和预处理，代码会先执行这些步骤。


=============================================
实验

	- 运行 run_wm_speech.py 会从头训练模型，并为 11 名 VCTK 说话人的语音添加水印，运行 run_evaluate.py 会对带水印语音进行评估。

	- 运行 run_speaker_adapt_YourTTS.py 和 run_speaker_adapt_SV2TTS.py 会在带水印的 VCTK 语音上分别执行 YourTTS 和 SV2TTS 的说话人自适应。运行 run_evaluate.py 会对生成的结果进行评估。

	- 伪造语音的文本可以在 "data" 文件夹中的 gen_sentences.txt 中找到，该文件包含来自 LibriSpeech "test-clean" 子集的 100 个句子。

	- 运行 run_prep_for_online.py 会将多个 VCTK 音频拼接成长音频，用于上传到 PlayHT 和 Speechify 以生成伪造语音。原始 VCTK 音频和带水印的 VCTK 音频会分别被拼接成不同的音频。

	- 伪造语音的文本可以在 "data" 文件夹中的 sentences_for_online.txt 中找到，该文件包含来自 LibriSpeech "test-clean" 子集的 14 个句子。

	- 运行 run_evaluate.py 会对PlayHT和Speechify生成的加语音进行评估。

	- 运行 run_evaluate.py 会对语音进行非自适应攻击和自适应攻击并进行评估。

	- 运行 run_wm_speech_diff_set.py 会训练我们的模型，并为另一组 11 名 VCTK 说话人的语音添加水印（在表 10 中记为 Training）。

	- 运行 run_attack_autoencoder.py 会训练一个用于去除该组说话人语音中水印的自编码器。

	- 运行 run_speaker_adapt_YourTTS_autoencoder.py 会在受害者的去噪带水印语音上微调 YourTTS。

	- 运行 run_evaluate.py 会对去噪自编码攻击进行评估。

	- 运行 run_wm_speech_Obama.py 会训练我们的模型，并为奥巴马的语音和 VCTK 说话人的语音添加水印。

	- 运行 run_attack_iter_YourTTS.py和run_effective_ftune_YourTTS.py 会对限制迭代轮数攻击进行评估。

	- 运行 run_wm_speech_pirate.py 会向已有带水印语音中混入任意水印。

	- 运行 run_speaker_adapt_YourTTS_pirate.py 会在双重水印语音上微调 YourTTS。

	- 运行 run_evaluate.py 会对水印混合情况进行评估。

	- 运行 run_speaker_adapt_YourTTS.py 会在原始语音和带水印语音的混合数据上微调 YourTTS。

	- 运行 run_evaluate.py 会对不同覆盖率水印情况进行检测。

==============================================
README 结束
