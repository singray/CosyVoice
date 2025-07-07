import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)  # 添加项目根目录到Python路径
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))  # 使用pip安装的matcha-tts可删除此行

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio




cosyvoice = CosyVoice2('../pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
#cosyvoice = CosyVoice2('../pretrained_models/CosyVoice-300M', load_jit=False, load_trt=False, fp16=False)
#cosyvoice = CosyVoice2('../pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)



# 3秒复刻
prompt_speech_16k = load_wav('template/my_voice_1.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('床前明月光，疑是地上霜，举头望明月，低头思故乡', '床前明月光，疑是地上霜，举头望明月，低头思故乡', prompt_speech_16k, stream=False)):
    torchaudio.save('output/my_voice_1_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# 3秒复刻
prompt_speech_16k = load_wav('template/my_voice_1.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '床前明月光，疑是地上霜，举头望明月，低头思故乡', prompt_speech_16k, stream=False)):
    torchaudio.save('output/my_voice_2_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


## 精细微调，即情感生成
#for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#    torchaudio.save('output/fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

## 指导生成
#for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
 #   torchaudio.save('output/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
