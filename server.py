import asyncio
import json
import sys
import numpy as np
import websockets

from postprocess_utils import rich_transcription_postprocess

# 初始化FunASR模型

import asyncio
import json
import numpy as np
import websockets
from SenseVoiceSmallRKNN2.sensevoice_rknn import *

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

class ASRService:
  def __init__(self, model_config=None):
    """初始化ASR服务，加载语音识别模型"""

    if model_config is None: 
      model_config = {
      "model_dir": "model/rknn/SenseVoiceSmall",
      "device": "npu",  # 也可以设置为"cuda"使用GPU
      "thread_num": 1
      }


    # self.model = AutoModel(
    #   model=model.config["model_dir"],
    #   trust_remote_code=True,
    #   remote_code="./model.py",
    #   vad_model="fsmn-vad",
    #   vad_kwargs={"max_single_segment_time": 30000},
    #   device="cuda:0",
    # )

    model_dir = model_config["model_dir"] 
    self.front = WavFrontend(os.path.join(model_dir, "am.mvn"))
    self.model = SenseVoiceInferenceSession(
        os.path.join(model_dir, "embedding.npy"),
        os.path.join(model_dir,"sense-voice-encoder.rknn"),
        os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
        model_config["device"],
        model_config["thread_num"],
    )

    # TODO(xl): thread unsafe
    self.vad = FSMNVad(model_dir)

  def post_process_result(self, input_str):
    index = input_str.rfind("|>")
    if index < -1 or index + 2 >= len(input_str):
      return input_str
    else:
      return input_str[index + 2:-1]

  def recognize(self, audio_data, audio_language, sample_rate=16000):
    """执行语音识别，返回文本结果"""
    try:
      segments = self.vad.segments_offline(audio_data)
      result = ""
      for part in segments:
        audio_feats = self.front.get_features(audio_data[part[0] * 16 : part[1] * 16])
        asr_result = self.model(
            audio_feats[None, ...],
            language=languages[audio_language],
            use_itn=True
        )
        logging.info(f"[[{part[0] / 1000}s - {part[1] / 1000}s] {asr_result}")
        
        result += self.post_process_result(asr_result)
      self.vad.vad.all_reset_detection()
      return {"text": result, "success": True}
    except Exception as e:
      print(f"识别过程发生错误: {e}")
      return {"error": str(e), "success": False}
    
  async def handle_websocket(self, websocket):
    """处理WebSocket连接，接收音频数据并返回识别结果"""
    try:
      print("新连接已建立")
      audio_buffer = bytearray()
      end_received = False
      audio_language = "auto"
      while not end_received:
        message = await websocket.recv()
        if isinstance(message, str):
          # 处理JSON消息（如结束标志）
          msg = json.loads(message)
          end_received = msg.get("end", False)
          audio_language = msg.get("language", "auto")
        else:
          # 处理音频数据
          audio_buffer.extend(message)
          # 3. 将字节数据转换为numpy数组
      audio_data = np.frombuffer(audio_buffer, dtype=np.float32)
      # 执行识别（使用线程池避免阻塞事件循环）
      loop = asyncio.get_event_loop()
      result = await loop.run_in_executor(None, self.recognize, audio_data, audio_language)
      # 发送识别结果给客户端
      await websocket.send(json.dumps(result))
            
    except Exception as e:
      print(f"处理连接时发生错误: {e}")
    finally:
      print("连接已关闭")


  def start(self, host="0.0.0.0", port=9101):
    """启动WebSocket服务器"""
    print(f"WebSocket服务器将在 {host}:{port} 启动")

    # 使用asyncio.run()创建并管理事件循环
    asyncio.run(self._start_server(host, port))

  async def _start_server(self, host, port):
    # 创建WebSocket服务器
    server = await websockets.serve(self.handle_websocket, host, port, max_size=4*1024*1024)

    print(f"WebSocket服务器已启动，监听端口 {port}")
    # 保持服务器运行
    await server.wait_closed()


# 使用示例
if __name__ == "__main__":
  asr_service = ASRService()
  
  # 启动服务
  asr_service.start()
