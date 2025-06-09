import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import librosa
import argparse

class ASRClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        
    async def send_audio_file(self, audio_path, language="auto"):
        """发送完整音频文件到服务器并获取识别结果"""
        async with websockets.connect(self.server_url) as websocket:
            # 1. 发送配置信息
            config = {"language": language}
            await websocket.send(json.dumps(config))
            
            # 2. 读取并处理音频文件
            audio_data, sample_rate = self._load_and_process_audio(audio_path)
            
            # 3. 发送音频数据（一次性发送）
            audio_bytes = audio_data.tobytes()
            await websocket.send(audio_bytes)
            
            # 4. 发送结束标志
            await websocket.send(json.dumps({"end": True}))
            
            # 5. 接收识别结果
            results = []
            async for message in websocket:
                result = json.loads(message)
                results.append(result)
                if result.get("is_final", False):
                    break
            
            return results
    
    def _load_and_process_audio(self, audio_path):
        """加载并预处理音频文件"""
        # 读取音频文件
        audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        
        # 确保采样率为16kHz（如果不是则重采样）
        if sample_rate != 16000:
            audio_data = librosa.resample(
                y=audio_data,
                orig_sr=sample_rate,
                target_sr=16000,
                res_type='kaiser_best'
            )
            sample_rate = 16000
        
        # 转换为float32类型（与服务器端保持一致）
        audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate

async def main():
    parser = argparse.ArgumentParser(description='ASR WebSocket客户端')
    parser.add_argument('--server', type=str, default='ws://localhost:8765', help='服务器地址')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--lang', type=str, default='auto', help='语言类型 (auto/zh/en/yue/ja/ko)')
    args = parser.parse_args()
    
    client = ASRClient(args.server)
    results = await client.send_audio_file(args.audio, args.lang)
    
    # 打印最终结果
    print("\n识别结果:")
    for result in results:
        if result.get("is_final", False):
            print(f"最终结果: {result['text']}")
        else:
            print(f"中间结果: {result['text']}")

if __name__ == "__main__":
    asyncio.run(main())

