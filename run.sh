i# #!/bin/bash

# 定义服务启动命令和路径
SERVICE_PATH="."
SERVICE_CMD="python srt_server.py"
# pid=`ps -ef | grep "$cmd" | grep -v grep | awk '{print $2}'`

# 检查服务是否在运行
if ! pgrep -f "$SERVICE_CMD" >/dev/null;
then
  echo "服务未运行，正在启动..."
  cd "$SERVICE_PATH"
  nohup $SERVICE_CMD > output.log 2>&1 &
else
  echo "服务正在运行，先停止再重启..."
  # 查找并杀死正在运行的python server.py进程
  pkill -f "$SERVICE_CMD"
  # 等待一段时间，确保进程已经结束
  sleep 2
  cd "$SERVICE_PATH"
  nohup $SERVICE_CMD > output.log 2>&1 &
  echo "服务已重启"
fi

