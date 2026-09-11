# test camera with customizable resolution and FPS
# 使用方法：
# python testcamera.py  # 使用默认分辨率 640x480 和默认FPS 30
# python testcamera.py --width 256 --height 256  # 设置分辨率为 256x256
# python testcamera.py --fps 60  # 设置FPS为60

import cv2
import argparse
import time

# 设置命令行参数
parser = argparse.ArgumentParser(description='Test camera with custom resolution and FPS')
parser.add_argument('--width', type=int, default=640, help='Width of camera capture')
parser.add_argument('--height', type=int, default=480, help='Height of camera capture')
parser.add_argument('--fps', type=int, default=30, help='FPS of camera capture')
args = parser.parse_args()

# 打印设置的分辨率和FPS
print(f"Setting camera resolution to {args.width}x{args.height}, FPS to {args.fps}")

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2)

# 设置摄像头解码器为MJPG格式，可以提高帧率性能
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 设置摄像头分辨率和FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap1.set(cv2.CAP_PROP_FPS, args.fps)

# check if camera is opened
if not cap.isOpened():
    print("Error: Could not open video stream 0.")
    exit()
if not cap1.isOpened():
    print("Error: Could not open video stream 1.")
    exit()

# 获取实际设置的分辨率和FPS
actual_width0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps0 = int(cap.get(cv2.CAP_PROP_FPS))

actual_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps1 = int(cap1.get(cv2.CAP_PROP_FPS))

print(f"Actual resolution for Camera 0: {actual_width0}x{actual_height0}, FPS: {actual_fps0}")
print(f"Actual resolution for Camera 2: {actual_width1}x{actual_height1}, FPS: {actual_fps1}")

# 用于计算FPS的变量
frame_count0 = 0
frame_count1 = 0
start_time = time.time()
last_time = start_time
current_fps0 = 0
current_fps1 = 0

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    
    # 计算实际FPS
    current_time = time.time()
    if ret:
        frame_count0 += 1
    if ret1:
        frame_count1 += 1
    
    # 每秒更新一次FPS计算
    if (current_time - last_time) >= 1.0:
        current_fps0 = frame_count0 / (current_time - last_time)
        current_fps1 = frame_count1 / (current_time - last_time)
        frame_count0 = 0
        frame_count1 = 0
        last_time = current_time
    
    # Add camera index, resolution and FPS overlay
    if ret:
        cv2.putText(frame, f"Camera 0 ({actual_width0}x{actual_height0})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Set FPS: {actual_fps0}, Actual FPS: {current_fps0:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if ret1:
        cv2.putText(frame1, f"Camera 2 ({actual_width1}x{actual_height1})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, f"Set FPS: {actual_fps1}, Actual FPS: {current_fps1:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    cv2.imshow("frame1", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap1.release()
cv2.destroyAllWindows()