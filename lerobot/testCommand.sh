 python -m lerobot.teleoperate \
    --robot.usb_index=1 \
    --robot.type=enpei_follower \
    --robot.id=enpei_follower \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 2, width: 320, height: 240, fps: 30}, fixed: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}" \
    --teleop.type=enpei_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=enpei_leader \
    --fps=30\
    --display_data=true \
    --enpei_speed_mode=record\
    --enpei_use_radian=true



python -m lerobot.teleoperate \
    --robot.ip_address="localhost" \
    --robot.port=12345 \
    --robot.type=enpei_follower \
    --robot.id=enpei_follower \
    --robot.cameras="{ }" \
    --teleop.type=enpei_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=enpei_leader \
    --fps=30\
    --display_data=false \
    --enpei_speed_mode=record