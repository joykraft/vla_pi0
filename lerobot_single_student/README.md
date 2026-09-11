[toc]

> 下面的流程、代码、工具等只适用于单臂遥操作系统，双臂遥操作系统请查看下一篇。

## 一、遥操作系统校准

### 1.1 环境配置

* Ubuntu电脑：我的版本是20.04.6 LTS，其他版本，尤其是更高版本，可能有些驱动装不了

* 使用Anaconda创建虚拟环境：`conda create -y -n lerobot python=3.10`

* 激活环境：`conda activate lerobot`

* 下载修改后的`lerobot`代码：`git clone https://github.com/enpeizhao/lerobot_single_student`

  > 不要使用[lerobot 官方代码](https://github.com/huggingface/lerobot?tab=readme-ov-file)，因为有代码改动（官方的代码不支持我的机械臂）

* 解压，进入目录`cd lerobot`

* 安装`lerobot`，使用：`pip install -e . -i https://mirrors.huaweicloud.com/repository/pypi/simple `

* 安装飞特舵机支持，使用：`pip install 'lerobot[feetech]'`

* 我的环境依赖参考在根目录的`lerobot_env.txt`





### 1.2 中位校准遥操主臂

* 将遥操主臂的电源和数据线插上，数据线另一头连接电脑

* 运行指令：`ls /dev/ttyACM*  `，检查驱动板在Ubuntu下的端口，比如我的是：`/dev/ttyACM0`

* 运行：`python -m lerobot.set_middle --port=/dev/ttyACM0`，进入校准程序，应该会输出这样的界面：

  ```bash
  INFO 2025-09-10 17:19:06 t_middle.py:117 {'motor_range': (1, 7), 'port': '/dev/ttyACM0'}
  Connected to Feetech motors on port /dev/ttyACM0
  已解锁所有电机（扭矩禁用）
  开始持续监控位置。电机范围: 1-7。按 Ctrl+C 停止。
  您可以在监控时手动移动机械臂。
  按 'r' 重置中位位置。
  按 'l' 切换电机锁定状态（启用/禁用扭矩）。
  原始位置: 电机 1: [2911], 电机 2: [1678], 电机 3: [3691], 电机 4: [2200], 电机 5: [1357], 电机 6: [1903], 电机 7: [2062] 
  ```

* 转动遥操主臂各个关节，你会看到电机位置数值变化

  > 飞特总线舵机编码器的分辨率是4096，即你会看到位置在（0~4096）范围内

* 切换英文输入法，按`r` ，会提示：`请手动移动机械臂到新的中位位置，然后按回车... `

  > 如果没有响应，可以先按一下回车键

* 将遥操主臂各个关节转到它的中间位置（编码器2048的位置），大概如下：

  * 不必要求特别精准，大概在中位就行了
  * 需要注意夹爪支架的方向，不然校准后不顺手
  * 需要注意夹爪手指环的要尽量在中位，不然可能遥操作的执行的主臂夹爪抓不紧
  * 当然，校准不是一次性的，如果不满意，随时可以重新校准

  | 侧视图                                                       | 后视图                                                       | 正视图                                                       |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509101724094.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509101724162.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509101727850.png?x-oss-process=style/resize) |

* 保持遥操主臂静止，按回车键，你会看到输出的电机角度全部校准为了2048
* 退出中位校准程序，保持遥操主臂通电、数据线连接电脑。



### 1.3 机械臂遥操作测试

> <font color="red">操作时，注意安全！！！</font>
>
> <font color="red">操作时，注意安全！！！</font>
>
> <font color="red">操作时，注意安全！！！</font>
>
> <font color="red">第一次最好找一个助手协助，如果执行的从臂异常运动，可以及时断电（注意用手托住关节）</font>

* 修改Episode1机械臂（注意，不是遥操主臂）六个驱动板的参数，将`Response`改为`None`，如果不会修改，请[查看这里](https://enpeicv.com/forum.php?mod=viewthread&tid=1541&extra=page%3D1)

* Episode1机械臂插上夹爪控制盒，夹爪先不用装在机械臂末端（还没有安装相机）

* Episode1机械臂上电，用上位机归零、回到默认位置

* 上位机关闭”启用日志“、”启用状态刷新“ 复选框

* 提前以下列姿态手握一下遥操主臂

  * 遥操作测试脚本运行的时候，机械臂会先运转到下面的姿态（机械臂遥操作准备状态）
  * 为了让主臂、从臂初始姿态尽量一致，建议以图二姿态握住遥操主臂

  | 机械臂遥操作准备状态                                         | 建议遥操主臂初始握住姿态                                     |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111645324.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111646505.png?x-oss-process=style/resize) |

* 运行指令：

  > <font color="red">操作时，注意安全！！！</font>
  >
  > <font color="red">下面是低速模式的配置，请不要改动`robot.ip_address`、`robot.port`和`teleop.port`外其他参数，以免发生危险！</font>

  ```bash
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
  ```

  robot表示从臂执行臂相关参数，telep表示遥操主臂相关参数，在测试时，你只需要修改`robot.ip_address`、`robot.port`和`teleop.port`

  | 参数                           | 是否需要修改 | 解释                                                 |
  | ------------------------------ | ------------ | ---------------------------------------------------- |
  | `robot.ip_address="localhost"` | ✅ 是         | 上位机API的IP地址                                    |
  | `robot.port=12345`             | ✅ 是         | 上位机API的端口                                      |
  | `robot.type=enpei_follower`    | 否           | 从臂类别，方便框架识别                               |
  | `robot.id=enpei_follower`      | 否           | 从臂ID，方便框架识别                                 |
  | `robot.cameras="{ }"`          | 否           | 相机参数，测试时暂不需要填写，实际采集数据的时候需要 |
  | `teleop.type=enpei_leader`     | 否           | 主臂类别，方便框架识别                               |
  | `teleop.port=/dev/ttyACM0`     | ✅  是        | 主臂端驱动板端口，可通过`ls /dev/ttyACM*  `查询      |
  | `teleop.id=enpei_leader`       | 否           | 主臂ID，方便框架识别                                 |
  | `fps=30`                       | 否           | 控制频率，范围0~100，越大，遥操作响应越快            |
  | `display_data=false`           | 否           | `rerun.io`可视化                                     |
  | `enpei_speed_mode=record`      | 否           | 速度模式                                             |

* 一切正常的话，程序启动后，便可以用遥操主臂操作从臂了，请检查：

  * 分别转动主臂前6个关节，看看从臂能否跟着转动

  * 看看夹爪能否控制

  * 终端应该有类似如下输出：

    <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111702265.png?x-oss-process=style/resize" style="zoom: 67%;" />

* 自己遥操作熟练一下，如果需要退出，在终端按`Ctrl+C`退出程序

* 高速模式：

  > <font color="red">操作时，注意安全！！！</font>
  >
  > <font color="red">操作时，注意安全！！！</font>
  >
  > <font color="red">操作时，注意安全！！！</font>

  ```bash
  python -m lerobot.teleoperate \
      --robot.ip_address="localhost" \
      --robot.port=12345 \
      --robot.type=enpei_follower \
      --robot.id=enpei_follower \
      --robot.cameras="{}" \
      --teleop.type=enpei_leader \
      --teleop.port=/dev/ttyACM0 \
      --teleop.id=enpei_leader \
      --fps=100\
      --display_data=false \
      --enpei_speed_mode=teleop
  ```



## 二、安装测试相机

* 运行：`python -m lerobot.episode_default_position`，让从臂运行到遥操默认位置，方便安装相机和夹爪

* 先安装腕部相机，再安装夹爪

  * 相机尽量在正前方，这样可以拍到夹爪
  * 可以用魔术贴扎带将线固定好

  | 位置1                                                        | 位置2                                                        |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111728325.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111728451.png?x-oss-process=style/resize) |

* 将固定位相机也摆好：

  <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111732354.png?x-oss-process=style/resize" style="zoom:50%;" />

* 两个相机的USB线插入电脑USB口，最好是USB3口

  * 如果线不够长，需要自购一根USB3延长线
  * 如果电脑没有多余的USB3接口，需要自购USB3扩展坞

* 运行：`python -m lerobot.find_cameras opencv `，应该输出类似信息：

  ```bash
  --- Detected Cameras ---
  Camera #0:
    Name: OpenCV Camera @ /dev/video0
    Type: OpenCV
    Id: /dev/video0
    Backend api: V4L2
    Default stream profile:
      Format: 16.0
      Width: 640
      Height: 480
      Fps: 30.0
  --------------------
  Camera #1:
    Name: OpenCV Camera @ /dev/video2
    Type: OpenCV
    Id: /dev/video2
    Backend api: V4L2
    Default stream profile:
      Format: 16.0
      Width: 640
      Height: 480
      Fps: 30.0
  --------------------
  
  Finalizing image saving...
  Image capture finished. Images saved to outputs/captured_images
  ```

  * 必须有2个相机（分割线分割），如果数量不对，请检查连线

  * FPS必须都要达到30

  * 确定ID，去`outputs/captured_images`下保存的图片查看

    * 结合文件名和拍摄内容，可以看到腕部相机ID是2，固定位是0
    * 腕部相机必须要能看到柔性夹爪的手指（下方蓝色手指），否则需要调整你的相机位置

    | opencv__dev_video0.png                                       | opencv__dev_video2.png                                       |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111739608.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509111739430.png?x-oss-process=style/resize) |



* 结合相机，再次遥操作：

  * 注意到`handeye`用的`index_or_path`是2，`fixed`用的是0
  * `display_data=true ` 表示打开`rerun.io`可视化（要在Ubuntu本机，不能是SSH远程）

  ```bash
   python -m lerobot.teleoperate \
      --robot.ip_address="localhost" \
      --robot.port=12345 \
      --robot.type=enpei_follower \
      --robot.id=enpei_follower \
      --robot.cameras="{ handeye: {type: opencv, index_or_path: 2, width: 320, height: 240, fps: 30}, fixed: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}" \
      --teleop.type=enpei_leader \
      --teleop.port=/dev/ttyACM0 \
      --teleop.id=enpei_leader \
      --fps=30\
      --display_data=true \
      --enpei_speed_mode=record
  ```

  一切正常的话，理应打开下图窗口：

  ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202509112156917.png?x-oss-process=style/wp)