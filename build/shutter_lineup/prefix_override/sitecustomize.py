import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eyi3/ros2_ws/src/shutter_lineup/install/shutter_lineup'
