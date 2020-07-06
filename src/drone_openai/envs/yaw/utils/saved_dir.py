#! /usr/bin/env python

import os
import rospkg
rospack = rospkg.RosPack()

model_dir = os.path.join(rospack.get_path("drone_openai"), "envs/models/")
log_dir = os.path.join(rospack.get_path("drone_openai"), "envs/logs/")