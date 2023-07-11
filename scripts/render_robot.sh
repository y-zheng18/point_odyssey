#!/bin/sh

python export_annotation.py --scene_dir ./data/demo_scene/robot.blend --save_dir ./results/robot --rendering --samples_per_pixel 64  \
	--exr --export_obj \
	--export_tracking --vis_num 200 --sampling_character_num 5000 --sampling_scene_num 2000