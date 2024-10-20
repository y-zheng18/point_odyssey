#!/bin/sh

python export_annotation.py --scene_dir ./data/demo_scene/kitchen_gfloor.blend --save_dir ./results/indoor --rendering --samples_per_pixel 128 --add_fog --randomize \
	--exr --export_obj \
	--export_tracking --sampling_character_num 5000 --sampling_scene_num 2000