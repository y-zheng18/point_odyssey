import json

import Imath
import OpenEXR
from typing import Dict, Sequence, Tuple, Union
import numpy as np
from pathlib import Path
import os
import cv2
import sklearn
import collections
from tqdm import tqdm
from utils.file_io import *

def read_channels_from_exr(exr: OpenEXR.InputFile, channel_names: Sequence[str]) -> np.ndarray:
  """Reads a single channel from an EXR file and returns it as a numpy array."""
  channels_header = exr.header()["channels"]
  window = exr.header()["dataWindow"]
  width = window.max.x - window.min.x + 1
  height = window.max.y - window.min.y + 1
  outputs = []
  for channel_name in channel_names:
    channel_type = channels_header[channel_name].type.v
    numpy_type = {
        Imath.PixelType.HALF: np.float16,
        Imath.PixelType.FLOAT: np.float32,
        Imath.PixelType.UINT: np.uint32,
    }[channel_type]
    array = np.frombuffer(exr.channel(channel_name), numpy_type)
    array = array.reshape([height, width])
    outputs.append(array)
  return np.stack(outputs, axis=-1)

def get_render_layers_from_exr(filename) -> Dict[str, np.ndarray]:
  exr = OpenEXR.InputFile(str(filename))
  layer_names = set()
  for n, _ in exr.header()["channels"].items():
    layer_name, _, _ = n.partition(".")
    layer_names.add(layer_name)

  output = {}
  if "Image" in layer_names:
    # Image is in RGBA format with range [0, inf]
    output["linear_rgba"] = read_channels_from_exr(exr, ["Image.R", "Image.G",
                                                         "Image.B", "Image.A"])
  if "Depth" in layer_names:
    # range [0, 10000000000.0]  # the value 1e10 is used for background / infinity
    output["depth"] = read_channels_from_exr(exr, ["Depth.V"])
  if "Vector" in layer_names:
    flow = read_channels_from_exr(exr, ["Vector.R", "Vector.G", "Vector.B", "Vector.A"])
    # Blender exports forward and backward flow in a single image,
    # and uses (-delta_col, delta_row) format, but we prefer (delta_row, delta_col)
    output["backward_flow"] = np.zeros_like(flow[..., :2])
    output["backward_flow"][..., 0] = flow[..., 1]
    output["backward_flow"][..., 1] = -flow[..., 0]

    output["forward_flow"] = np.zeros_like(flow[..., 2:])
    output["forward_flow"][..., 0] = flow[..., 3]
    output["forward_flow"][..., 1] = -flow[..., 2]

  if "Normal" in layer_names:
    # range: [-1, 1]
    output["normal"] = read_channels_from_exr(exr, ["Normal.X", "Normal.Y", "Normal.Z"])

  if "UV" in layer_names:
    # range [0, 1]
    output["uv"] = read_channels_from_exr(exr, ["UV.X", "UV.Y", "UV.Z"])

  if "CryptoObject00" in layer_names:
    # CryptoMatte stores the segmentation of Objects using two kinds of channels:
    #  - index channels (uint32) specify the object index for a pixel
    #  - alpha channels (float32) specify the corresponding mask value
    # there may be many cryptomatte layers, which allows encoding a pixel as belonging to multiple
    # objects at once (up to a maximum of # of layers many objects per pixel)
    # In the EXR this is stored with 2 layers per RGBA image  (CryptoObject00, CryptoObject01, ...)
    # with RG being the first layer and BA being the second
    # So the R and B channels are uint32 and the G and A channels are float32.
    crypto_layers = [n for n in layer_names if n.startswith("CryptoObject")]
    index_channels = [n + "." + c for n in crypto_layers for c in "RB"]
    idxs = read_channels_from_exr(exr, index_channels)
    idxs.dtype = np.uint32
    output["segmentation_indices"] = idxs
    alpha_channels = [n + "." + c for n in crypto_layers for c in "GA"]
    alphas = read_channels_from_exr(exr, alpha_channels)
    output["segmentation_alphas"] = alphas
  if "ObjectCoordinates" in layer_names:
    output["object_coordinates"] = read_channels_from_exr(exr,
      ["ObjectCoordinates.R", "ObjectCoordinates.G", "ObjectCoordinates.B"])
  return output

def mm3hash(name):
  """ Compute the uint32 hash that Blenders Cryptomatte uses.
  https://github.com/Psyop/Cryptomatte/blob/master/specification/cryptomatte_specification.pdf
  """
  hash_32 = sklearn.utils.murmurhash3_32(name, positive=True)
  exp = hash_32 >> 23 & 255
  if exp in (0, 255):
    hash_32 ^= 1 << 23
  return hash_32

def replace_cryptomatte_hashes_by_asset_index(
    segmentation_ids: None,
    assets: list):
  """Replace (inplace) the cryptomatte hash (from Blender) by the index of each asset + 1.
  (the +1 is to ensure that the 0 for background does not interfere with asset index 0)

  Args:
    segmentation_ids: Segmentation array of cryptomatte hashes as returned by Blender.
    assets: List of assets to use for replacement.
  """
  # replace crypto-ids with asset index
  new_segmentation_ids = np.zeros_like(segmentation_ids)
  for idx, asset in enumerate(assets, start=1):
    #print(asset)
    asset_hash = mm3hash(asset)
    new_segmentation_ids[segmentation_ids == asset_hash] = idx
  return new_segmentation_ids


def process_depth(exr_layers, cam_info):
  # blender returns z values (distance to camera plane)
  # convert them into depth (distance to camera center)
  def _z_to_depth(z, cam_info):
      z = np.array(z)
      assert z.ndim >= 3
      h, w, _ = z.shape[-3:]

      pixel_centers_x = (np.arange(-w / 2, w / 2, dtype=np.float32) + 0.5) / w * cam_info['sensor_width']
      pixel_centers_y = (np.arange(-h / 2, h / 2, dtype=np.float32) + 0.5) / h * cam_info['sensor_height']
      squared_distance_from_center = np.sum(np.square(np.meshgrid(
          pixel_centers_x,  # X-Axis (columns)
          pixel_centers_y,  # Y-Axis (rows)
          indexing="xy",
      )), axis=0)

      depth_scaling = np.sqrt(1 + squared_distance_from_center / cam_info['focal_length'] ** 2)
      depth_scaling = depth_scaling.reshape((1,) * (z.ndim - 3) + depth_scaling.shape + (1,))
      return z * depth_scaling
  return exr_layers["depth"] # _z_to_depth(exr_layers["depth"], cam_info)


def process_z(exr_layers, scene):  # pylint: disable=unused-argument
  # blender returns z values (distance to camera plane)
  return exr_layers["depth"]


def process_backward_flow(exr_layers, scene):  # pylint: disable=unused-argument
  return exr_layers["backward_flow"]


def process_forward_flow(exr_layers, scene):  # pylint: disable=unused-argument
  return exr_layers["forward_flow"]


def process_uv(exr_layers, scene):  # pylint: disable=unused-argument
  # convert range [0, 1] to uint16
  return (exr_layers["uv"].clip(0.0, 1.0) * 65535).astype(np.uint16)


def process_normal(exr_layers, scene):  # pylint: disable=unused-argument
  # convert range [-1, 1] to uint16
  return ((exr_layers["normal"].clip(-1.0, 1.0) + 1) * 65535 / 2
          ).astype(np.uint16)


def process_object_coordinates(exr_layers, scene):  # pylint: disable=unused-argument
  # sometimes these values can become ever so slightly negative (e.g. 1e-10)
  # we clip them to [0, 1] to guarantee this range for further processing.
  return (exr_layers["object_coordinates"].clip(0.0, 1.0) * 65535
          ).astype(np.uint16)


def process_segementation(exr_layers, scene_info):  # pylint: disable=unused-argument
  # map the Blender cryptomatte hashes to asset indices
  return replace_cryptomatte_hashes_by_asset_index(
      exr_layers["segmentation_indices"][:, :, :1], scene_info['assets'])


def process_rgba(exr_layers, scene):  # pylint: disable=unused-argument
  # map the Blender cryptomatte hashes to asset indices
  return exr_layers["rgba"]


def process_rgb(exr_layers, scene):  # pylint: disable=unused-argument
  return exr_layers["rgba"][..., :3]


def postprocess(
      from_dir: str,
      return_layers: Sequence[str],
      batch_size: int,
      output_dir: str,
      frame_idx: int):
    post_processors = {
        "backward_flow": process_backward_flow,
        "forward_flow": process_forward_flow,
        "depth": process_depth,
        "z": process_z,
        "uv": process_uv,
        "normal": process_normal,
        "object_coordinates": process_object_coordinates,
        "segmentation": process_segementation,
        "rgb": process_rgb,
        "rgba": process_rgba}

    # --- collect all layers for all frames
    from_dir = epath.Path(from_dir)
    data_stack = collections.defaultdict(list)

    exr_frames = sorted((from_dir / "exr").glob("*.exr"))[frame_idx:]
    png_frames = [from_dir / "images" / (exr_filename.stem + ".png")
                  for exr_filename in exr_frames]
    scene_info = json.load(open(from_dir / 'scene_info.json', 'r'))
    scene_info['assets'] += ['Plane']
    #scene_info['assets'] = ['background'] + [i for i in scene_info['assets'] if 'robot' in i or 'Rock' in i or 'Cube' in i] # + ['Sphere'] # The last one is the background

    # output dir
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = frame_idx
    for exr_filename, png_filename in tqdm(zip(exr_frames, png_frames)):
      source_layers = get_render_layers_from_exr(exr_filename)
      # Use the contrast-normalized PNG instead of the EXR for RGBA.
      source_layers["rgba"] = read_png(png_filename)
      for key in return_layers:
        post_processor = post_processors[key]
        data_stack[key].append(post_processor(source_layers, scene_info))
      if len(data_stack[return_layers[0]]) == batch_size:
        save_data = {key: np.stack(data_stack[key], axis=0)
                     for key in return_layers}
        write_image_dict(save_data, output_dir, frame_idx=frame_idx)
        for key in return_layers:
          data_stack[key] = []

        data_stack = collections.defaultdict(list)
        frame_idx += batch_size
    save_data = {key: np.stack(data_stack[key], axis=0)
                 for key in return_layers}
    if len(save_data[return_layers[0]]) > 0:
      write_image_dict(save_data, output_dir, frame_idx=frame_idx)
    return

def write_image_dict(data_dict: Dict[str, np.ndarray], directory: str,
                     file_templates: Dict[str, str] = (), max_write_threads=16, frame_idx=0):
  for key, data in data_dict.items():
    if key in file_templates:
      DEFAULT_WRITERS[key](data, directory, file_template=file_templates[key],
                           max_write_threads=max_write_threads, frame_idx=frame_idx)
    else:
      DEFAULT_WRITERS[key](data, directory, max_write_threads=max_write_threads, frame_idx=frame_idx)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract EXR files.')
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='/Users/yangzheng/code/project/renderer/results/robot/')
    parser.add_argument('--output_dir', type=str, metavar='PATH',
                        default='/Users/yangzheng/code/project/renderer/results/robot/exr_img',
                        help='img save dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frame_idx', type=int, default=0)
    args = parser.parse_args()
    print("args:{0}".format(args))

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dict = postprocess(from_dir=args.data_dir,
                  return_layers=("rgb", #"backward_flow", "forward_flow",
                                 "depth",
                                 "normal", # "object_coordinates",
                                 "segmentation"), batch_size=args.batch_size,
                                  output_dir=args.output_dir,
                              frame_idx=args.frame_idx)

    # write_image_dict(frames_dict, args.output_dir)
