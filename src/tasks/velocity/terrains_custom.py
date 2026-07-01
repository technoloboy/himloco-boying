"""Custom terrains for the velocity task.

This module defines a "rough sloped" terrain: a pyramid slope with random-uniform
roughness noise added on top. BOTH the slope steepness AND the noise amplitude scale
with the difficulty parameter, so the terrain grows harder along a curriculum.

The implementation mirrors the mjlab source terrains so the generated mesh, hfield
normalization, coloring, flat-patch sampling, and TerrainOutput are handled identically:
  - `HfPyramidSlopedTerrainCfg.function` (slope scales with difficulty)
  - `HfRandomUniformTerrainCfg.function` (uniform noise, here scaled by difficulty)
"""

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.interpolate as interpolate

from mjlab.terrains.heightfield_terrains import (
  HfPyramidSlopedTerrainCfg,
  _compute_flat_patches,
  color_by_height,
)
from mjlab.terrains.terrain_generator import (
  TerrainGeometry,
  TerrainOutput,
)


@dataclass(kw_only=True)
class HfRoughSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
  """Pyramid slope + random-uniform roughness, both scaling with difficulty."""

  noise_range: tuple[float, float] = (0.0, 0.06)
  """Min and max roughness height noise, in meters. The effective range is scaled
  by difficulty so roughness grows along the curriculum."""
  noise_step: float = 0.005
  """Height quantization step, in meters. Sampled heights are multiples of this
  value within the (difficulty-scaled) noise range."""
  downsampled_scale: float = 0.2
  """Spacing between randomly sampled roughness points before interpolation, in
  meters. Must be >= horizontal_scale."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    if self.inverted:
      slope = -self.slope_range[0] - difficulty * (
        self.slope_range[1] - self.slope_range[0]
      )
    else:
      slope = self.slope_range[0] + difficulty * (
        self.slope_range[1] - self.slope_range[0]
      )

    if self.border_width > 0 and self.border_width < self.horizontal_scale:
      raise ValueError(
        f"Border width ({self.border_width}) must be >= horizontal scale "
        f"({self.horizontal_scale})"
      )

    if self.downsampled_scale < self.horizontal_scale:
      raise ValueError(
        f"Downsampled scale must be >= horizontal scale: "
        f"{self.downsampled_scale} < {self.horizontal_scale}"
      )
    downsampled_scale = self.downsampled_scale

    border_pixels = int(self.border_width / self.horizontal_scale)
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)

    inner_width_pixels = width_pixels - 2 * border_pixels
    inner_length_pixels = length_pixels - 2 * border_pixels

    noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    # --- Pyramid slope field (matches HfPyramidSlopedTerrainCfg.function) ---
    if border_pixels > 0:
      height_max = int(
        slope * (inner_width_pixels * self.horizontal_scale) / 2 / self.vertical_scale
      )

      center_x = int(inner_width_pixels / 2)
      center_y = int(inner_length_pixels / 2)

      x = np.arange(0, inner_width_pixels)
      y = np.arange(0, inner_length_pixels)
      xx, yy = np.meshgrid(x, y, sparse=True)

      xx = (center_x - np.abs(center_x - xx)) / center_x
      yy = (center_y - np.abs(center_y - yy)) / center_y

      xx = xx.reshape(inner_width_pixels, 1)
      yy = yy.reshape(1, inner_length_pixels)

      hf_raw = height_max * xx * yy

      platform_width = int(self.platform_width / self.horizontal_scale / 2)
      x_pf = inner_width_pixels // 2 - platform_width
      y_pf = inner_length_pixels // 2 - platform_width
      z_pf = hf_raw[x_pf, y_pf] if x_pf >= 0 and y_pf >= 0 else 0
      hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

      noise[
        border_pixels : -border_pixels if border_pixels else width_pixels,
        border_pixels : -border_pixels if border_pixels else length_pixels,
      ] = np.rint(hf_raw).astype(np.int16)
    else:
      height_max = int(slope * self.size[0] / 2 / self.vertical_scale)

      center_x = int(width_pixels / 2)
      center_y = int(length_pixels / 2)

      x = np.arange(0, width_pixels)
      y = np.arange(0, length_pixels)
      xx, yy = np.meshgrid(x, y, sparse=True)

      xx = (center_x - np.abs(center_x - xx)) / center_x
      yy = (center_y - np.abs(center_y - yy)) / center_y

      xx = xx.reshape(width_pixels, 1)
      yy = yy.reshape(1, length_pixels)

      hf_raw = height_max * xx * yy

      platform_width = int(self.platform_width / self.horizontal_scale / 2)
      x_pf = width_pixels // 2 - platform_width
      y_pf = length_pixels // 2 - platform_width
      z_pf = hf_raw[x_pf, y_pf]
      hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

      noise = np.rint(hf_raw).astype(np.int16)

    # --- Random-uniform roughness field (matches HfRandomUniformTerrainCfg.function),
    # but with the effective noise_range scaled by difficulty so it grows with
    # the curriculum. Added element-wise onto the slope field.
    effective_noise_min = self.noise_range[0] * difficulty
    effective_noise_max = self.noise_range[1] * difficulty

    height_min = int(effective_noise_min / self.vertical_scale)
    height_max_noise = int(effective_noise_max / self.vertical_scale)
    height_step = int(self.noise_step / self.vertical_scale)

    if border_pixels > 0:
      inner_size = (
        inner_width_pixels * self.horizontal_scale,
        inner_length_pixels * self.horizontal_scale,
      )

      width_downsampled = int(inner_size[0] / downsampled_scale)
      length_downsampled = int(inner_size[1] / downsampled_scale)

      height_range = np.arange(height_min, height_max_noise + height_step, height_step)
      height_field_downsampled = rng.choice(
        height_range, size=(width_downsampled, length_downsampled)
      )

      x = np.linspace(0, inner_size[0], width_downsampled)
      y = np.linspace(0, inner_size[1], length_downsampled)
      func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

      x_upsampled = np.linspace(0, inner_size[0], inner_width_pixels)
      y_upsampled = np.linspace(0, inner_size[1], inner_length_pixels)
      z_upsampled = func(x_upsampled, y_upsampled)

      roughness = np.zeros((width_pixels, length_pixels), dtype=np.int16)
      roughness[
        border_pixels : -border_pixels if border_pixels else width_pixels,
        border_pixels : -border_pixels if border_pixels else length_pixels,
      ] = np.rint(z_upsampled).astype(np.int16)
    else:
      width_downsampled = int(self.size[0] / downsampled_scale)
      length_downsampled = int(self.size[1] / downsampled_scale)

      height_range = np.arange(height_min, height_max_noise + height_step, height_step)
      height_field_downsampled = rng.choice(
        height_range, size=(width_downsampled, length_downsampled)
      )

      x = np.linspace(0, self.size[0], width_downsampled)
      y = np.linspace(0, self.size[1], length_downsampled)
      func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

      x_upsampled = np.linspace(0, self.size[0], width_pixels)
      y_upsampled = np.linspace(0, self.size[1], length_pixels)
      z_upsampled = func(x_upsampled, y_upsampled)
      roughness = np.rint(z_upsampled).astype(np.int16)

    # Combine slope + roughness (element-wise, same shape/dtype).
    noise = (noise.astype(np.int32) + roughness.astype(np.int32)).astype(np.int16)

    # --- Downstream hfield creation / normalization / color / flat patches ---
    # (matches HfPyramidSlopedTerrainCfg.function).
    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )

    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio

    if elevation_range > 0:
      normalized_elevation = (noise - elevation_min) / elevation_range
    else:
      normalized_elevation = np.zeros_like(noise)

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        base_thickness,
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
    )

    if self.inverted:
      hfield_z_offset = -max_physical_height
    else:
      hfield_z_offset = 0

    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[
        self.size[0] / 2,
        self.size[1] / 2,
        hfield_z_offset,
      ],
      material=material_name,
    )

    if self.inverted:
      spawn_height = hfield_z_offset
    else:
      spawn_height = max_physical_height

    origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

    flat_patches = _compute_flat_patches(
      noise,
      self.vertical_scale,
      self.horizontal_scale,
      hfield_z_offset,
      self.flat_patch_sampling,
      rng,
    )

    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)
