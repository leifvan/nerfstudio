from dataclasses import dataclass
from typing import List
import torch
from torch import nn

from jaxtyping import Float, Int, Shaped, Bool

from nerfstudio.cameras.rays import PiecewiseRayBundle
from nerfstudio.utils.tensor_dataclass import TensorDataclass

@dataclass
class RayIntersections(TensorDataclass):
    """Describes ray intersections."""

    t: Float[torch.Tensor, "*bsz 1"]
    """Intersection location along the ray."""

    normals: Float[torch.Tensor, "*bsz 3"]
    """Normal at the intersection location."""

    is_valid: Bool[torch.Tensor, "*bsz 1"]
    """Indicates if the intersection is valid."""

    def merge_with_(self, other: "RayIntersections") -> None:
        is_closer = other.is_valid & (other.t < self.t)                
        self.t[is_closer] = other.t
        self.normals[is_closer] = other.normals
        self.is_valid[is_closer] = True

    def reflect_(self, ray_bundle: PiecewiseRayBundle, bounce_idx: int) -> None:
        """In-place applies reflection to the given RayBundle."""
        *batch_size, num_segments, _ = ray_bundle.start_t.shape
        
        segment_idxs = torch.searchsorted(ray_bundle.start_t[..., 0], self.t)  # (*bsz,)
        expanded_mask = torch.zeros(num_segments, *batch_size, dtype=torch.bool)  # (num_segments, *bsz,)

        for i in range(num_segments):
            expanded_mask[i, i >= segment_idxs] = self.is_valid[..., 0]
        expanded_mask = expanded_mask.permute(*range(1, len(batch_size) + 1), 0) # (*bsz, num_segments)

        d_dot_n = (ray_bundle.directions[..., None, :] @ self.normals[..., None, :, None])[..., 0, 0]
        reflected_directions = ray_bundle.directions - 2 * d_dot_n * self.normals

        ray_bundle.directions[expanded_mask] = reflected_directions[expanded_mask]
        ray_bundle.start_t[expanded_mask] = self.t[..., None][expanded_mask]


class GeometryObject(nn.Module):
    def forward(self, ray_bundle: PiecewiseRayBundle) -> RayIntersections:
        """Computes the intersections with the geometry object."""
        raise NotImplementedError


class GeometryGroup(GeometryObject):
    def __init__(self, objects: List[GeometryObject]):
        super().__init__()
        self.objects: List[GeometryObject] = objects

    def forward(self, ray_bundle: PiecewiseRayBundle) -> RayIntersections:
        *batch_shape, _ = ray_bundle.directions.shape
        
        closest_intersections = self.objects[0].get_intersections(ray_bundle)
        
        for obj in self.objects[1:]:
            closest_intersections.merge_with(obj.get_intersections(ray_bundle))

        return closest_intersections


class Triangle(GeometryObject):
    def __init__(self, vertices: Float[torch.Tensor, "3 3"], one_sided: bool = True) -> None:
        super().__init__()
        self.vertices = vertices
        self.one_sided = one_sided

        self.edge_ab = self.vertices[1] - self.vertices[0]
        self.edge_ac = self.vertices[2] - self.vertices[0]
        self.normal = torch.cross(self.vertices[1] - self.vertices[0], self.vertices[2] - self.vertices[1])
        self.normal = self.normal / torch.linalg.norm(self.normal)

    def forward(self, ray_bundle: PiecewiseRayBundle) -> RayIntersections:
        # pick last origins & directions from bundle
        cur_directions = ray_bundle.directions[..., -1, :]
        cur_origins = ray_bundle.origins[..., -1, :]

        ray_cross_ac = torch.cross(cur_directions, self.edge_ac.expand_as(cur_directions)) # (*bsz, 3)
        det = ray_cross_ac @ self.edge_ab  # (*bsz,)

        # is ray parallel to triangle?
        is_invalid = torch.isclose(det, torch.zeros(1))  # (*bsz,)

        inv_det = 1.0 / det  # (*bsz,)
        s = cur_origins - self.vertices[0]  # (*bsz, 3)
        u = inv_det * (s[..., None, :] @ ray_cross_ac[..., :, None])[..., 0, 0]  # (*bsz,)

        s_cross_ab = torch.cross(s, self.edge_ab.expand_as(s))  # (*bsz, 3)
        v = inv_det * (cur_directions[..., None, :] @ s_cross_ab[..., :, None])[..., 0, 0]  # (*bsz,)

        # is ray outside triangle?
        is_invalid = is_invalid | (u < 0) | (u > 1)
        is_invalid = is_invalid | (v < 0) | ((u + v) > 1) # (*bsz,)
        
        if self.one_sided:
            # does ray point opposite to normal?
            ray_dot_n = cur_directions @ self.normal
            is_invalid = is_invalid | (ray_dot_n > 0)

        t = inv_det * (s_cross_ab @ self.edge_ac) # (*bsz,)
        t[is_invalid] = 0

        return RayIntersections(
            t=t,
            normals=self.normal.expand_as(cur_directions),
            is_valid=~is_invalid
        )
