import torch
import numpy as np
import pymeshlab as pml
from importlib.metadata import version

PML_VER = version('pymeshlab') 

# the code assumes the latest 2023.12 version, but we can patch older versions
if PML_VER.startswith('0.2'):
    # monkey patch for 0.2 (only the used functions in this file!)
    pml.MeshSet.meshing_decimation_quadric_edge_collapse = pml.MeshSet.simplification_quadric_edge_collapse_decimation
    pml.MeshSet.meshing_isotropic_explicit_remeshing = pml.MeshSet.remeshing_isotropic_explicit_remeshing
    pml.MeshSet.meshing_remove_unreferenced_vertices = pml.MeshSet.remove_unreferenced_vertices
    pml.MeshSet.meshing_merge_close_vertices = pml.MeshSet.merge_close_vertices
    pml.MeshSet.meshing_remove_duplicate_faces = pml.MeshSet.remove_duplicate_faces
    pml.MeshSet.meshing_remove_null_faces = pml.MeshSet.remove_zero_area_faces
    pml.MeshSet.meshing_remove_connected_component_by_diameter = pml.MeshSet.remove_isolated_pieces_wrt_diameter
    pml.MeshSet.meshing_remove_connected_component_by_face_number = pml.MeshSet.remove_isolated_pieces_wrt_face_num
    pml.MeshSet.meshing_repair_non_manifold_edges = pml.MeshSet.repair_non_manifold_edges_by_removing_faces
    pml.MeshSet.meshing_repair_non_manifold_vertices = pml.MeshSet.repair_non_manifold_vertices_by_splitting
    pml.PercentageValue = pml.Percentage
    pml.PureValue = float
elif PML_VER.startswith('2022.2'):
    # monkey patch for 2022.2
    pml.PercentageValue = pml.Percentage
    pml.PureValue = pml.AbsoluteValue

def rotation_matrix(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle_rad), -np.sin(angle_rad)],
                         [0, np.sin(angle_rad), np.cos(angle_rad)]]).astype(np.float32)
    elif axis == 'y':
        return np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                         [0, 1, 0],
                         [-np.sin(angle_rad), 0, np.cos(angle_rad)]]).astype(np.float32)
    elif axis == 'z':
        return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                         [np.sin(angle_rad), np.cos(angle_rad), 0],
                         [0, 0, 1]]).astype(np.float32)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def scale_to_unit_sphere(points):
    max_xyz, _ = points.max(0)
    min_xyz, _ = points.min(0)
    bb_centroid = (max_xyz + min_xyz) / 2.
    zero_mean_points = points - bb_centroid
    dist = np.linalg.norm(points, axis=1)
    normalized_points = zero_mean_points / np.max(dist)
    return normalized_points

def scale_to_unit_cube(points):
    max_xyz, _ = points.max(0)
    min_xyz, _ = points.min(0)
    bb_centroid = (max_xyz + min_xyz) / 2.
    global_scale_max = (max_xyz - min_xyz).max()
    zero_mean_points = points - bb_centroid
    normalized_points = zero_mean_points * (1.8 / global_scale_max)
    return normalized_points

def decimate_mesh(
    verts, faces, target=5e4, backend="pymeshlab", remesh=False, optimalplacement=True
):
    """ perform mesh decimation.

    Args:pml
        verts (np.ndarray): mesh vertices, float [N, 3]
        faces (np.ndarray): mesh faces, int [M, 3]
        target (int): targeted number of faces
        backend (str, optional): algorithm backend, can be "pymeshlab" or "pyfqmr". Defaults to "pymeshlab".
        remesh (bool, optional): whether to remesh after decimation. Defaults to False.
        optimalplacement (bool, optional): For flat mesh, use False to prevent spikes. Defaults to True.

    Returns:
        Tuple[np.ndarray]: vertices and faces after decimation.
    """

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        m.compact()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
    remesh_iters=3,
):
    """ perform mesh cleaning, including floater removal, non manifold repair, and remeshing.

    Args:
        verts (np.ndarray): mesh vertices, float [N, 3]
        faces (np.ndarray): mesh faces, int [M, 3]
        v_pct (int, optional): percentage threshold to merge close vertices. Defaults to 1.
        min_f (int, optional): maximal number of faces for isolated component to remove. Defaults to 64.
        min_d (int, optional): maximal diameter percentage of isolated component to remove. Defaults to 20.
        repair (bool, optional): whether to repair non-manifold faces (cannot gurantee). Defaults to True.
        remesh (bool, optional): whether to perform a remeshing after all cleaning. Defaults to True.
        remesh_size (float, optional): the targeted edge length for remeshing. Defaults to 0.01.
        remesh_iters (int, optional): the iterations of remeshing. Defaults to 3.

    Returns:
        Tuple[np.ndarray]: vertices and faces after decimation.
    """
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=remesh_iters, targetlen=pml.PureValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    m.compact()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces

@torch.no_grad()
def compute_edge_to_face_mapping(faces):
    """ compute edge to face mapping.

    Args:
        faces (torch.Tensor): mesh faces, int [M, 3]

    Returns:
        torch.Tensor: indices to faces for each edge, long, [N, 2]
    """
    # Get unique edges
    # Create all edges, packed by triangle
    all_edges = torch.cat((
        torch.stack((faces[:, 0], faces[:, 1]), dim=-1),
        torch.stack((faces[:, 1], faces[:, 2]), dim=-1),
        torch.stack((faces[:, 2], faces[:, 0]), dim=-1),
    ), dim=-1).view(-1, 2)

    # Swap edge order so min index is always first
    order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
    sorted_edges = torch.cat((
        torch.gather(all_edges, 1, order),
        torch.gather(all_edges, 1, 1 - order)
    ), dim=-1)

    # Elliminate duplicates and return inverse mapping
    unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

    tris = torch.arange(faces.shape[0]).repeat_interleave(3).cuda()

    tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

    # Compute edge to face table
    mask0 = order[:,0] == 0
    mask1 = order[:,0] == 1
    tris_per_edge[idx_map[mask0], 0] = tris[mask0]
    tris_per_edge[idx_map[mask1], 1] = tris[mask1]

    return tris_per_edge