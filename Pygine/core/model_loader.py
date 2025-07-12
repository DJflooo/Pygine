
import numpy as np
import os

def load_obj_model(file_path: str) -> tuple[list[float], list[int]]:
    """
    Loads a 3D model from an OBJ file.
    Assumes vertex data is interleaved: position (3f), normal (3f), UV (2f).
    Handles 'v', 'vn', 'vt', and 'f' (triangles only, with v/vt/vn indexing).

    :param file_path: Path to the .obj file.
    :return: A tuple containing (interleaved_vertices, indices).
             interleaved_vertices: [x, y, z, nx, ny, nz, u, v, ...]
             indices: [i1, i2, i3, ...] for triangle faces.
    """
    vertices = []
    normals = []
    uvs = []
    faces = [] # Stores (v_idx, vt_idx, vn_idx) tuples for each vertex in a face

    if not os.path.exists(file_path):
        print(f"Error: OBJ file not found at {file_path}")
        return [], []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                prefix = parts[0]
                if prefix == 'v':
                    vertices.append([float(p) for p in parts[1:4]])
                elif prefix == 'vn':
                    normals.append([float(p) for p in parts[1:4]])
                elif prefix == 'vt':
                    uvs.append([float(p) for p in parts[1:3]])
                elif prefix == 'f':
                    # Faces can be triangles (3 vertices) or quads (4 vertices).
                    # We'll triangulate quads if encountered.
                    face_verts = []
                    for part in parts[1:]:
                        # OBJ indices are 1-based, so subtract 1
                        # Format: v/vt/vn or v//vn or v/vt
                        v_idx, vt_idx, vn_idx = 0, 0, 0
                        sub_parts = part.split('/')
                        v_idx = int(sub_parts[0]) - 1 # Position index (mandatory)
                        if len(sub_parts) > 1 and sub_parts[1]: # Check if vt is present
                            vt_idx = int(sub_parts[1]) - 1
                        if len(sub_parts) > 2 and sub_parts[2]: # Check if vn is present
                            vn_idx = int(sub_parts[2]) - 1
                        elif len(sub_parts) == 2 and not sub_parts[1]: # Case: v//vn
                            vn_idx = int(sub_parts[1]) - 1 # This case is actually handled by the above if/else for sub_parts[1]
                                                            # If sub_parts[1] is empty, vt_idx remains 0, and this picks up vn_idx

                        face_verts.append((v_idx, vt_idx, vn_idx))
                    faces.append(face_verts)
    except Exception as e:
        print(f"Error parsing OBJ file {file_path}: {e}")
        return [], []

    # Now, assemble the interleaved vertex data and indices
    # We need to create a unique vertex for each unique combination of (v_idx, vt_idx, vn_idx)
    # This is because OpenGL requires a single index per vertex, and a vertex is unique
    # if any of its attributes (pos, normal, uv) differ.
    unique_vertex_data = [] # Stores [x, y, z, nx, ny, nz, u, v]
    unique_vertex_map = {}  # Maps (v_idx, vt_idx, vn_idx) tuple to its index in unique_vertex_data
    final_indices = []

    for face in faces:
        # Triangulate if face has more than 3 vertices (e.g., quads)
        # Simple fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
        for i in range(len(face) - 2):
            face_indices_to_process = [face[0], face[i+1], face[i+2]]

            for v_data_tuple in face_indices_to_process:
                if v_data_tuple not in unique_vertex_map:
                    v_idx, vt_idx, vn_idx = v_data_tuple

                    # Get position, normal, UV
                    pos = vertices[v_idx]
                    norm = normals[vn_idx] if normals and vn_idx < len(normals) else [0.0, 0.0, 0.0] # Default normal if not found
                    uv = uvs[vt_idx] if uvs and vt_idx < len(uvs) else [0.0, 0.0] # Default UV if not found

                    # Create interleaved vertex
                    interleaved_vertex = pos + norm + uv
                    unique_vertex_data.extend(interleaved_vertex)

                    # Store mapping and add to final indices
                    new_index = len(unique_vertex_map)
                    unique_vertex_map[v_data_tuple] = new_index
                    final_indices.append(new_index)
                else:
                    # Vertex combination already exists, reuse its index
                    final_indices.append(unique_vertex_map[v_data_tuple])

    return unique_vertex_data, final_indices

