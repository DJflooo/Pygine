
import numpy as np
from math import radians, sin, cos, tan
from core.datatypes import float3

class Camera:
    def __init__(self, position=float3(), rotation=float3(), fov=60.0, near=0.1, far=100.0):
        self.position = position # Camera's world position
        self.rotation = rotation # Camera's world rotation (Euler angles in degrees)
        self.fov = fov           # Field of View in degrees
        self.near = near         # Near clipping plane
        self.far = far           # Far clipping plane
    
    def create_view_matrix(self):
        """
        Creates the view matrix for the camera.
        This matrix transforms world coordinates into camera coordinates.
        It's essentially the inverse of the camera's 'model' matrix.
        """
        # Order of operations: Rotate, then Translate (when viewed from the object's perspective)
        # So, View = (Translate @ Rotate)^-1 = Rotate^-1 @ Translate^-1

        # Camera rotations (in radians)
        rot_x_rad = radians(self.rotation.x)
        rot_y_rad = radians(self.rotation.y)
        rot_z_rad = radians(self.rotation.z)

        # Camera's own "model" matrix
        cam_rot_x = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos(rot_x_rad), -sin(rot_x_rad), 0.0],
            [0.0, sin(rot_x_rad),  cos(rot_x_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        cam_rot_y = np.array([
            [cos(rot_y_rad), 0.0, sin(rot_y_rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin(rot_y_rad), 0.0, cos(rot_y_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        cam_rot_z = np.array([
            [cos(rot_z_rad), -sin(rot_z_rad), 0.0, 0.0],
            [sin(rot_z_rad),  cos(rot_z_rad), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        # For a standard look-at camera, it's often XYZ (pitch, then yaw) or YXZ (yaw, then pitch).
        # So, the matrix multiplication order will be: Z @ Y @ X
        combined_camera_rotation = cam_rot_z @ cam_rot_y @ cam_rot_x


        # Camera's translation matrix
        cam_translation_matrix = np.array([
            [1.0, 0.0, 0.0, self.position.x],
            [0.0, 1.0, 0.0, self.position.y],
            [0.0, 0.0, 1.0, self.position.z],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        # The camera's "model" matrix (where it *is* in the world)
        # We apply rotation, then translation, similar to how an object's model matrix is built.
        camera_model_matrix = cam_translation_matrix @ combined_camera_rotation

        # The view matrix is the inverse of the camera's model matrix
        view_matrix = np.linalg.inv(camera_model_matrix)
        return view_matrix
    
    def Move(self, x, y, z):
        """
        Moves the camera by a given delta along its own local axes.
        x: movement along local right/left axis (+X is right)
        y: movement along local up/down axis (+Y is up)
        z: movement along local forward/backward axis (+Z is backward in OpenGL/right-handed system, -Z is forward)
        
        This method applies horizontal movement relative to the camera's Y-rotation (yaw),
        and vertical (Y) movement along the world Y-axis.
        """
        # Get the camera's rotation matrix for horizontal movement (Y-axis only)
        # This aligns the X and Z movement with the camera's horizontal facing direction.
        rot_y_rad = radians(self.rotation.y)

        horizontal_orientation_matrix = np.array([
            [cos(rot_y_rad), 0.0, sin(rot_y_rad)],
            [0.0, 1.0, 0.0], # Not strictly needed if only handling XZ, but conceptually for full matrix
            [-sin(rot_y_rad), 0.0, cos(rot_y_rad)]
        ], dtype='f4')
        
        # Create a movement vector for horizontal components
        local_horizontal_movement = np.array([x, 0.0, z], dtype='f4')
        
        # Transform the local horizontal movement vector by the Y-rotation matrix
        transformed_horizontal_movement = horizontal_orientation_matrix @ local_horizontal_movement
        
        # Apply the transformed horizontal movement to the camera's global position
        self.position.x += transformed_horizontal_movement[0]
        self.position.z += transformed_horizontal_movement[2]
        
        # Vertical movement (Y) is applied directly, in world space.
        self.position.y += y
        # print(f"[Camera] moved to {self.position}", flush=True) # For debugging
    
    def get_forward_vector_flat(self) -> float3:
        """
        Calculates the camera's forward vector, flattened to the XZ plane (ignoring pitch).
        Returns a normalized float3 vector.
        """
        yaw_rad = radians(self.rotation.y)
        # In a right-handed system with +Z forward, -Z is often the 'look' direction.
        # If your engine's forward is -Z:
        x = -sin(yaw_rad)
        z = -cos(yaw_rad)
        # If your engine's forward is +Z:
        # x = sin(yaw_rad)
        # z = cos(yaw_rad)

        forward = float3(x, 0.0, z)
        return forward.normalize() if forward.length() > 1e-6 else float3(0, 0, -1) # Default if perfectly vertical

    def get_right_vector_flat(self) -> float3:
        """
        Calculates the camera's right vector, flattened to the XZ plane.
        Returns a normalized float3 vector.
        """
        yaw_rad = radians(self.rotation.y)
        # Right vector is typically (cos(yaw), 0, sin(yaw)) if forward is (-sin(yaw), 0, -cos(yaw))
        x = cos(yaw_rad)
        z = sin(yaw_rad) # Or -sin(yaw_rad) depending on coordinate system/handedness
        
        right = float3(x, 0.0, z)
        return right.normalize() if right.length() > 1e-6 else float3(1, 0, 0) # Default if perfectly vertical
    
    def Turn(self, x, y, z):
        """Rotates the camera."""
        # Just like obj, but consider that for FPS-style cameras:
        # X rotation (pitch) is typically limited to prevent flipping.
        # Y rotation (yaw) is typically unrestricted.
        # Z rotation (roll) is less common in FPS, more for flight sims.
        self.rotation.x += x
        self.rotation.y += y
        self.rotation.z += z
        # print(f"[Camera] rotated to {self.rotation}", flush=True)