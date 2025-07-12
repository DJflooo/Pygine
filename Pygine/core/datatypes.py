################# Here, all the custom datatypes are stored #################
# float3 -> three float values. Most used for position/rotation/scale #######
# float4 -> four float values. Most used for RGBA and Quaternions ###########
# AABB -> data for Collision boxes. Just go see the explanation there #######
# OBB -> data for Oriented Bounding Boxes. More precise for rotated objects #
#############################################################################

import math
import numpy as np
import struct

# --- Helper function for OBB: Convert Euler angles to a 3x3 rotation matrix ---
# This will be crucial for defining the OBB's orientation from an obj's rotation.
def matrix_from_euler_angles(rotation_x_deg: float, rotation_y_deg: float, rotation_z_deg: float) -> np.ndarray:
    """
    Creates a 3x3 rotation matrix from Euler angles (in degrees, YXZ order for consistency with camera).
    This order means Yaw (Y) first, then Pitch (X), then Roll (Z).
    """
    # Convert degrees to radians
    rx = math.radians(rotation_x_deg)
    ry = math.radians(rotation_y_deg)
    rz = math.radians(rotation_z_deg)

    # Rotation matrices for each axis
    # X-axis rotation (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ], dtype='f4')

    # Y-axis rotation (Yaw)
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ], dtype='f4')

    # Z-axis rotation (Roll)
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ], dtype='f4')

    # Combine rotations in YXZ order (Yaw * Pitch * Roll)
    # This order is common in game engines (like Unity's default Euler interpretation)
    # and often works well for camera/object rotations.
    return Ry @ Rx @ Rz


class float3:
    """
    Contains three float values (x, y, z). Mostly used for position, rotation and scale.
    """
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        if isinstance(other, float3):
            return float3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return float3(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, float3):
            return float3(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return float3(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float3):
            # Element-wise multiplication (Hadamard product)
            return float3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return float3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, np.ndarray) and other.shape == (3, 3):
            # Vector-matrix multiplication (treat float3 as column vector)
            # result = other @ [self.x, self.y, self.z].T
            # For a 3x3 matrix, this is a rotation/scaling of the vector
            vec_np = np.array([self.x, self.y, self.z], dtype='f4')
            transformed_vec = other @ vec_np
            return float3(transformed_vec[0], transformed_vec[1], transformed_vec[2])
        else:
            return NotImplemented

    def __rmul__(self, other): # For when scalar is on the left (e.g., 5 * float3)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, float3):
            # Element-wise division
            return float3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return float3(self.x / other, self.y / other, self.z / other)
        else:
            return NotImplemented
    
    def __neg__(self):
        return float3(-self.x, -self.y, -self.z)

    def dot(self, other: 'float3') -> float:
        """Calculates the dot product with another float3 vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        """Calculates the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'float3':
        """Returns a new normalized vector (unit vector)."""
        len_val = self.length()
        if len_val == 0:
            return float3(0, 0, 0) # Or raise an error
        return float3(self.x / len_val, self.y / len_val, self.z / len_val)

    def cross(self, other: 'float3') -> 'float3':
        """Calculates the cross product with another float3 vector."""
        return float3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __repr__(self):
        return f"float3({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def length_sq(self) -> float:
        """Calculates the squared magnitude (length squared) of the vector."""
        return self.x**2 + self.y**2 + self.z**2
    
    def to_list(self) -> list[float]:
        """Converts the vector to a list of its components."""
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        """Converts the vector to a tuple of its components."""
        return (self.x, self.y, self.z)

    def copy(self) -> 'float3':
        """Returns a new float3 instance with the same values."""
        return float3(self.x, self.y, self.z)

    def to_bytes(self):
        return struct.pack('fff', self.x, self.y, self.z) # 'fff' for 3 floats

class float4:
    """
    Contains four float values (x, y, z, w). Commonly used for RGBA colors (Red, Green, Blue, Alpha) or Quaternions.
    """
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w) # The fourth component, e.g., alpha for color or w for quaternion

    def __add__(self, other):
        if isinstance(other, float4):
            return float4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
        elif isinstance(other, (int, float)):
            return float4(self.x + other, self.y + other, self.z + other, self.w + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, float4):
            return float4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
        elif isinstance(other, (int, float)):
            return float4(self.x - other, self.y - other, self.z - other, self.w - other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float4):
            # Element-wise multiplication (Hadamard product)
            return float4(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return float4(self.x * other, self.y * other, self.z * other, self.w * other)
        # Note: No matrix multiplication for float4 here unless explicitly for 4x4 matrices.
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, float4):
            # Element-wise division
            return float4(self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return float4(self.x / other, self.y / other, self.z / other, self.w / other)
        else:
            return NotImplemented
    
    def __neg__(self):
        return float4(-self.x, -self.y, -self.z, -self.w)

    def dot(self, other: 'float4') -> float:
        """Calculates the dot product with another float4 vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def length(self) -> float:
        """Calculates the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalize(self) -> 'float4':
        """Returns a new normalized vector (unit vector)."""
        len_val = self.length()
        if len_val == 0:
            return float4(0, 0, 0, 0)
        return float4(self.x / len_val, self.y / len_val, self.z / len_val, self.w / len_val)
    
    def __repr__(self):
        return f"float4({self.x}, {self.y}, {self.z}, {self.w})"

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, {self.w:.2f})"
    
    def length_sq(self) -> float:
        """Calculates the squared magnitude (length squared) of the vector."""
        return self.x**2 + self.y**2 + self.z**2 + self.w**2
    
    def to_list(self) -> list[float]:
        """Converts the vector to a list of its components."""
        return [self.x, self.y, self.z, self.w]

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Converts the vector to a tuple of its components."""
        return (self.x, self.y, self.z, self.w)

    def copy(self) -> 'float4':
        """Returns a new float4 instance with the same values."""
        return float4(self.x, self.y, self.z, self.w)
    
    def to_bytes(self):
        return struct.pack('ffff', self.x, self.y, self.z, self.w) # 'ffff' for 4 floats
    
class AABB:
    """
    Represents an Axis-Aligned Bounding Box in 3D space.
    Stored in local space, relative to the object's origin (by default).
    """
    def __init__(self, min_extents: float3, max_extents: float3):
        # Ensure min_extents are truly minimal and max_extents are maximal
        self.min_extents = float3(
            min(min_extents.x, max_extents.x),
            min(min_extents.y, max_extents.y),
            min(min_extents.z, max_extents.z)
        )
        self.max_extents = float3(
            max(min_extents.x, max_extents.x),
            max(min_extents.y, max_extents.y),
            max(min_extents.z, max_extents.z)
        )

    @classmethod
    def from_center_and_size(cls, center: float3, size: float3):
        """
        Creates an AABB from a center point (local) and its full size (width, height, depth).
        The 'center' here refers to the center of the bounding box relative to the object's origin.
        For a cube centered at (0,0,0) with size (2,2,2), center would be (0,0,0) and size (2,2,2).
        """
        half_size = size / 2.0
        min_extents = center - half_size
        max_extents = center + half_size
        return cls(min_extents, max_extents)

    @property
    def center(self) -> float3:
        return (self.min_extents + self.max_extents) * 0.5

    @property
    def size(self) -> float3:
        return self.max_extents - self.min_extents
    
    def contains_point(self, point: float3) -> bool:
        return (point.x >= self.min_extents.x and point.x <= self.max_extents.x and
                point.y >= self.min_extents.y and point.y <= self.max_extents.y and
                point.z >= self.min_extents.z and point.z <= self.max_extents.z)

    def intersects(self, other_aabb: 'AABB') -> bool:
        # Check for overlap on all three axes
        overlap_x = (self.min_extents.x <= other_aabb.max_extents.x and 
                     self.max_extents.x >= other_aabb.min_extents.x)
        overlap_y = (self.min_extents.y <= other_aabb.max_extents.y and 
                     self.max_extents.y >= other_aabb.min_extents.y)
        overlap_z = (self.min_extents.z <= other_aabb.max_extents.z and 
                     self.max_extents.z >= other_aabb.min_extents.z)
        
        return overlap_x and overlap_y and overlap_z

    def __repr__(self):
        return f"AABB(min={self.min_extents}, max={self.max_extents})"


class OBB:
    """
    Represents an Oriented Bounding Box in 3D space.
    Defined by its center, half-extents along its local axes, and its orientation matrix.
    """
    def __init__(self, center: float3, half_extents: float3, orientation: np.ndarray):
        """
        :param center: The world-space center of the OBB.
        :param half_extents: A float3 representing the half-size along the OBB's local X, Y, Z axes.
        :param orientation: A 3x3 NumPy array representing the rotation matrix of the OBB's local axes
                            relative to the world axes. Columns are the normalized local X, Y, Z axes.
        """
        self.center = center
        self.half_extents = half_extents
        # Ensure orientation is a 3x3 matrix
        if not isinstance(orientation, np.ndarray) or orientation.shape != (3, 3):
            raise ValueError("Orientation must be a 3x3 NumPy array (rotation matrix).")
        self.orientation = orientation
    
    @classmethod
    def from_center_half_extents_and_rotation_matrix(cls, center: float3, half_extents: float3, rotation_matrix: np.ndarray):
        """
        Convenience constructor for creating an OBB.
        """
        return cls(center, half_extents, rotation_matrix)

    @classmethod
    def from_aabb_and_transform(cls, aabb: AABB, position: float3, world_rotation_matrix: np.ndarray, scale: float3):
        """
        Creates an OBB from a local-space AABB and an object's world transform.
        This is useful for converting an obj's default AABB collider into a world-space OBB.
        :param aabb: The local-space AABB (e.g., from obj.collider).
        :param position: The object's world position (float3).
        :param world_rotation_matrix: The object's 3x3 world-space rotation matrix (np.ndarray).
        :param scale: The object's world-space scale (float3).
        """
        # Calculate the AABB's local center and half-size
        local_center = aabb.center
        local_half_size = aabb.size / 2.0

        # Apply object's scale to the local half-size
        scaled_half_extents = float3(
            local_half_size.x * scale.x,
            local_half_size.y * scale.y,
            local_half_size.z * scale.z
        )

        # Transform the local center by the world rotation matrix and then translation to get world center
        # local_center is a float3, world_rotation_matrix is np.ndarray
        rotated_local_center_np = world_rotation_matrix @ np.array(local_center.to_list(), dtype='f4')
        world_center = position + float3(rotated_local_center_np[0], rotated_local_center_np[1], rotated_local_center_np[2])

        # Directly use the provided world_rotation_matrix for the OBB's orientation
        return cls(world_center, scaled_half_extents, world_rotation_matrix)

    def get_axes(self) -> list[float3]:
        """
        Returns the OBB's three normalized local axes (X, Y, Z) in world space.
        These are the columns of the orientation matrix.
        """
        # Columns of the orientation matrix are the local axes in world space
        axis_x = float3(self.orientation[0, 0], self.orientation[1, 0], self.orientation[2, 0]).normalize()
        axis_y = float3(self.orientation[0, 1], self.orientation[1, 1], self.orientation[2, 1]).normalize()
        axis_z = float3(self.orientation[0, 2], self.orientation[1, 2], self.orientation[2, 2]).normalize()
        return [axis_x, axis_y, axis_z]

    def get_vertices(self) -> list[float3]:
        """
        Calculates and returns the 8 world-space vertices of the OBB.
        """
        axes = self.get_axes()
        ax = axes[0] * self.half_extents.x
        ay = axes[1] * self.half_extents.y
        az = axes[2] * self.half_extents.z

        vertices = []
        # Generate all 8 combinations of +/- half-extents along each axis
        # Center + (ax * sx) + (ay * sy) + (az * sz) where sx,sy,sz are +/-1
        vertices.append(self.center + ax + ay + az)
        vertices.append(self.center + ax + ay - az)
        vertices.append(self.center + ax - ay + az)
        vertices.append(self.center + ax - ay - az)
        vertices.append(self.center - ax + ay + az)
        vertices.append(self.center - ax + ay - az)
        vertices.append(self.center - ax - ay + az)
        vertices.append(self.center - ax - ay - az)
        return vertices

    def __repr__(self):
        # For a more readable representation, especially for debugging
        return (f"OBB(center={self.center}, "
                f"half_extents={self.half_extents}, "
                f"orientation=\n{self.orientation})")

