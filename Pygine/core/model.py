
import moderngl
import struct
import numpy as np

class Model:
    def __init__(self, ctx: moderngl.Context, vertices: list[float], indices: list[int], program: moderngl.Program):
        """
        Initializes a Model with given vertex data, index data, and shader program.
        This version expects vertex data to be interleaved: [x, y, z, nx, ny, nz, u, v, ...]
        :param ctx: The ModernGL context from the engine.
        :param vertices: A flat list of floats for vertex positions, normals, AND UVs
                         (e.g., [vx1, vy1, vz1, nx1, ny1, nz1, u1, v1, ...]).
        :param indices: A flat list of integers for vertex indices (e.g., [i1, i2, i3, ...]).
        :param program: The ModernGL shader program to use for rendering this model.
        """
        self.ctx = ctx
        self.program = program 

        # Each vertex now has 3 positions (x,y,z), 3 normals (nx,ny,nz), and 2 UVs (u,v) = 8 floats per vertex
        self.vertex_size = 3 + 3 + 2 # position (3f) + normal (3f) + UV (2f) = 8 floats per vertex

        # Creates Vertex Buffer Object (VBO)
        self.vbo = self.ctx.buffer(struct.pack('f' * len(vertices), *vertices))

        # Creates Index Buffer Object (IBO)
        self.ibo = self.ctx.buffer(struct.pack('I' * len(indices), *indices))

        # Creates Vertex Array Object (VAO)
        # We now tell it to use 'in_vert' as vec3, 'in_normal' as vec3, and 'in_uv' as vec2.
        # '3f 3f 2f' means 3 floats for in_vert, then 3 floats for in_normal, then 2 floats for in_uv.
        # The stride is 8 * 4 bytes (8 floats * 4 bytes/float)
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 2f', 'in_vert', 'in_normal', 'in_uv')], # '3f' pos, '3f' normal, '2f' UV
            self.ibo
        )

    def render(self):
        """Renders the model."""
        self.vao.render(mode=moderngl.TRIANGLES)

    def release(self):
        """Frees OpenGL resources associated with this model."""
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        # Note: We don't release the program here, as it's owned by the Renderer.
