import moderngl
import numpy as np
from math import radians, sin, cos, tan
from PIL import Image
from core.objects import obj
from core.datatypes import float3, float4 # Import float4

# --- Matrix Creation Functions --- (These remain the same)
def create_translation_matrix(x, y, z):
    """Creates a 4x4 translation matrix."""
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype='f4')

def create_scale_matrix(sx, sy, sz):
    """Creates a 4x4 scale matrix."""
    return np.array([
        [sx,   0.0, 0.0, 0.0],
        [0.0, sy,   0.0, 0.0],
        [0.0, 0.0, sz,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype='f4')

def create_rotation_x_matrix(angle_radians):
    """Creates a 4x4 rotation matrix around the X-axis."""
    c = cos(angle_radians)
    s = sin(angle_radians)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,  -s, 0.0],
        [0.0,   s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype='f4')

def create_rotation_y_matrix(angle_radians):
    """Creates a 4x4 rotation matrix around the Y-axis."""
    c = cos(angle_radians)
    s = sin(angle_radians)
    return np.array([
        [  c, 0.0,   s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [ -s, 0.0,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype='f4')

def create_rotation_z_matrix(angle_radians):
    """Creates a 4x4 rotation matrix around the Z-axis."""
    c = cos(angle_radians)
    s = sin(angle_radians)
    return np.array([
        [  c,  -s, 0.0, 0.0],
        [  s,   c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype='f4')

def create_perspective_matrix(fov_degrees, aspect_ratio, near, far):
    """Creates a 4x4 perspective projection matrix."""
    f = 1.0 / tan(radians(fov_degrees / 2.0))
    return np.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype='f4')
# --- End Matrix Creation Functions ---


class Renderer:
    """
    Handles all rendering operations for the engine.
    Manages shader programs, light properties, and drawing objects.
    """
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # --- SHADER PROGRAM (STANDARD VERSION) ---
        self.main_program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_vert;    // Vertex position
                in vec3 in_normal;  // Vertex normal for lighting calculations
                in vec2 in_uv;      // Vertex UV coordinates for texturing
                
                uniform mat4 mvp;   // Model-View-Projection matrix
                uniform mat4 model; // Model matrix (for transforming normals)
                
                out vec3 v_normal;   // Normal to fragment shader (in world space)
                out vec3 v_position; // Vertex position to fragment shader (in world space)
                out vec2 v_uv;       // UV to fragment shader

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    v_normal = normalize((model * vec4(in_normal, 0.0)).xyz);
                    v_position = (model * vec4(in_vert, 1.0)).xyz;
                    v_uv = in_uv; // Pass UVs directly to fragment shader
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;   // Interpolated normal from vertex shader
                in vec3 v_position; // Interpolated world position from vertex shader
                in vec2 v_uv;       // Interpolated UV from vertex shader

                uniform vec3 u_light_position;  // Position of the light source
                uniform vec3 u_light_color;     // Color of the light source
                uniform vec4 u_object_color;    // Base color of the object (now RGBA)
                uniform sampler2D u_texture_sampler; // Texture sampler uniform

                out vec4 fragColor;

                void main() {
                    // Sample the texture using the interpolated UV coordinates
                    vec4 texture_sample = texture(u_texture_sampler, v_uv);
                    
                    // Object color is tinted by the texture's sampled color.
                    // The texture's alpha is combined with the object's alpha.
                    vec4 base_material_color_rgba = u_object_color * texture_sample;
                    vec3 base_material_color_rgb = base_material_color_rgba.rgb;


                    // Ambient light component
                    vec3 ambient = 0.9 * base_material_color_rgb;

                    // Diffuse light component
                    vec3 light_dir = normalize(u_light_position - v_position);
                    float diff = max(dot(v_normal, light_dir), 0.0);
                    vec3 diffuse = diff * u_light_color * base_material_color_rgb;

                    // Combine ambient and diffuse
                    vec3 final_rgb = ambient + diffuse;
                    // The final alpha is derived from the combined object and texture alpha.
                    fragColor = vec4(final_rgb, base_material_color_rgba.a);
                }
            '''
        )
        # --- END SHADER PROGRAM ---

        self.light_position = float3(5.0, 5.0, 5.0)
        self.light_color = float3(1.0, 1.0, 1.0)   # White light

        # Create a 1x1 white texture to use as a default fallback
        # Ensure default texture is RGBA (4 channels)
        white_pixel = np.array([255, 255, 255, 255], dtype='u1')
        self.default_white_texture = self.ctx.texture(
            (1, 1),
            4, # 4 components (RGBA)
            white_pixel.tobytes()
        )
        self.default_white_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.default_white_texture.swizzle = 'RGBA'

    def load_texture(self, image_path: str) -> moderngl.Texture:
        """
        Loads an image from the given path and creates a ModernGL texture.
        :param image_path: Path to the image file.
        :return: A ModernGL Texture object.
        """
        try:
            # Ensure image is converted to RGBA
            image = Image.open(image_path).convert("RGBA")
            # Create texture with 4 components (RGBA)
            texture = self.ctx.texture(image.size, 4, image.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.swizzle = 'RGBA'
            return texture
        except FileNotFoundError:
            print(f"Error: Texture file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error loading texture {image_path}: {e}")
            return None

    def render_scene(self, camera, objects_to_render: list[obj], aspect_ratio: float):
        """
        Renders all objects in the scene from the perspective of the given camera.
        :param camera: The active Camera object (main_camera or scene_camera).
        :param objects_to_render: A list of obj instances to draw.
        :param aspect_ratio: The aspect ratio of the window.
        """
        # Clear color now includes alpha (e.g., 0.1, 0.1, 0.1, 1.0 for opaque dark gray)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0, depth=1.0)
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        # Standard alpha blending equation: out = src_alpha * src_color + (1 - src_alpha) * dest_color
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


        projection_matrix = create_perspective_matrix(camera.fov, aspect_ratio, camera.near, camera.far)
        view_matrix = camera.create_view_matrix()
        
        self.main_program['u_light_position'].write(self.light_position.to_bytes())
        self.main_program['u_light_color'].write(self.light_color.to_bytes())
        self.main_program['u_texture_sampler'].value = 0

        for game_obj in objects_to_render:
            if game_obj.model:
                game_obj.model.program = self.main_program
                
                # Get the object's full world-space model matrix from the obj instance.
                # This matrix correctly incorporates parent transformations.
                model_matrix = game_obj.get_model_matrix()
                
                mvp = projection_matrix @ view_matrix @ model_matrix
                
                self.main_program['mvp'].write(mvp.T.tobytes())
                self.main_program['model'].write(model_matrix.T.tobytes())
                
                # Assume game_obj.color is now a float4 or can be converted to one.
                # If game_obj.color is still float3, you'll need to adapt how it's handled in `obj`
                # or ensure it's converted to float4 with alpha=1.0 before being passed here.
                # For now, assuming obj.color is already a float4 or similar with a .to_bytes() method for 4 components.
                self.main_program['u_object_color'].write(game_obj.color.to_bytes())

                if game_obj.texture:
                    game_obj.texture.use(0)
                else:
                    self.default_white_texture.use(0)

                game_obj.model.render()
        
        # Disable blending after rendering transparent objects (or if you only have opaque objects)
        # This is good practice to prevent blending from affecting subsequent draws if not intended.
        self.ctx.disable(moderngl.BLEND)


    def release(self):
        """Frees OpenGL resources associated with the renderer."""
        self.main_program.release()
        self.default_white_texture.release()