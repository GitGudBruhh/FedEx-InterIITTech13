import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import numpy as np

def draw_cuboid(grid, x, y, z, l, w, h):
    """
    Create a cuboid and add it to the plot.
    :param grid: The PyVista grid to add the cuboid to.
    :param x, y, z: Coordinates of the bottom-left-front corner.
    :param l, w, h: Length, width, and height of the cuboid.
    """
    # Define vertices of the cuboid
    points = np.array([
        [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],  # Bottom face
        [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]  # Top face
    ], dtype=np.int32)

    points = points.astype(np.float32)
    # Define the faces of the cuboid (each face consists of four vertices)
    faces = np.array([
        [4, 0, 1, 2, 3],  # Bottom face
        [4, 7, 6, 5, 4],  # Top face
        [4, 0, 3, 7, 4],  # Left face
        [4, 1, 2, 6, 5],  # Right face
        [4, 0, 1, 5, 4],  # Front face
        [4, 3, 2, 6, 7]  # Back face
    ], dtype=np.int32)

    # Create the cuboid mesh
    cuboid = pv.PolyData(points, faces)

    # Color the cuboid with a random color
    color = np.random.rand(3, )  # Random color
    grid.add_mesh(cuboid, color=color, show_edges=True, edge_color='black')  # Add the cuboid mesh


def visualize_3d_packing(L_big, W_big, H_big, x_vals, y_vals, z_vals, lengths, widths, heights, fitted_flags):
    # Create a PyVista plotter object
    plotter = pv.Plotter()

    # Add a 3D container box
    container_points = np.array([
        [0, 0, 0], [L_big, 0, 0], [L_big, W_big, 0], [0, W_big, 0],  # Bottom face
        [0, 0, H_big], [L_big, 0, H_big], [L_big, W_big, H_big], [0, W_big, H_big]  # Top face
    ], dtype=np.int32)

    container_faces = np.array([
        [4, 0, 1, 2, 3],  # Bottom face
        [4, 7, 6, 5, 4],  # Top face
        [4, 0, 3, 7, 4],  # Left face
        [4, 1, 2, 6, 5],  # Right face
        [4, 0, 1, 5, 4],  # Front face
        [4, 3, 2, 6, 7]  # Back face
    ], dtype=np.int32)

    # Create the container as a mesh
    container = pv.PolyData(container_points, container_faces)

    # Add the container mesh with edge color
    plotter.add_mesh(container, color='lightgray', opacity=0.5, show_edges=True, edge_color='black')

    # Add the cuboids to the plot
    for i in range(len(fitted_flags)):
        if fitted_flags[i] <= 0.5:  # If the cuboid is not fitted, skip it
            continue
        x, y, z = x_vals[i], y_vals[i], z_vals[i]
        l, w, h = lengths[i], widths[i], heights[i]
        draw_cuboid(plotter, x, y, z, l, w, h)

    # Set the camera position for a good view
    plotter.view_isometric()

    # Show the plot interactively
    plotter.show()


def visualize_cuboid_packing(L_big, W_big, H_big, cuboids, x_vals, y_vals, z_vals, lengths, widths, heights,
                             fitted_flags):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the container
    ax.bar3d(0, 0, 0, L_big, W_big, H_big, alpha=0.2, color='grey', edgecolor='black')

    # Draw each cuboid
    for i in range(len(cuboids)):
        if fitted_flags[i] <= 0.5:
            continue

        x, y, z = x_vals[i], y_vals[i], z_vals[i]
        l, w, h = lengths[i], widths[i], heights[i]

        # Define the vertices of the cuboid
        vertices = [
            (x, y, z),
            (x + l, y, z),
            (x + l, y + w, z),
            (x, y + w, z),
            (x, y, z + h),
            (x + l, y, z + h),
            (x + l, y + w, z + h),
            (x, y + w, z + h),
        ]

        # Define the faces of the cuboid
        faces = [
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
        ]

        # Random color for each cuboid
        color = plt.cm.Paired(i / len(cuboids))

        # Draw the cuboid
        poly3d = Poly3DCollection(faces, alpha=0.8, facecolors=color, edgecolors='black')
        ax.add_collection3d(poly3d)

        # Add a label
        ax.text(x + l / 2, y + w / 2, z + h / 2, str(i + 1), color='black', ha='center', va='center')

    # Set limits and labels
    ax.set_xlim(0, L_big)
    ax.set_ylim(0, W_big)
    ax.set_zlim(0, H_big)
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    plt.title("Cuboid Packing Solution")
    plt.show()
