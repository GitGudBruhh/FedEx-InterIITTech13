from gurobipy import Model, GRB
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


# Dimensions of the large container
L_big, W_big, H_big = 10, 10, 10

# Dimensions of small cuboids
cuboids = [
    (4, 6, 3), (3, 8, 4), (5, 4, 5), (6, 3, 2), (4, 4, 3),
    (7, 3, 6), (2, 6, 5), (3, 5, 3), (5, 2, 4), (6, 4, 2),
    (5, 5, 5), (4, 4, 4), (3, 3, 3), (3, 3, 3), (3, 2, 1),
    (4, 4, 5), (3, 2, 3), (4, 3, 5), (5, 5, 5), (3, 2, 1),
    (4, 5, 1)

]
n = len(cuboids)

# cuboids = sorted(cuboids, key=lambda c: c[0] * c[1] * c[2], reverse=True)

# Create the model
model = Model("CuboidPacking")

# Decision variables
x = model.addVars(n, vtype=GRB.INTEGER, name="x")
y = model.addVars(n, vtype=GRB.INTEGER, name="y")
z = model.addVars(n, vtype=GRB.INTEGER, name="z")
f = model.addVars(n, vtype=GRB.BINARY, name="fitted")  # Fitting indicator

# Rotation indicators (one for each axis)
# rot_x = model.addVars(n, vtype=GRB.BINARY, name="rot_x")
# rot_y = model.addVars(n, vtype=GRB.BINARY, name="rot_y")
# rot_z = model.addVars(n, vtype=GRB.BINARY, name="rot_z")

# Dimensions depending on rotation
length = {i: model.addVar(vtype=GRB.INTEGER, name=f"length_{i}") for i in range(n)}
width = {i: model.addVar(vtype=GRB.INTEGER, name=f"width_{i}") for i in range(n)}
height = {i: model.addVar(vtype=GRB.INTEGER, name=f"height_{i}") for i in range(n)}
rot = {i: model.addVars(6, vtype=GRB.BINARY, name=f"rot_{i}") for i in range(n)}

for i in range(n):
    model.addConstr(rot[i].sum() == 1, name=f"rotation_one_{i}")

for i, (l, w, h) in enumerate(cuboids):
    model.addConstr(
        length[i] == l * rot[i][0] + l * rot[i][1] + w * rot[i][2] + w * rot[i][3] + h * rot[i][4] + h * rot[i][5],
        name=f"length_assignment_{i}")
    model.addConstr(
        width[i] == w * rot[i][0] + h * rot[i][1] + l * rot[i][2] + h * rot[i][3] + l * rot[i][4] + w * rot[i][5],
        name=f"width_assignment_{i}")
    model.addConstr(
        height[i] == h * rot[i][0] + w * rot[i][1] + h * rot[i][2] + l * rot[i][3] + w * rot[i][4] + l * rot[i][5],
        name=f"height_assignment_{i}")

# Assign dimensions based on rotation
# for i, (l, w, h) in enumerate(cuboids):
#     # Ensure the cuboid's dimensions are correctly assigned based on the rotation flags
#     model.addConstr(length[i] == h * rot_x[i] + l * (1 - rot_x[i]) + (w - l) * rot_y[i],
#                     name=f"length_rot_{i}")
#     model.addConstr(width[i] == l * rot_y[i] * (1 -rot_x[i]) * (1-rot_z[i]) +
#                                 (h-l) * rot_y[i] * (1 - rot_x[i]) * rot_z[i] +
#                                 w * (1-rot_y[i]) * (1 - rot_x[i]) * (1 - rot_z[i]) +
#                                 (h - w) * (1-rot_y[i]) * (1 - rot_x[i]) * rot_z[i] +
#                                 l * (1 - rot_y[i]) * rot_x[i] * (1 - rot_z[i]) +
#                                 (w - l) * (1 - rot_y[i]) * rot_x[i] * rot_z[i] ,
#                     name=f"width_rot_{i}")
#     model.addConstr(height[i] == h * (1 - rot_x[i] - rot_z[i]) + l * rot_x[i] + w * rot_z[i],
#                     name=f"height_rot_{i}")
#
#     # Ensure no dimension can be zero for a valid rotation
#     # A dimension will be non-zero if at least one of the corresponding rotation flags is 1
#     model.addConstr(length[i] >= 1, name=f"length_non_zero_{i}")
#     model.addConstr(width[i] >= 1, name=f"width_non_zero_{i}")
#     model.addConstr(height[i] >= 1, name=f"height_non_zero_{i}")
#
#     # Ensure only one rotation is chosen (either x, y, or z)
#     model.addConstr(rot_x[i] + rot_y[i] + rot_z[i] <= 2, name=f"rotation_choice_{i}")

# Fit within the large container
for i in range(n):
    model.addConstr(x[i] >= 0)
    model.addConstr(y[i] >= 0)
    model.addConstr(z[i] >= 0)
    model.addConstr(x[i] + length[i] <= L_big)
    model.addConstr(y[i] + width[i] <= W_big)
    model.addConstr(z[i] + height[i] <= H_big)

# Non-overlapping constraints
for i in range(n):
    for j in range(i + 1, n):
        left = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
        right = model.addVar(vtype=GRB.BINARY, name=f"right_{i}_{j}")
        below = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}")
        above = model.addVar(vtype=GRB.BINARY, name=f"above_{i}_{j}")
        front = model.addVar(vtype=GRB.BINARY, name=f"front_{i}_{j}")
        behind = model.addVar(vtype=GRB.BINARY, name=f"behind_{i}_{j}")

        # At least one condition must hold
        model.addConstr(left + right + below + above + front + behind >= f[i] + f[j] - 1)

        # Big-M constraints
        M = max(L_big, W_big, H_big)
        model.addConstr(x[i] + length[i] <= x[j] + M * (1 - left) + M * (1 - f[i]) + M * (1 - f[j]))
        model.addConstr(x[j] + length[j] <= x[i] + M * (1 - right) + M * (1 - f[i]) + M * (1 - f[j]))
        model.addConstr(y[i] + width[i] <= y[j] + M * (1 - front) + M * (1 - f[i]) + M * (1 - f[j]))
        model.addConstr(y[j] + width[j] <= y[i] + M * (1 - behind) + M * (1 - f[i]) + M * (1 - f[j]))
        model.addConstr(z[i] + height[i] <= z[j] + M * (1 - above) + M * (1 - f[i]) + M * (1 - f[j]))
        model.addConstr(z[j] + height[j] <= z[i] + M * (1 - below) + M * (1 - f[i]) + M * (1 - f[j]))

# Set the objective to maximize the total packed volume
model.setParam('MIPGap', 0.2)
model.setObjective(f.sum(), GRB.MAXIMIZE)
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    for i in range(n):
        print(f"Cuboid {i + 1}:")
        print(f"  Placed: {'Yes' if f[i].x > 0.5 else 'No'}")
        if f[i].x <= 0.5: continue
        print(f"  Bottom-left-front corner: ({x[i].x}, {y[i].x}, {z[i].x})")
        print(f"  Length: {length[i].x}, Width: {width[i].x}, Height: {height[i].x}")
        print(f"  Rotation {rot[i]}")
else:
    print(f"Solver ended with status {model.status}.")

# Visualization
if model.status == GRB.OPTIMAL:
    x_vals = [x[i].x for i in range(n)]
    y_vals = [y[i].x for i in range(n)]
    z_vals = [z[i].x for i in range(n)]
    lengths = [length[i].x for i in range(n)]
    widths = [width[i].x for i in range(n)]
    heights = [height[i].x for i in range(n)]
    fitted_flags = [f[i].x for i in range(n)]

    visualize_cuboid_packing(L_big, W_big, H_big, cuboids, x_vals, y_vals, z_vals, lengths, widths, heights,
                             fitted_flags)

    visualize_3d_packing(L_big, W_big, H_big, x_vals, y_vals, z_vals, lengths, widths, heights, fitted_flags)
