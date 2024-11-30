import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gurobipy import Model, GRB

# Visualization function
def visualize_solution(W_big, H_big, rectangles, x_vals, y_vals, widths, heights, rotated_flags,f):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the container
    ax.add_patch(patches.Rectangle((0, 0), W_big, H_big, edgecolor='black', facecolor='lightgrey', lw=2))

    # Draw each rectangle
    for i in range(len(rectangles)):
        x = x_vals[i]
        y = y_vals[i]
        w = widths[i]
        h = heights[i]
        rotated = rotated_flags[i]

        if f[i] <= 0.5: continue
        # Random color for each rectangle
        color = plt.cm.Paired(i / len(rectangles))

        # Draw the rectangle
        ax.add_patch(
            patches.Rectangle((x, y), w, h, edgecolor='black', facecolor=color, lw=1)
        )

        # Label the rectangle
        ax.text(
            x + w / 2,
            y + h / 2,
            f"{i + 1}\n{int(w)}x{int(h)}",
            ha='center',
            va='center',
            color='black',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

    # Set limits and aspect
    ax.set_xlim(0, W_big)
    ax.set_ylim(0, H_big)
    ax.set_aspect('equal', adjustable='box')

    plt.title("Rectangle Packing Solution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True)
    plt.show()

# Dimensions of the large rectangle
W_big = 20
H_big = 15

# Dimensions of small rectangles
rectangles = [
    (4, 6), (3, 8), (5, 4), (6, 3), (4, 4),
    (7, 3), (2, 6), (3, 5), (5, 2), (6, 4),
    (5,5),(4, 6), (3, 8), (5, 4), (6, 3), (4, 4),
    (7, 3), (2, 6), (3, 5), (5, 2), (6, 4),
    (5,5)
]
n = len(rectangles)

# Create the model
model = Model("RectanglePacking")

# Decision variables
x = model.addVars(n, vtype=GRB.CONTINUOUS, name="x")  # x-coordinates
y = model.addVars(n, vtype=GRB.CONTINUOUS, name="y")  # y-coordinates
rotated = model.addVars(n, vtype=GRB.BINARY, name="rotated")  # Rotation indicator (0 - no rotation/ 1 - 90 deg rotation)
f = model.addVars(n, vtype=GRB.BINARY, name="fitted") # whether it is fitted or not (1 - fitted/ 0 - Not)


# Dimensions depending on rotation
width = {i: model.addVar(vtype=GRB.CONTINUOUS, name=f"width_{i}") for i in range(n)}
height = {i: model.addVar(vtype=GRB.CONTINUOUS, name=f"height_{i}") for i in range(n)}

# Assign dimensions based on rotation
for i, (w, h) in enumerate(rectangles):
    model.addConstr(width[i] == rotated[i] * h + (1 - rotated[i]) * w)
    model.addConstr(height[i] == rotated[i] * w + (1 - rotated[i]) * h)

# Fit within the large rectangle
for i in range(n):
    model.addConstr(x[i] >= 0)
    model.addConstr(y[i] >= 0)
    model.addConstr(x[i] + width[i] <= W_big)
    model.addConstr(y[i] + height[i] <= H_big)

# Non-overlapping constraints
for i in range(n):
    for j in range(i + 1, n):
        # Binary variables for relative positioning
        left = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
        right = model.addVar(vtype=GRB.BINARY, name=f"right_{i}_{j}")
        below = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}")
        above = model.addVar(vtype=GRB.BINARY, name=f"above_{i}_{j}")

        # At least one condition must hold
        model.addConstr(left + right + below + above >= f[i] + f[j] - 1, name=f"disjoint_{i}_{j}")

        # Big-M constraints to activate conditions only if both rectangles are placed
        M = max(W_big, H_big)  # Big-M constant (large enough to deactivate constraints)

        model.addConstr(x[i] + width[i] <= x[j] + M * (1 - left) + M * (1 - f[i]) + M * (1 - f[j]), name=f"left_{i}_{j}")
        model.addConstr(x[j] + width[j] <= x[i] + M * (1 - right) + M * (1 - f[i]) + M * (1 - f[j]), name=f"right_{i}_{j}")
        model.addConstr(y[i] + height[i] <= y[j] + M * (1 - below) + M * (1 - f[i]) + M * (1 - f[j]), name=f"below_{i}_{j}")
        model.addConstr(y[j] + height[j] <= y[i] + M * (1 - above) + M * (1 - f[i]) + M * (1 - f[j]), name=f"above_{i}_{j}")

area = model.addVars(n, vtype=GRB.CONTINUOUS, name="area")

# Big-M constant
M = W_big * H_big  # Maximum possible area of a single rectangle

# Constraints to define the area
for i in range(n):
    model.addConstr(area[i] <= width[i] * height[i], name=f"area_bound1_{i}")
    model.addConstr(area[i] <= M * f[i], name=f"area_bound2_{i}")

rem_area = model.addVars(n, vtype=GRB.CONTINUOUS, name="rem_area")

rem_ar = M - area.sum()

# Constraints to define the area
for i in range(n):
    model.addConstr(rem_area[i] <= width[i] * height[i], name=f"rem_area_bound1_{i}")
    model.addConstr(rem_area[i] <= M * (1-f[i]), name=f"rem_area_bound2_{i}")
    model.addConstr(rem_area[i] <= rem_ar, name=f"rem_area_bound3_{i}")

model.addConstr(rem_area.sum() <= rem_ar,name=f"stopping_condition")
# Set the objective to maximize the total packed area
model.setParam('MIPGap',0.1)
model.setObjective(rem_area.sum(), GRB.MINIMIZE)
model.setObjective(f.sum() + area.sum(), GRB.MAXIMIZE)
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    for i in range(n):
        print(f"Rectangle {i + 1}:")
        print(f"  Placed: {'Yes' if f[i].x > 0.5 else 'No'}")
        if f[i].x <= 0.5:continue
        print(f"  Bottom-left corner: ({x[i].x}, {y[i].x})")
        print(f"  Width: {width[i].x}, Height: {height[i].x}")
        print(f"  Rotated: {'Yes' if rotated[i].x > 0.5 else 'No'}")
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
else:
    print(f"Solver ended with status {model.status}.")


# Collect data for visualization
if model.status == GRB.OPTIMAL:
    fl = [f[i].x for i in range(n)]
    x_vals = [x[i].x for i in range(n)]
    y_vals = [y[i].x for i in range(n)]
    widths = [width[i].x for i in range(n)]
    heights = [height[i].x for i in range(n)]
    rotated_flags = [rotated[i].x for i in range(n)]

    # Call the visualization function
    visualize_solution(W_big, H_big, rectangles, x_vals, y_vals, widths, heights, rotated_flags,fl)
