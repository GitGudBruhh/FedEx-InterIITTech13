from gurobipy import Model, GRB
import visualize
import data
from visualize import visualize_cuboid_packing

# Read data
ULD, packages = data.read_data_from_csv("uld.csv", "package.csv")

n = len(packages)
m = len(ULD)

# Create the model
model = Model("MultiULD_CuboidPacking")

# Decision variables
x = model.addVars(n, m, vtype=GRB.INTEGER, name="x")  # x-coordinate of each package in each ULD
y = model.addVars(n, m, vtype=GRB.INTEGER, name="y")  # y-coordinate
z = model.addVars(n, m, vtype=GRB.INTEGER, name="z")  # z-coordinate
f = model.addVars(n, m, vtype=GRB.BINARY, name="fitted")  # Whether package i is in ULD k
u = model.addVars(m, vtype=GRB.BINARY, name="used")  # Whether ULD k is used

# Dimensions depending on rotation
length = {(i, k): model.addVar(vtype=GRB.INTEGER, name=f"length_{i}_{k}") for i in range(n) for k in range(m)}
width = {(i, k): model.addVar(vtype=GRB.INTEGER, name=f"width_{i}_{k}") for i in range(n) for k in range(m)}
height = {(i, k): model.addVar(vtype=GRB.INTEGER, name=f"height_{i}_{k}") for i in range(n) for k in range(m)}
rot = {(i, k): model.addVars(6, vtype=GRB.BINARY, name=f"rot_{i}_{k}") for i in range(n) for k in range(m)}

# Constraints for rotation
for i, package in enumerate(packages):
    l = package.dimensions[0]
    w = package.dimensions[1]
    h = package.dimensions[2]
    for k in range(m):
        model.addConstr(rot[i, k].sum() == f[i, k], name=f"rotation_one_{i}_{k}")
        model.addConstr(
            length[i, k] == l * rot[i, k][0] + l * rot[i, k][1] +
                          w * rot[i, k][2] + w * rot[i, k][3] +
                          h * rot[i, k][4] + h * rot[i, k][5],
            name=f"length_assignment_{i}_{k}")
        model.addConstr(
            width[i, k] == w * rot[i, k][0] + h * rot[i, k][1] +
                          l * rot[i, k][2] + h * rot[i, k][3] +
                          l * rot[i, k][4] + w * rot[i, k][5],
            name=f"width_assignment_{i}_{k}")
        model.addConstr(
            height[i, k] == h * rot[i, k][0] + w * rot[i, k][1] +
                           h * rot[i, k][2] + l * rot[i, k][3] +
                           w * rot[i, k][4] + l * rot[i, k][5],
            name=f"height_assignment_{i}_{k}")

# Fit within each ULD
for k, uld in enumerate(ULD):
    for i in range(n):
        model.addConstr(x[i, k] >= 0)
        model.addConstr(y[i, k] >= 0)
        model.addConstr(z[i, k] >= 0)
        model.addConstr(x[i, k] + length[i, k] <= uld.dimensions[0] * f[i, k])
        model.addConstr(y[i, k] + width[i, k] <= uld.dimensions[1] * f[i, k])
        model.addConstr(z[i, k] + height[i, k] <= uld.dimensions[2] * f[i, k])

# Each package can be assigned to at most one ULD
for i in range(n):
 model.addConstr(f.sum(i, "*") <= 1, name=f"single_ULD_{i}")

# Weight constraints per ULD
for k, uld in enumerate(ULD):
    model.addConstr(
        sum(f[i, k] * packages[i].weight for i in range(n)) <= uld.weight_limit * u[k],
        name=f"weight_limit_{k}")

# Non-overlapping constraints within each ULD
for k, uld in enumerate(ULD):
    for i in range(n):
        for j in range(i + 1, n):
            left = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}_{k}")
            right = model.addVar(vtype=GRB.BINARY, name=f"right_{i}_{j}_{k}")
            below = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}_{k}")
            above = model.addVar(vtype=GRB.BINARY, name=f"above_{i}_{j}_{k}")
            front = model.addVar(vtype=GRB.BINARY, name=f"front_{i}_{j}_{k}")
            behind = model.addVar(vtype=GRB.BINARY, name=f"behind_{i}_{j}_{k}")

            # At least one condition must hold
            model.addConstr(left + right + below + above + front + behind >= f[i, k] + f[j, k] - 1)

            # Big-M constraints
            M = max(uld.dimensions[0], uld.dimensions[1], uld.dimensions[2])
            model.addConstr(x[i, k] + length[i, k] <= x[j, k] + M * (1 - left))
            model.addConstr(x[j, k] + length[j, k] <= x[i, k] + M * (1 - right))
            model.addConstr(y[i, k] + width[i, k] <= y[j, k] + M * (1 - front))
            model.addConstr(y[j, k] + width[j, k] <= y[i, k] + M * (1 - behind))
            model.addConstr(z[i, k] + height[i, k] <= z[j, k] + M * (1 - above))
            model.addConstr(z[j, k] + height[j, k] <= z[i, k] + M * (1 - below))


# Set the objective to maximize the total packed volume
# model.setParam("MIPGap",0.02)
model.setObjective(f.sum(), GRB.MAXIMIZE)

# Optimize
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    for k, uld in enumerate(ULD):
        print(f"ULD {uld.id}:")
        for i in range(n):
            if f[i, k].x > 0.5:
                print(f"  Package {i + 1} placed in ULD {uld.id} at ({x[i, k].x}, {y[i, k].x}, {z[i, k].x})")
else:
    print(f"Solver ended with status {model.status}.")

# Visualization for each ULD
if model.status == GRB.OPTIMAL:
    for k, uld in enumerate(ULD):
        print(f"Visualization for ULD {uld.id}:")

        # Collect data for the ULD
        x_vals = [x[i, k].x for i in range(n) if f[i, k].x > 0.5]
        y_vals = [y[i, k].x for i in range(n) if f[i, k].x > 0.5]
        z_vals = [z[i, k].x for i in range(n) if f[i, k].x > 0.5]
        lengths = [length[i, k].x for i in range(n) if f[i, k].x > 0.5]
        widths = [width[i, k].x for i in range(n) if f[i, k].x > 0.5]
        heights = [height[i, k].x for i in range(n) if f[i, k].x > 0.5]
        fitted_flags = [f[i, k].x for i in range(n) if f[i, k].x > 0.5]

        # Check if any packages are fitted in this ULD
        if not x_vals:
            print(f"  No packages packed in ULD {uld.id}.")
            continue

        # Use the visualize function for this ULD
        visualize.visualize_3d_packing(
            uld.dimensions[0], uld.dimensions[1], uld.dimensions[2],
            x_vals, y_vals, z_vals, lengths, widths, heights, fitted_flags
        )
else:
    print(f"Solver ended with status {model.status}.")
