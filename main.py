import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors


# Define the ULD class
class ULD:
    def __init__(self, id: str, length: int, width: int, height: int, weight_limit: int):
        self.id = id
        self.dimensions = (length, width, height)
        self.weight_limit = weight_limit
        self.current_weight = 0
        self.occupied_positions = []  # Tracks occupied spaces within the ULD
        self.available_areas = []  # Tracks available areas for packing


# Define the Package class
class Package:
    def __init__(self, id: str, length: int, width: int, height: int, weight: int, is_priority: bool, delay_cost: int):
        self.id = id
        self.dimensions = (length, width, height)
        self.weight = weight
        self.is_priority = is_priority
        self.delay_cost = delay_cost


# Define the ULDPacker class
class ULDPacker:
    def __init__(self, ulds: List[ULD], packages: List[Package], priority_spread_cost: int):
        self.ulds = ulds
        self.packages = packages
        self.priority_spread_cost = priority_spread_cost
        self.packed_positions = []  # [(package_id, uld_id, x, y, z)]
        self.unpacked_packages = []

    def _find_available_space(self, uld: ULD, package: Package) -> Tuple[bool, np.ndarray]:
        length, width, height = package.dimensions
        for area in uld.available_areas:
            x, y, z, al, aw, ah = area
            if length <= al and width <= aw and height <= ah:
                return True, np.array([x, y, z])
        return False, None

    def _try_pack_package(self, package: Package, uld: ULD) -> bool:
        if package.weight + uld.current_weight > uld.weight_limit:
            return False  # Exceeds weight limit

        can_fit, position = self._find_available_space(uld, package)
        if can_fit:
            x, y, z = position
            length, width, height = package.dimensions
            uld.occupied_positions.append(np.array([x, y, z, length, width, height]))
            uld.current_weight += package.weight
            self.packed_positions.append((package.id, uld.id, x, y, z))
            self._update_available_areas(uld, position, package)
            return True
        return False

    def _update_available_areas(self, uld: ULD, position: np.ndarray, package: Package):
        length, width, height = package.dimensions
        x, y, z = position

        updated_areas = []
        for area in uld.available_areas:
            ax, ay, az, al, aw, ah = area

            # Break area into smaller sections if necessary
            if not (x + length <= ax or x >= ax + al or
                    y + width <= ay or y >= ay + aw or
                    z + height <= az or z >= az + ah):
                # Top section
                if y + width < ay + aw:
                    updated_areas.append([ax, y + width, az, al, aw - (y + width - ay), ah])
                # Bottom section
                if y > ay:
                    updated_areas.append([ax, ay, az, al, y - ay, ah])
                # Left section
                if x > ax:
                    updated_areas.append([ax, ay, az, x - ax, aw, ah])
                # Right section
                if x + length < ax + al:
                    updated_areas.append([x + length, ay, az, al - (x + length - ax), aw, ah])
                # Front section
                if z > az:
                    updated_areas.append([ax, ay, az, al, aw, z - az])
                # Back section
                if z + height < az + ah:
                    updated_areas.append([ax, ay, z + height, al, aw, ah - (z + height - az)])
            else:
                updated_areas.append(area)

        uld.available_areas = updated_areas

    def _generate_3d_plot(self):
        for uld in self.ulds:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for package_id, uld_id, x, y, z in self.packed_positions:
                if uld.id == uld_id:
                    package = next(pkg for pkg in self.packages if pkg.id == package_id)
                    length, width, height = package.dimensions

                    vertices = [
                        [x, y, z],
                        [x + length, y, z],
                        [x + length, y + width, z],
                        [x, y + width, z],
                        [x, y, z + height],
                        [x + length, y, z + height],
                        [x + length, y + width, z + height],
                        [x, y + width, z + height],
                    ]

                    faces = [
                        [vertices[0], vertices[1], vertices[5], vertices[4]],
                        [vertices[1], vertices[2], vertices[6], vertices[5]],
                        [vertices[2], vertices[3], vertices[7], vertices[6]],
                        [vertices[3], vertices[0], vertices[4], vertices[7]],
                        [vertices[0], vertices[1], vertices[2], vertices[3]],
                        [vertices[4], vertices[5], vertices[6], vertices[7]],
                    ]

                    color = mcolors.to_rgba('blue', alpha=0.5)
                    poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.5)
                    ax.add_collection3d(poly3d)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Packed ULD {uld.id}')
            ax.set_xlim(0, uld.dimensions[0])
            ax.set_ylim(0, uld.dimensions[1])
            ax.set_zlim(0, uld.dimensions[2])

            plt.savefig(f'packed_uld_{uld.id}.png')
            plt.show()

    def count_priority_packages_in_uld(self):
        priority_count_per_uld = {}
        for package_id, uld_id, _, _, _ in self.packed_positions:
            package = next(pkg for pkg in self.packages if pkg.id == package_id)
            if package.is_priority:
                if uld_id not in priority_count_per_uld:
                    priority_count_per_uld[uld_id] = 0
                priority_count_per_uld[uld_id] += 1
        return priority_count_per_uld

    def pack(self):
        for uld in self.ulds:
            uld.available_areas = [[0, 0, 0, *uld.dimensions]]

        priority_packages = sorted(
            [pkg for pkg in self.packages if pkg.is_priority], key=lambda p: p.delay_cost, reverse=True
        )
        economy_packages = sorted(
            [pkg for pkg in self.packages if not pkg.is_priority], key=lambda p: p.delay_cost, reverse=True
        )

        for package in priority_packages:
            packed = False
            for uld in self.ulds:
                if self._try_pack_package(package, uld):
                    packed = True
                    break
            if not packed:
                self.unpacked_packages.append(package)

        for package in economy_packages:
            packed = False
            for uld in self.ulds:
                if self._try_pack_package(package, uld):
                    packed = True
                    break
            if not packed:
                self.unpacked_packages.append(package)

        remaining_packages = self.unpacked_packages.copy()
        for uld in self.ulds:
            for package in remaining_packages:
                if self._try_pack_package(package, uld):
                    remaining_packages.remove(package)

        total_delay_cost = sum(pkg.delay_cost for pkg in self.unpacked_packages)

        ulds_with_priority = set(
            uld_id for package_id, uld_id, _, _, _ in self.packed_positions
            if any(pkg.id == package_id and pkg.is_priority for pkg in self.packages)
        )

        priority_spread_cost = self.priority_spread_cost * len(ulds_with_priority)

        total_cost = total_delay_cost + priority_spread_cost

        return self.packed_positions, self.unpacked_packages, total_cost, list(ulds_with_priority)


def read_data_from_csv(uld_file: str, package_file: str) -> Tuple[List[ULD], List[Package]]:
    uld_data = pd.read_csv(uld_file)
    package_data = pd.read_csv(package_file)

  

    # Parse ULD data
    ulds = [
        ULD(
            id=row['ULD Identifier'],
            length=row['Length (cm)'],
            width=row['Width (cm)'],
            height=row['Height (cm)'],
            weight_limit=row['Weight Limit (kg)'],
        )
        for _, row in uld_data.iterrows()
    ]

    # Parse Package data
    packages = [
        Package(
            id=row['Package Identifier'],
            length=row['Length (cm)'],
            width=row['Width (cm)'],
            height=row['Height (cm)'],
            weight=row['Weight (kg)'],
            is_priority=row['Type (P/E)'] == 'Priority',
            delay_cost=int(row['Cost of Delay']) if str(row['Cost of Delay']).isdigit() else 0,
        )
        for _, row in package_data.iterrows()
    ]

    return ulds, packages

def format_output(packed_positions, unpacked_packages, total_cost, ulds_with_priority):
    output = []
    
    # Packing results for packed positions
    for package_id, uld_id, x, y, z in packed_positions:
        output.append(f"{package_id} is packed in ULD {uld_id} at position ({x}, {y}, {z})")
    
    # Unpacked packages
    for package in unpacked_packages:
        output.append(f"{package.id} could not be packed")
    
    # Add summary information
    output.append(f"Total Cost: {total_cost}")
    output.append(f"ULDs with priority packages: {', '.join(ulds_with_priority)}")
    
    # Summary statistics
    output.append("Packing Results:")
    output.append(f"Total packages: {len(packed_positions) + len(unpacked_packages)}")
    output.append(f"Packed packages: {len(packed_positions)}")
    output.append(f"Unpacked packages: {len(unpacked_packages)}")
    output.append(f"Total cost (delay + priority spread): {total_cost}")
    output.append(f"ULDs with priority packages: {', '.join(ulds_with_priority)}")
    
    # Priority packages per ULD

    
    return "\n".join(output)

# Assuming the read_data_from_csv function and ULDPacker class are defined properly elsewhere
if __name__ == "__main__":
    uld_file = "ulds.csv"
    package_file = "packages.csv"
    ulds, packages = read_data_from_csv(uld_file, package_file)
    priority_spread_cost = 40  # Example cost for spreading priority packages

    packer = ULDPacker(ulds, packages, priority_spread_cost)
    packed_positions, unpacked_packages, total_cost, ulds_with_priority= packer.pack()

    # Format the output
    output = format_output(
        packed_positions, unpacked_packages, total_cost, ulds_with_priority,
    )
    print(output)

    # Generate 3D plot if the method is available
    packer._generate_3d_plot()

