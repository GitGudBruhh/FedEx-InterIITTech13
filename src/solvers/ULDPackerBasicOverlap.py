from typing import List, Tuple
from dataclass.ULD import ULD
from dataclass.Package import Package
import numpy as np

from .ULDPackerBase import ULDPackerBase

SIZE_BOUND = 5000


# Define the ULDPacker class
class ULDPackerBasicOverlap(ULDPackerBase):
    """
    A class for packing packages into ULDs using a basic strategy of checking
    overlapping spaces.
    """
    def __init__(
        self,
        ulds: List[ULD],
        packages: List[Package],
        priority_spread_cost: int,
        max_passes: int = 1,
    ):
        """
        Initializes the ULDPackerMixed instance.

        :param ulds: List of ULDs available for packing.
        :param packages: List of packages to be packed.
        :param priority_spread_cost: Cost associated with spreading priority packages.
        :param max_passes: Maximum number of packing passes (default is 1).
        """
        super().__init__(
            ulds,
            packages,
            priority_spread_cost,
            max_passes,
        )

    def _find_available_space(
        self, uld: ULD, package: Package, orientation: Tuple[int], policy: str
    ) -> Tuple[bool, np.ndarray]:
        """
        Finds available space in the specified ULD for the given package and given orientation.

        :param uld: The ULD in which to find space.
        :param package: The package to be packed.
        :param orientation: The orientation of the package.
        :param policy: The policy for finding available space (first_find, min_volume, ...)
        :return: A tuple indicating whether space was found and the coordinates of the space.
        """
        length, width, height = orientation
        best_position = None
        best_idx = None
        if policy != "max_surface_area" and policy != "max_volume":
            best_x = SIZE_BOUND
            best_y = SIZE_BOUND
            best_z = SIZE_BOUND

            best_surface_area = SIZE_BOUND**2
            best_volume = SIZE_BOUND**3
        else:
            best_x = 0
            best_y = 0
            best_z = 0

            best_surface_area = 0
            best_volume = 0

        for idx, area in enumerate(self.available_spaces[uld.id]):
            x, y, z, al, aw, ah = area
            if length <= al and width <= aw and height <= ah:
                if policy == "first_find":
                    best_position = np.array([x, y, z])
                    best_idx = idx
                    break

                elif policy == "origin_bias":
                    # Check for the best position based on best_x, best_y, and best_z
                    if (
                        (x < best_x)
                        or (x == best_x and y < best_y)
                        or (x == best_x and y == best_y and z < best_z)
                    ):
                        best_x = x
                        best_y = y
                        best_z = z
                        best_position = np.array([x, y, z])
                        best_idx = idx

                elif policy == "min_length_sum":
                    # Check for the best position based on sum of best_x, best_y, and best_z
                    if best_x + best_y + best_z > x + y + z:
                        best_x = x
                        best_y = y
                        best_z = z
                        best_position = np.array([x, y, z])
                        best_idx = idx

                elif policy == "min_surface_area":
                    # Check for the best position based on surface area
                    if best_surface_area > al * aw + aw + ah + ah * al:
                        best_surface_area = al * aw + aw + ah + ah * al
                        best_position = np.array([x, y, z])
                        best_idx = idx

                elif policy == "max_surface_area":
                    # Check for the best position based on surface area
                    if best_surface_area < al * aw + aw + ah + ah * al:
                        best_surface_area = al * aw + aw + ah + ah * al
                        best_position = np.array([x, y, z])
                        best_idx = idx

                elif policy == "min_volume":
                    # Check for the best position based on surface area
                    if best_volume > al * aw * ah:
                        best_volume = al * aw * ah
                        best_position = np.array([x, y, z])
                        best_idx = idx

                elif policy == "max_volume":
                    # Check for the best position based on surface area
                    if best_volume < al * aw * ah:
                        best_volume = al * aw * ah
                        best_position = np.array([x, y, z])
                        best_idx = idx

        if best_position is not None:
            return True, best_position, best_idx
        return False, None, -1

    def _update_available_spaces(
        self,
        uld: ULD,
        position: np.ndarray,
        orientation: Tuple[int],
        package: Package,
        space_index: int,
    ):
        length, width, height = orientation
        x, y, z = position

        # New spaces after the space is cut
        updated_spaces = []
        for space in self.available_spaces[uld.id]:
            ax, ay, az, al, aw, ah = space

            # Check for remaining free areas after packing
            if not (
                x + length <= ax
                or x >= ax + al
                or y + width <= ay
                or y >= ay + aw
                or z + height <= az
                or z >= az + ah
            ):
                space1 = [ax, y + width, az, al, aw - (y + width - ay), ah]  # Left full
                space2 = [ax, ay, az, al, y - ay, ah]  # Right full
                space3 = [ax, ay, az, x - ax, aw, ah]  # Back full
                # Front full
                space4 = [x + length, ay, az, al - (x + length - ax), aw, ah]
                space5 = [ax, ay, az, al, aw, z - az]  # Above full
                # Down full
                space6 = [ax, ay, z + height, al, aw, ah - (z + height - az)]

                if y + width < ay + aw and all(v >= self.minimum_dimension for v in space1[3::]):
                    updated_spaces.append(space1)
                    # print(f"Appending {space1}")

                if y > ay and all(v >= self.minimum_dimension for v in space2[3::]):
                    updated_spaces.append(space2)
                    # print(f"Appending {space2}")

                if x > ax and all(v >= self.minimum_dimension for v in space3[3::]):
                    updated_spaces.append(space3)
                    # print(f"Appending {space3}")

                if x + length < ax + al and all(v >= self.minimum_dimension for v in space4[3::]):
                    updated_spaces.append(space4)
                    # print(f"Appending {space4}")

                if z > az and all(v >= self.minimum_dimension for v in space5[3::]):
                    updated_spaces.append(space5)
                    # print(f"Appending {space5}")

                if z + height < az + ah and all(v >= self.minimum_dimension for v in space6[3::]):
                    updated_spaces.append(space6)
                    # print(f"Appending {space6}")

            else:
                updated_spaces.append(space)

        self.available_spaces[uld.id] = updated_spaces

    def pack(self):
        n_packs = 0

        self.minimum_dimension = min([np.min(p.dimensions) for p in self.packages])

        for package in self.packages:
            packed = False
            for uld in self.ulds:
                can_fit = self._try_pack_package(
                    package,
                    uld,
                    space_find_policy="first_find",
                    orientation_choose_policy="no_rot",
                )
                if can_fit:
                    packed = True
                    n_packs += 1
                    print(f"Packed Priority {package.id} in {uld.id}, {n_packs}")
                    break
            if not packed:
                self.unpacked_packages.append(package)

        total_delay_cost = sum(pkg.delay_cost for pkg in self.unpacked_packages)
        priority_spread_cost = sum(
            [self.priority_spread_cost if is_prio_uld else 0 for is_prio_uld in self.prio_ulds.values()]
        )
        total_cost = total_delay_cost + priority_spread_cost

        return (
            self.packed_positions,
            self.packed_packages,
            self.unpacked_packages,
            self.prio_ulds,
            total_cost,
        )
