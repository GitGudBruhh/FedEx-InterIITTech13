import pandas as pd
from typing import List, Tuple


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
