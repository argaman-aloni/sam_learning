"""Problem generator for the zeno-travel domain."""
import random
from enum import Enum

from typing import List, Tuple

import numpy

MAX_RAND = 10 ** 6


class AllowedDomainTypes(Enum):
    """Enum for the allowed domain types."""
    STRIPS = 1
    NUMERIC = 2


class ProblemMap:
    """Defines the map of the problem."""
    num_locations: int
    num_distances: int
    map: numpy.ndarray
    domain_type: AllowedDomainTypes

    def __init__(self, num_locations: int, num_distances: int, domain_type: AllowedDomainTypes):
        self.domain_type = domain_type
        self.num_locations = num_locations
        self.num_distances = num_distances
        self.map = numpy.zeros(shape=(num_locations, num_distances))
        for i in range(num_locations):
            for j in range(i + 1, num_distances):
                self.map[i][j] = random.randint(0, num_distances // 2) + num_distances / 2
                self.map[j][i] = self.map[i][j]

    def define_city_map(self) -> List[str]:
        """

        :return:
        """
        map_str = []
        if self.domain_type == AllowedDomainTypes.NUMERIC:
            for i in range(self.num_locations):
                for j in range(self.num_locations):
                    map_str.append(f"\t(= (distance city{i} city{j}) {self.map[i][j]})")

        return map_str

    def define_city_objects(self) -> List[str]:
        """

        :return:
        """
        return [f"\tcity{i} - city" for i in range(self.num_locations)]


class Locatable:
    location: int
    destination: int
    interesting: bool
    id: int

    def __init__(self, num_location: int):
        self.location = random.randint(0, num_location)
        self.destination = random.randint(0, num_location)
        self.interesting = True

class Airplane(Locatable):
    """Defines the airplane."""
    slow_burn_rate: int
    slow_speed: int
    fuel: int
    capacity: int
    fast_speed: int
    fast_burn_rate: int
    refuel_rate: int
    zoom_limit: int
    num_planes: int

    def __init__(self, id: int, num_locations: int, num_distances: int, num_planes: int, domain_type: AllowedDomainTypes):
        super(Airplane, self).__init__(num_locations)
        self.id = id
        self.num_planes = num_planes
        self.slow_burn_rate = random.randint(1, 5)
        self.slow_speed = random.randint(0, 100) + 100
        self.fuel = random.randint(0, self.slow_burn_rate * num_distances)
        self.capacity = random.randint(0, int(2.1 * random.random()) * self.slow_burn_rate * num_distances)
        self.fast_speed = random.randint(0, int(1.0 + random.random() * 2) * self.slow_speed)
        self.fast_burn_rate = random.randint(0, int(2.0 + random.random() * 2) * self.slow_burn_rate)
        self.refuel_rate = 2 * random.randint(0, self.slow_burn_rate * num_distances)
        self.zoom_limit = 1 + random.randint(0, 10)
        self.domain_type = domain_type
        if random.randint(0, 10) < 7:
            self.interesting = False

    def define_airplane_objects(self) -> List[str]:
        """

        :return:
        """
        return [f"\tplane{i} - aircraft" for i in range(self.num_planes)]

    def define_airplane_map(self) -> List[str]:
        """"""
        airplane_map = []
        airplane_location = f"\t(at plane{self.id} city{self.location})"
        if self.domain_type == AllowedDomainTypes.NUMERIC:
            # "\t(= (" << x << " plane" << id << ") " << y << ")\n";



class ZenoTravelGenerator:
    """Problem generator for the zeno-travel domain."""

    domain_type: AllowedDomainTypes

    def __init__(self, domain_type: AllowedDomainTypes):
        """Initialize the generator."""
        self.domain_type = domain_type
