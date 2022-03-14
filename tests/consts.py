"""Constants for the tests."""
import os
from pathlib import Path

from pddl_plus_parser.models import PDDLType, Predicate

CWD = os.getcwd()
EXAMPLES_DIR_PATH = Path(CWD, "examples")
DOMAIN_NO_CONSTS_PATH = EXAMPLES_DIR_PATH / "domain-logistics.pddl"
DOMAIN_WITH_CONSTS_PATH = EXAMPLES_DIR_PATH / "woodworking-domain.pddl"
NUMERIC_DOMAIN_WITH_PATH = EXAMPLES_DIR_PATH / "depot_numeric.pddl"

OBJECT_TYPE = PDDLType(name="object")
AGENT_TYPE = PDDLType(name="agent")
CITY_TYPE = PDDLType(name="city", parent=OBJECT_TYPE)
WOODOBJ_TYPE = PDDLType(name="woodobj", parent=OBJECT_TYPE)
SURFACE_TYPE = PDDLType(name="surface", parent=OBJECT_TYPE)
TREATMENT_STATUS_TYPE = PDDLType(name="treatmentstatus", parent=OBJECT_TYPE)
PART_TYPE = PDDLType(name="part", parent=WOODOBJ_TYPE)
TAXI_TYPE = PDDLType(name="taxi", parent=AGENT_TYPE)
TRUCK_TYPE = PDDLType(name="truck", parent=AGENT_TYPE)
AIRPLANE_TYPE = PDDLType(name="airplane", parent=AGENT_TYPE)
LOCATION_TYPE = PDDLType(name="location", parent=OBJECT_TYPE)

AT_TRUCK_PREDICATE = Predicate(name="at",
                               signature={"?a": AGENT_TYPE,
                                          "?loc": LOCATION_TYPE})

