import json
import random
import shutil
from pathlib import Path

crafting_table_location_in_map = [20, 4, 20]
NUM_MAPS_TO_GENERATE = 200


def generate_maps_using_random_trees(
        example_map_path: Path, output_directory_path: Path, compiled_json_file: Path) -> None:
    """

    :param example_map_path:
    :param output_directory_path:
    :param compiled_json_file:
    :return:
    """
    with open(example_map_path, "r") as map_file:
        map_json = json.load(map_file)
        for i in range(NUM_MAPS_TO_GENERATE):
            output_map_file_path = output_directory_path / f"test_map_instance_{i}.json"
            num_trees = random.randint(4, 20)
            tree_locations_in_map = []
            objects_in_map = [{'blockPos': [20, 4, 20], 'blockName': 'minecraft:crafting_table'}]
            for j in range(num_trees):
                new_x_coordinate = random.randint(1, 30)
                new_y_coordinate = random.randint(1, 30)
                while [new_x_coordinate, 4, new_y_coordinate] in tree_locations_in_map or \
                        [new_x_coordinate, 4, new_y_coordinate] == crafting_table_location_in_map:
                    new_x_coordinate = random.randint(1, 30)
                    new_y_coordinate = random.randint(1, 30)

                tree_locations_in_map.append([new_x_coordinate, 4, new_y_coordinate])
                objects_in_map.append({'blockPos': [new_x_coordinate, 4, new_y_coordinate], 'blockName': 'tree'})

            map_json["features"][2]["blockList"] = objects_in_map
            with open(output_map_file_path, "wt") as output:
                json.dump(map_json, output)

            shutil.copy(compiled_json_file, output_directory_path / f"test_map_instance_{i}.json2")


if __name__ == '__main__':
    generate_maps_using_random_trees(
        example_map_path=EXAMPLES_DIR_PATH / "minecraft_map.json",
        output_directory_path=EXAMPLES_DIR_PATH / "minecraft_maps",
        compiled_json_file=EXAMPLES_DIR_PATH / "minecraft_map.json2"
    )
