from typing import List


def powerset(given_set: set) -> List[set]:
    set_size = len(given_set)
    power_set = []
    given_list = list(given_set)
    for i in range(1 << set_size):  # 2^set_size
        power_set.append(set([given_list[j] for j in range(set_size) if (i & (1 << j))]))

    return power_set


def combine_groupings(groupings: List[List[set]]) -> List[set]:
    """
    Combine sets that share common elements across all groupings into new groupings.
    """
    combined = []

    # Flatten and combine all sets that share elements
    for grouping in groupings:
        for param_set in grouping:
            merged = False

            for existing_set in combined:
                if not param_set.isdisjoint(existing_set):
                    existing_set.update(param_set)
                    merged = True
                    break

            if not merged:
                combined.append(param_set)

    return combined

