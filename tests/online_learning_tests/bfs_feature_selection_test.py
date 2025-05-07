"""Module test for the Breadth First Search (BFS) feature selection algorithm."""

import pytest
from pandas import DataFrame

from sam_learning.core.online_learning.bfs_feature_selection import BFSFeatureSelector

TEST_ACTION_NAME = "test_action"
TEST_FUNCTION_NAMES = ["(x)", "(y)", "(z)", "(w)"]
TEST_PREDICATES = ["(px)", "(py)", "(pz)", "(pw)"]


@pytest.fixture
def bfs_feature_selector() -> BFSFeatureSelector:
    bfs_feature_selector = BFSFeatureSelector(action_name=TEST_ACTION_NAME, pb_monomials=TEST_FUNCTION_NAMES, pb_predicates=TEST_PREDICATES)
    bfs_feature_selector.initialize_open_list()
    return bfs_feature_selector


def test_initialize_open_list_initialize_the_open_list_with_increasing_internal_list_sizes(bfs_feature_selector: BFSFeatureSelector) -> None:
    assert len(bfs_feature_selector.open_list) == 16
    assert bfs_feature_selector.open_list == [
        [],
        ["(x)"],
        ["(y)"],
        ["(z)"],
        ["(w)"],
        ["(x)", "(y)"],
        ["(x)", "(z)"],
        ["(x)", "(w)"],
        ["(y)", "(z)"],
        ["(y)", "(w)"],
        ["(z)", "(w)"],
        ["(x)", "(y)", "(z)"],
        ["(x)", "(y)", "(w)"],
        ["(x)", "(z)", "(w)"],
        ["(y)", "(z)", "(w)"],
        ["(x)", "(y)", "(z)", "(w)"],
    ]


def test_add_new_observation_when_adding_the_first_observation_correctly_sets_the_observations_dataframe(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(observation_df, is_successful=True)
    assert len(bfs_feature_selector._observations) == 1
    assert len(bfs_feature_selector._observations.iloc[0]) == 8 + 1  # 8 features + 1 success
    assert bfs_feature_selector._observations.iloc[0].tolist() == [1, 2, 3, 4, True, False, True, False, True]


def test_find_matching_successful_transition_row_when_dataset_dataframe_contains_a_row_with_subset_of_new_input_observation_and_the_state_was_successful_returns_true(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    reference_df = DataFrame.from_dict(
        {
            "(x)": [1, 2, 3, 4, 5],
            "(y)": [2, 3, 4, 5, 6],
            "(z)": [3, 4, 5, 6, 7],
            "(w)": [4, 5, 6, 7, 8],
            "(px)": [True, True, False, True, False],
            "(py)": [False, False, True, False, True],
            "(pz)": [True, True, False, True, False],
            "(pw)": [False, False, True, False, True],
            "is_successful": [True, True, False, True, False],
        }
    )
    single_row_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [True], "(pz)": [True], "(pw)": [False],}
    )

    match_found = bfs_feature_selector._find_matching_successful_transition_row(
        reference_df=reference_df, single_row_df=single_row_df, columns=["(px)", "(py)", "(pz)", "(pw)"]
    )
    assert match_found


def test_find_matching_successful_transition_row_when_dataset_dataframe_contains_a_row_with_data_same_as_observation_and_the_state_was_successful_returns_true(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    reference_df = DataFrame.from_dict(
        {
            "(x)": [1, 2, 3, 4, 5],
            "(y)": [2, 3, 4, 5, 6],
            "(z)": [3, 4, 5, 6, 7],
            "(w)": [4, 5, 6, 7, 8],
            "(px)": [True, True, False, True, False],
            "(py)": [False, False, True, False, True],
            "(pz)": [True, True, False, True, False],
            "(pw)": [False, False, True, False, True],
            "is_successful": [True, True, False, True, False],
        }
    )
    single_row_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )

    match_found = bfs_feature_selector._find_matching_successful_transition_row(
        reference_df=reference_df, single_row_df=single_row_df, columns=["(px)", "(py)", "(pz)", "(pw)"]
    )
    assert match_found


def test_find_matching_successful_transition_row_when_dataset_dataframe_contains_a_row_with_superset_of_state_data_and_the_state_was_successful_returns_false(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    reference_df = DataFrame.from_dict(
        {
            "(x)": [1, 2, 3, 4, 5],
            "(y)": [2, 3, 4, 5, 6],
            "(z)": [3, 4, 5, 6, 7],
            "(w)": [4, 5, 6, 7, 8],
            "(px)": [True, True, False, True, False],
            "(py)": [False, False, True, False, True],
            "(pz)": [True, True, False, True, False],
            "(pw)": [False, False, True, False, True],
            "is_successful": [True, True, False, True, False],
        }
    )
    single_row_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [False], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )

    match_found = bfs_feature_selector._find_matching_successful_transition_row(
        reference_df=reference_df, single_row_df=single_row_df, columns=["(px)", "(py)", "(pz)", "(pw)"]
    )
    assert not match_found


def test_find_matching_successful_transition_row_when_dataset_dataframe_contains_a_row_with_data_same_as_observation_and_the_state_was_unsuccessful_returns_false(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    reference_df = DataFrame.from_dict(
        {
            "(x)": [1, 2, 3, 4, 5],
            "(y)": [2, 3, 4, 5, 6],
            "(z)": [3, 4, 5, 6, 7],
            "(w)": [4, 5, 6, 7, 8],
            "(px)": [True, True, False, True, False],
            "(py)": [False, False, True, False, True],
            "(pz)": [True, True, False, True, False],
            "(pw)": [False, False, True, False, True],
            "is_successful": [False, False, False, False, True],
        }
    )
    single_row_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )

    match_found = bfs_feature_selector._find_matching_successful_transition_row(
        reference_df=reference_df, single_row_df=single_row_df, columns=["(px)", "(py)", "(pz)", "(pw)"]
    )
    assert not match_found


def test_add_new_observation_when_adding_the_first_observation_returns_empty_list_of_string_as_selected_features(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    selected_features = bfs_feature_selector.add_new_observation(observation_df, is_successful=True)
    assert len(selected_features) == 0


def test_add_new_observation_when_adding_the_same_observation_twice_does_not_change_the_observations_dataframe(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(observation_df, is_successful=True)
    bfs_feature_selector.add_new_observation(observation_df, is_successful=True)
    assert len(bfs_feature_selector._observations) == 1


def test_add_new_observation_when_an_observation_exists_and_additional_observation_added_when_the_state_is_successful_does_not_change_selected_features(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    first_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(first_observation_df, is_successful=True)
    second_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [False], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    selected_features = bfs_feature_selector.add_new_observation(second_observation_df, is_successful=True)
    assert len(selected_features) == 0


def test_add_new_observation_when_an_observation_exists_and_additional_observation_added_when_the_state_is_unsuccessful_but_the_predicates_are_different_and_the_numeric_part_is_the_same_does_not_change_the_selected_features(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    first_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(first_observation_df, is_successful=True)
    second_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [False], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    selected_features = bfs_feature_selector.add_new_observation(second_observation_df, is_successful=False)
    assert len(selected_features) == 0


def test_add_new_observation_when_an_observation_exists_and_additional_observation_added_when_the_state_is_unsuccessful_adds_observation_to_observations_df(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    first_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(first_observation_df, is_successful=True)
    second_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [False], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(second_observation_df, is_successful=False)
    assert len(bfs_feature_selector._observations) == 2


def test_add_new_observation_when_an_observation_exists_and_additional_observation_added_when_the_state_is_unsuccessful_and_predicates_are_the_same_but_numeric_values_differ_changes_the_selected_features(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    first_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(first_observation_df, is_successful=True)
    second_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [8], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    selected_features = bfs_feature_selector.add_new_observation(second_observation_df, is_successful=False)
    assert len(selected_features) == 1


def test_add_new_observation_when_have_unsuccessful_observation_in_dataframe_does_not_change_the_selected_features(
    bfs_feature_selector: BFSFeatureSelector,
) -> None:
    first_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [2], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    bfs_feature_selector.add_new_observation(first_observation_df, is_successful=False)
    second_observation_df = DataFrame.from_dict(
        {"(x)": [1], "(y)": [8], "(z)": [3], "(w)": [4], "(px)": [True], "(py)": [False], "(pz)": [True], "(pw)": [False],}
    )
    selected_features = bfs_feature_selector.add_new_observation(second_observation_df, is_successful=False)
    assert len(selected_features) == 0
