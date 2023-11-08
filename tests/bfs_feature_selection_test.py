"""Module test for the Breadth First Search (BFS) feature selection algorithm."""
from collections import defaultdict

import math

import pytest
from pandas import DataFrame

from sam_learning.core.bfs_feature_selection import BFSFeatureSelector

TEST_ACTION_NAME = 'test_action'
TEST_FUNCTION_NAMES = ["(x)", "(y)", "(z)", "(w)"]


@pytest.fixture
def bfs_feature_selector() -> BFSFeatureSelector:
    bfs_feature_selector = BFSFeatureSelector(action_name=TEST_ACTION_NAME)
    return bfs_feature_selector


def test_init_search_data_structures_initialize_the_function_vocabulary_correctly(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    assert bfs_feature_selector._function_vocabulary == TEST_FUNCTION_NAMES


def test_expand_node_when_conducted_for_the_first_time_updates_open_list_and_close_list_correctly(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    assert len(bfs_feature_selector.open_list) == 1
    assert len(bfs_feature_selector.closed_list) == 0

    initial_feature_set = bfs_feature_selector._expand_node()
    assert len(bfs_feature_selector.open_list) == 4
    assert len(bfs_feature_selector.closed_list) == 1
    assert initial_feature_set == set()


def test_expand_node_when_conducted_for_the_second_time_updates_open_list_and_close_list_correctly(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    bfs_feature_selector._expand_node()
    next_set_of_features = bfs_feature_selector._expand_node()
    assert len(bfs_feature_selector.open_list) == len(TEST_FUNCTION_NAMES) - 1 + math.comb(
        len(TEST_FUNCTION_NAMES) - 1, 2)
    assert len(bfs_feature_selector.closed_list) == 2
    assert len(next_set_of_features) == 1


def test_expand_node_when_conducted_for_the_third_time_creates_open_list_with_unique_items(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    item_count = defaultdict(int)
    for item in bfs_feature_selector.open_list:
        item_count[str(sorted(list(item)))] += 1

    assert all(value == 1 for value in item_count.values())


def test_expand_node_when_conducted_for_the_sixth_time_creates_open_list_three_items_in_the_set(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    bfs_feature_selector._expand_node()
    assert max([len(item) for item in bfs_feature_selector.open_list]) == 2
    assert len(bfs_feature_selector.open_list) == math.comb(len(TEST_FUNCTION_NAMES), 2)
    bfs_feature_selector._expand_node()
    assert max([len(item) for item in bfs_feature_selector.open_list]) == 3


def test_apply_feature_selection_when_there_are_no_negative_samples_returns_empty_set(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    positive_samples_data = {
        '(x)': [1, 2, 3],
        '(y)': [1, 2, 3],
        '(z)': [1, 2, 3],
        '(w)': [1, 2, 3]
    }
    bfs_feature_selector.numeric_positive_samples = DataFrame(data=positive_samples_data)
    next_set_of_features = bfs_feature_selector.apply_feature_selection()
    assert next_set_of_features == []


def test_apply_feature_Selection_when_there_is_one_negative_sample_and_it_is_one_of_the_positive_samples_returns_empty_set(
        bfs_feature_selector: BFSFeatureSelector) -> None:
    bfs_feature_selector.init_search_data_structures(TEST_FUNCTION_NAMES)
    positive_samples_data = {
        '(x)': [1, 2, 3],
        '(y)': [2, 2, 3],
        '(z)': [1, 2, 7],
        '(w)': [1, 4, 32]
    }
    negative_samples_data = {
        '(x)': [3.5],
        '(y)': [1],
        '(z)': [1],
        '(w)': [1]
    }
    bfs_feature_selector.numeric_positive_samples = DataFrame(data=positive_samples_data)
    bfs_feature_selector.numeric_negative_samples = DataFrame(data=negative_samples_data)
    next_set_of_features = bfs_feature_selector.apply_feature_selection()
    assert next_set_of_features == ["(x)"]
