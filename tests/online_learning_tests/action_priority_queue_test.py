"""Module test for the action priority queue."""

from sam_learning.core.online_learning.priority_queue import PriorityQueue


def test_insert_adds_an_item_with_priority():
    """Tests that the insert method adds an item with a given priority."""
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    assert len(queue) == 1


def test_insert_adds_an_item_with_priority_to_the_same_list_when_inserting_priority_a_second_time():
    """Tests that the insert method adds an item with a given priority adds to the same queue when the priority is the
    same."""
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    queue.insert("item2", 1.0, 0.5)
    assert len(queue) == 2
    assert len(queue._prioritized_queue[1.0]) == 2
    assert len(queue._prioritized_queue) == 1


def test_get_item_when_there_is_only_one_item_removes_the_key_from_the_dictionary():
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    queue.get_item()
    assert len(queue) == 0


def test_get_item_get_the_item_that_was_inserted_without_the_additional_data():
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    result = queue.get_item()
    assert result == "item"


def test_get_item_removes_item_from_the_list_when_the_internal_queue_contains_more_than_one_item():
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    queue.insert("item2", 1.0, 0.5)
    queue.get_item()
    assert len(queue) == 1


def test_clear_removes_all_items_from_internal_queue():
    queue = PriorityQueue()
    queue.insert("item", 1.0, 0.5)
    queue.insert("item2", 1.0, 0.5)
    queue.clear()
    assert len(queue) == 0
