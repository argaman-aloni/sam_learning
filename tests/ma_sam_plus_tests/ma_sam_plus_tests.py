"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent
from pytest import fixture

from sam_learning.core import LiteralCNF, group_params_from_clause
from sam_learning.learners import MASAMPlus, combine_groupings
from sam_learning.learners.ma_sam_plus import generate_supersets_of_actions

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0", "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam_plus(woodworking_ma_combined_domain: Domain) -> MASAMPlus:
    return MASAMPlus(woodworking_ma_combined_domain)


@fixture()
def do_plane_observation_component(multi_agent_observation: MultiAgentObservation) -> MultiAgentComponent:
    return multi_agent_observation.components[1]


@fixture()
def do_plane_first_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer0", "p2", "verysmooth", "natural", "varnished"])


@fixture()
def do_plane_second_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer1", "p2", "verysmooth", "natural", "untreated"])


@fixture()
def communicate_image_data_action_call() -> ActionCall:
    return ActionCall("communicate_image_data", ["rover0", "lander0", "objective4", "colour", "waypoint0", "waypoint1"])


@fixture()
def woodworking_literals_cnf(woodworking_ma_combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in woodworking_ma_combined_domain.actions.keys()]
    return LiteralCNF(action_names)


@fixture()
def rovers_literals_cnf(ma_rovers_domain: Domain) -> LiteralCNF:
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    return LiteralCNF(action_names)


@fixture()
def driverlog_ma_sam_plus(ma_driverlog_domain) -> MASAMPlus:
    return MASAMPlus(ma_driverlog_domain)


def test_learn_action_model_with_colliding_actions_returns_model_with_macro_actions(rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation):
    try:
        learned_domain, _, _ = rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation])
        print(learned_domain.to_pddl())
        assert True

    except:
        assert False


def test_learn_action_model_with_colliding_actions_returns_model_with_macro_actions_driverlog(
    driverlog_ma_sam_plus: MASAMPlus, ma_driverlog_observation
):
    try:
        learned_domain, _, _ = driverlog_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_driverlog_observation])
        print(learned_domain.to_pddl())
        assert True

    except:
        assert False


def test_extract_relevant_action_groups_with_no_observed_actions_returns_no_action_group(
    woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])

    action_group = woodworking_ma_sam_plus.extract_relevant_action_groups()
    assert len(action_group) == 0


def test_extract_relevant_action_groups_with_observed_actions_with_no_unsafe_actions_returns_no_action_group(
    woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_ma_sam_plus.observed_actions = ["do-immersion-varnish", "do-grind", "do-plane"]
    woodworking_ma_sam_plus.safe_actions = ["do-immersion-varnish"]

    action_group = woodworking_ma_sam_plus.extract_relevant_action_groups()
    assert len(action_group) == 0


def test_extract_relevant_action_groups_with_observed_actions_with_unsafe_actions_returns_action_groups(
    woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):
    # Note: the safeness of the actions is learned through the cnfs
    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf
    woodworking_literals_cnf.add_possible_effect(
        [("do-grind", "(surface-condition ?m ?oldcolour)"), ("do-immersion-varnish", "(surface-condition ?m ?colour)")]
    )
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"), ("do-plane", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect(
        [("do-plane", "(surface-condition ?m ?colour)"), ("do-immersion-varnish", "(surface-condition ?m ?colour)")]
    )
    woodworking_literals_cnf.add_possible_effect(
        [
            ("do-grind", "(surface-condition ?m ?oldcolour)"),
            ("do-plane", "(surface-condition ?m ?colour)"),
            ("do-immersion-varnish", "(surface-condition ?m ?colour)"),
        ]
    )
    woodworking_ma_sam_plus.observed_actions = ["do-grind", "do-plane", "do-immersion-varnish"]
    woodworking_ma_sam_plus.safe_actions = ["do-immersion-varnish"]
    woodworking_ma_sam_plus._unsafe_actions = {"do-grind", "do-plane"}

    action_groups = woodworking_ma_sam_plus.extract_relevant_action_groups()
    action_groups_names = {frozenset(u.name for u in group) for group in action_groups}
    expected_names = [
        {"do-immersion-varnish", "do-grind"},
        {"do-grind", "do-plane"},
        {"do-plane", "do-immersion-varnish"},
        {"do-grind", "do-plane", "do-immersion-varnish"},
    ]

    assert all(action_group in expected_names for action_group in action_groups_names)
    assert len(action_groups) >= 4


def test_extract_relevant_parameter_groupings_with_existing_action_group_returns_valid_parameter_groupings(
    woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])

    woodworking_literals_cnf.add_possible_effect(
        [
            ("do-grind", "(surface-condition ?m ?oldcolour)"),
            ("do-immersion-varnish", "(surface-condition ?m ?newcolour)"),
            ("do-plane", "(surface-condition ?m ?oldcolour)"),
        ]
    )

    woodworking_literals_cnf.add_possible_effect(
        [("do-grind", "(surface-condition ?m ?oldcolour)"), ("do-plane", "(surface-condition ?m ?oldcolour)")]
    )

    action_group_names = ["do-immersion-varnish", "do-grind", "do-plane"]

    parameter_grouping = woodworking_ma_sam_plus.extract_relevant_parameter_groupings(action_group_names)[0]
    real_parameter_grouping = [
        {("do-immersion-varnish", "?m"), ("do-grind", "?m"), ("do-plane", "?m")},
        {("do-immersion-varnish", "?newcolour"), ("do-grind", "?oldcolour"), ("do-plane", "?oldcolour")},
    ]

    assert parameter_grouping == real_parameter_grouping


def test_extract_relevant_parameter_groupings_with_non_existing_action_group_returns_no_parameter_groupings(
    woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_literals_cnf.add_possible_effect(
        [
            ("do-grind", "(surface-condition ?m ?oldcolour)"),
            ("do-immersion-varnish", "(surface-condition ?m ?newcolour)"),
            ("do-plane", "(surface-condition ?m ?oldcolour)"),
        ]
    )
    woodworking_literals_cnf.add_possible_effect(
        [("do-grind", "(surface-condition ?m ?oldcolour)"), ("do-plane", "(surface-condition ?m ?oldcolour)")]
    )

    action_group_names = ["do-glaze", "do-grind"]
    parameter_grouping = woodworking_ma_sam_plus.extract_relevant_parameter_groupings(action_group_names)[0]

    assert len(parameter_grouping) == 0


def test_group_params_from_clause_handles_non_unit_clause_returns_parameters_grouping_with_groups_consisting_of_several_actions():
    clause = [("do-grind", "(m ?x ?y)"), ("do-plane", "(m ?x ?z)")]

    group = group_params_from_clause(clause)
    real_group = [{("do-grind", "?x"), ("do-plane", "?x")}, {("do-grind", "?y"), ("do-plane", "?z")}]

    assert group == real_group


def test_group_params_from_clause_handles_unit_clause_returns_parameters_grouping_with_groups_consisting_of_solo_actions():
    clause = [
        ("do-grind", "(m ?x ?y)"),
    ]

    group = group_params_from_clause(clause)

    assert group == [{("do-grind", "?x")}, {("do-grind", "?y")}]


def test_extract_effects_for_macro_from_cnf_returns_effects_adapted_to_macro_naming(
    rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, ma_rovers_domain: Domain
):
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    cnf2 = LiteralCNF(action_names)
    cnf3 = LiteralCNF(action_names)
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = cnf1
    rovers_ma_sam_plus.literals_cnf["fluent_test2 ?w"] = cnf2
    rovers_ma_sam_plus.literals_cnf["fluent_test3 ?x"] = cnf3

    cnf1.add_possible_effect([("communicate_rock_data", "(fluent_test1 ?p)"), ("navigate", "(fluent_test1 ?y)")])
    cnf2.add_possible_effect([("communicate_rock_data", "(fluent_test2 ?p)"), ("navigate", "(fluent_test2 ?y)")])

    param_grouping = [{("communicate_rock_data", "?p"), ("navigate", "?y")}]
    lma_names = ["navigate", "communicate_rock_data"]
    action_group = {action for action in rovers_ma_sam_plus.partial_domain.actions.values() if action.name in lma_names}

    mapping = {
        ("navigate", "?x"): "?x'0",
        ("navigate", "?y"): "?yp",
        ("navigate", "?z"): "?z'0",
        ("communicate_rock_data", "?r"): "?r'1",
        ("communicate_rock_data", "?l"): "?l'1",
        ("communicate_rock_data", "?p"): "?yp",
        ("communicate_rock_data", "?x"): "?x'1",
        ("communicate_rock_data", "?y"): "?y'1",
    }

    effects = rovers_ma_sam_plus.extract_effects_for_macro_from_cnf(action_group, param_grouping, mapping)

    effects_rep = [effect.untyped_representation for effect in effects]
    assert "(fluent_test1 ?yp)" in effects_rep
    assert "(fluent_test2 ?yp)" in effects_rep
    assert len(effects_rep) == 2


def test_extract_effects_for_macro_from_cnf_returns_effects_with_no_duplications(
    rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, ma_rovers_domain: Domain
):
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = cnf1
    cnf1.add_possible_effect(
        [
            ("communicate_rock_data", "(fluent_test1 ?p)"),
            ("communicate_rock_data", "(fluent_test1 ?p)"),
            ("navigate", "(fluent_test1 ?y)"),
            ("navigate", "(fluent_test1 ?y)"),
        ]
    )

    param_grouping = [{("communicate_rock_data", "?p"), ("navigate", "?y")}]
    lma_names = ["navigate", "communicate_rock_data"]
    action_group = {action for action in rovers_ma_sam_plus.partial_domain.actions.values() if action.name in lma_names}
    mapping = {
        ("navigate", "?x"): "?x'0",
        ("navigate", "?y"): "?yp",
        ("navigate", "?z"): "?z'0",
        ("communicate_rock_data", "?r"): "?r'1",
        ("communicate_rock_data", "?l"): "?l'1",
        ("communicate_rock_data", "?p"): "?yp",
        ("communicate_rock_data", "?x"): "?x'1",
        ("communicate_rock_data", "?y"): "?y'1",
    }

    effects = rovers_ma_sam_plus.extract_effects_for_macro_from_cnf(action_group, param_grouping, mapping)
    effects_rep = [effect.untyped_representation for effect in effects]
    assert len(set(effects_rep)) == len(effects_rep)
    print(effects_rep)


def test_extract_preconditions_for_macro_from_cnf_returns_preconditions_adapted_to_macro_naming(
    rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, ma_rovers_domain: Domain
):
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    cnf2 = LiteralCNF(action_names)
    cnf3 = LiteralCNF(action_names)
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = cnf1
    rovers_ma_sam_plus.literals_cnf["fluent_test2 ?w"] = cnf2
    rovers_ma_sam_plus.literals_cnf["fluent_test3 ?x"] = cnf3

    # fluent_test1 and fluent_test2 will have consistent clauses hence only fluent_test3 will extract a precondition.
    cnf1.add_possible_effect([("communicate_rock_data", "(fluent_test1 ?p)"), ("navigate", "(fluent_test1 ?y)")])
    cnf2.add_possible_effect([("communicate_rock_data", "(fluent_test2 ?p)"), ("navigate", "(fluent_test2 ?y)")])
    cnf2.add_possible_effect([("communicate_rock_data", "(fluent_test3 ?p)"), ("drop", "(fluent_test3 ?x)")])

    parameter_grouping = [{("communicate_rock_data", "?p"), ("navigate", "?y")}]
    action_group = {rovers_ma_sam_plus.partial_domain.actions["navigate"], rovers_ma_sam_plus.partial_domain.actions["communicate_rock_data"]}

    mapping = {
        ("navigate", "?x"): "?x'0",
        ("navigate", "?y"): "?yp",
        ("navigate", "?z"): "?z'0",
        ("communicate_rock_data", "?r"): "?r'1",
        ("communicate_rock_data", "?l"): "?l'1",
        ("communicate_rock_data", "?p"): "?yp",
        ("communicate_rock_data", "?x"): "?x'1",
        ("communicate_rock_data", "?y"): "?y'1",
    }

    precondition = rovers_ma_sam_plus.extract_preconditions_for_macro_from_cnf(action_group, parameter_grouping, mapping)

    precondition_rep = [precondition.untyped_representation for precondition in precondition.root.operands]
    assert len(precondition_rep) == 1
    assert "(fluent_test3 ?yp)" in precondition_rep


def test_extract_preconditions_for_macro_from_cnf_returns_no_duplication_of_preconditions(
    rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall, ma_rovers_domain: Domain, rovers_literals_cnf
):
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = rovers_literals_cnf

    rovers_literals_cnf.add_possible_effect(
        [("communicate_rock_data", "(fluent_test1 ?p)"), ("navigate", "(fluent_test1 ?y)"), ("sample_rock", "(fluent_test1 ?x)")]
    )

    parameter_grouping = [{("communicate_rock_data", "?p"), ("navigate", "?y")}]
    action_group = {rovers_ma_sam_plus.partial_domain.actions["navigate"], rovers_ma_sam_plus.partial_domain.actions["communicate_rock_data"]}

    mapping = {
        ("navigate", "?x"): "?x'0",
        ("navigate", "?y"): "?yp",
        ("navigate", "?z"): "?z'0",
        ("communicate_rock_data", "?r"): "?r'1",
        ("communicate_rock_data", "?l"): "?l'1",
        ("communicate_rock_data", "?p"): "?yp",
        ("communicate_rock_data", "?x"): "?x'1",
        ("communicate_rock_data", "?y"): "?y'1",
        ("sample_rock", "?x"): "?x'2",
    }

    # clause is not consistent because it has one more action not relevant to the action group.
    # in theory there are two preconditions here extracted from the cnf, for navigate and for sample rock.
    # but after adaptation to macro, they look the same, hence only one is returned.
    precondition = rovers_ma_sam_plus.extract_preconditions_for_macro_from_cnf(action_group, parameter_grouping, mapping)

    precondition_rep = [precondition.untyped_representation for precondition in precondition.root.operands]
    assert len(precondition_rep) == 1
    assert "(fluent_test1 ?yp)" in precondition_rep


def test_combine_groupings_with_no_shared_elements_returns_no_change_to_grouping():
    # No elements are shared, so the output should be the same as the input.
    groupings = [{("navigate", "?x"), ("go", "?y")}, {("go", "?l"), ("arrive", "?z")}, {("fly", "?w")}, {("navigate", "?y"), ("fly", "?m")}]

    result = combine_groupings(groupings)

    # Expected output should be the same as the input
    expected_output = [{("navigate", "?x"), ("go", "?y")}, {("go", "?l"), ("arrive", "?z")}, {("fly", "?w")}, {("navigate", "?y"), ("fly", "?m")}]
    assert result == expected_output


def test_combine_groupings_with_some_shared_elements_returns_grouping_with_all_shared_elements_in_same_group():
    # Sets with shared elements should be merged.
    groupings = [
        {("navigate", "?x"), ("go", "?y")},
        {("go", "?y"), ("arrive", "?z")},
        {("fly", "?w"), ("arrive", "?z")},
        {("navigate", "?y"), ("fly", "?z")},
    ]
    expected_output = [{("navigate", "?x"), ("go", "?y"), ("arrive", "?z"), ("fly", "?w")}, {("navigate", "?y"), ("fly", "?z")}]
    assert combine_groupings(groupings) == expected_output


def test_combine_groupings_with_all_shared_elements_returns_grouping_with_one_maximal_group():
    # All sets should merge into one big set.
    groupings = [
        {("navigate", "?x"), ("go", "?y")},
        {("go", "?y"), ("arrive", "?z")},
        {("arrive", "?z"), ("fly", "?w")},
        {("fly", "?w"), ("navigate", "?x")},
    ]
    expected_output = {("navigate", "?x"), ("go", "?y"), ("arrive", "?z"), ("fly", "?w")}
    assert combine_groupings(groupings) == [expected_output]


def test_generate_supersets_does_not_change_original_set_if_cannot_create_supersets():
    # Test 1: Basic Functionality
    original_sets = [{"1", "2"}]
    result = generate_supersets_of_actions(original_sets)
    expected = [{"1", "2"}]  # The original set and all its supersets
    assert sorted(result, key=lambda x: (len(x), x)) == sorted(expected, key=lambda x: (len(x), x)), "Test failed!"


def test_generate_supersets_creates_supersets_containing_unique_objects_only():
    # Test 1: Basic Functionality
    original_sets = [{"1", "2"}, {"2", "3"}]
    result = generate_supersets_of_actions(original_sets)
    expected = [{"1", "2"}, {"2", "3"}, {"1", "2", "3"}]  # The original set and all its supersets
    assert sorted(result, key=lambda x: (len(x), x)) == sorted(expected, key=lambda x: (len(x), x)), "Test failed!"


def test_generate_supersets_does_not_break_original_sets_and_only_adds_new_sets_combining_all_the_original_strings():
    # Test 1: Basic Functionality
    original_sets = [{"1", "2"}, {"3", "4"}]
    result = generate_supersets_of_actions(original_sets)
    expected = [{"1", "2"}, {"3", "4"}, {"1", "2", "3", "4"}]  # The original set and all its supersets
    assert sorted(result, key=lambda x: (len(x), x)) == sorted(expected, key=lambda x: (len(x), x)), "Test failed!"
