import os.path
from pathlib import Path

from macq.generate.pddl import Generator
from macq.generate.pddl.generator import PlanningDomainsAPIError
from macq.generate.plan import Plan
from macq.trace import Trace, State, Action

base = Path(__file__).parent

def state_as_str(state: State, is_init=False) -> str:
    state_str = "(:init" if is_init else "(:state"
    for fluent, is_positive in state.fluents.items():
        if is_positive:
            objects = " ".join(o.name for o in fluent.objects)
            state_str += f" ({fluent.name} {objects})"
    state_str += ")"
    return state_str
def operator_as_str(action: Action)-> str:
    action_str = f"(operator: ({action.name} "
    objects = " ".join(o.name for o in action.obj_params)
    action_str += f"{objects}))"
    return action_str

def solve_problem_via_api(domain_filename: str ="", problem_filename: str="", generator: Generator=None) -> Plan:
    if generator is None:
        if not domain_filename or not problem_filename:
            raise ValueError("if no generator is given as argument, domain and problem file must have bounded values")

        generator = Generator(dom=domain_filename, prob=problem_filename)


    plan: Plan = generator.generate_plan()
    return plan


def create_trajectory(plan: Plan, domain_filename: str, problem_filename: str, generator: Generator=None) -> str:
    trajectory: str = "(\n"
    if generator is None:
        generator = Generator(dom=domain_filename, prob=problem_filename)

    trace: Trace = generator.generate_single_trace_from_plan(plan)
    trajectory += state_as_str(trace[0].state, is_init=True)+"\n"
    if trace[0].action is not None:
        trajectory += f"\n{operator_as_str(trace[0].action)}\n"

    if len(trace) >= 1:
        for step in trace[1:]:
            trajectory+= "\n"+state_as_str(step.state)+"\n"
            if step.action is None:
                break
            trajectory += f"\n{operator_as_str(step.action)}\n"

    trajectory += ")"
    return trajectory

def solve_and_create_trajectory(domain_filename: str, problem_filename: str) -> str:
    generator: Generator = Generator(dom=domain_filename, prob=problem_filename)
    plan: Plan = generator.generate_plan()
    print(plan.actions)

    trajectory = create_trajectory(plan, domain_filename, problem_filename, generator)
    return trajectory


if __name__ == '__main__':
    domains_dir = base / "domains"
    for diriec in os.listdir(str(domains_dir)):
        if diriec not in ["storage"]:
            continue
        domain = domains_dir / diriec / "domain.pddl"
        for i in range(1,len(os.listdir(domains_dir / diriec))):
            if i==2:
                if f"pfile{i}.pddl" not in os.listdir(domains_dir / diriec):
                    break
                try:
                    traj = solve_and_create_trajectory(domain_filename=str(domain),
                                                    problem_filename=str(domains_dir/ diriec/ f"pfile{i}.pddl"))
                    with open(domains_dir/ diriec/ f"pfile{i}.trajectory", "wt") as f:
                        f.write(traj)
                        f.close()

                except PlanningDomainsAPIError as e:
                    print(diriec+str(i))
                    print(e.__cause__)
                    continue

                except Exception as e:
                    print(diriec+str(i))
                    print(str(e))
                    continue

