import os
import shutil


def copy_matching_trajectories(dataset_dir, pddl_dir):
    # Collect base names of all PDDL files
    pddl_basenames = {os.path.splitext(f)[0] for f in os.listdir(pddl_dir) if f.endswith(".pddl")}

    # Ensure target directory exists
    os.makedirs(pddl_dir, exist_ok=True)

    # Iterate over trajectory files in dataset
    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".trajectory"):
            continue

        base = os.path.splitext(filename)[0]

        # Copy only if the base matches a PDDL base name
        if base in pddl_basenames:
            src = os.path.join(dataset_dir, filename)
            dst = os.path.join(pddl_dir, filename)
            shutil.copy2(src, dst)
            print(f"Copied: {filename}")


if __name__ == "__main__":
    dataset_dir = "/home/mordocha/numeric_planning/domains/domains_for_new_nsam/satellite/"
    pddl_dir = "/home/mordocha/numeric_planning/domains/domains_for_new_nsam/satellite/train/fold_0_3_80/"
    copy_matching_trajectories(dataset_dir, pddl_dir)
