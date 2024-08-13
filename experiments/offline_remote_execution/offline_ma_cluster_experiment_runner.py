"""Runs the MA-SAM experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    # args = sys.argv
    # os.system(f"nohup bash -c '{sys.executable} multi_agent_experiment_runner.py "
    #           f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
    #           f"--executing_agents {args[3]}  --problems_prefix {args[4]} --negative_preconditions_policy {args[5]} "
    #           f"> results-{args[2]}.txt ' &")

    configurations = [
        # ("/sise/home/karato/work_tools/DomainsData/satellite_enhanced/", "satellite_combined_domain.pddl",
        #  "[satellite0,satellite1,satellite2,satellite3,satellite4,satellite5,satellite6,satellite7,satellite8,satellite9]",
        #  "pfile", 2),
        # ("/sise/home/karato/work_tools/DomainsData/blocksworld_enhanced/", "blocks_combined_domain.pddl",
        #  "[a1,a2,a3,a4]", "pfile", 2),
        # ("/sise/home/karato/work_tools/DomainsData/depots_enhanced/", "depots_combined_domain.pddl",
        #  "[depot0,depot1,depot2,depot3,distributor0,distributor1,distributor2,distributor3,driver0,driver1,driver2,driver3]",
        #  "pfile", 2),
        # ("/sise/home/karato/work_tools/DomainsData/driverlog_enhanced/", "driverlog_combined_domain.pddl",
        #  "[driver1,driver2,driver3,driver4,driver5,driver6,driver7,driver8]", "pfile", 2),
        ("/sise/home/karato/work_tools/DomainsData/logistics_enhanced/", "logistics_combined_domain.pddl",
         "[apn1,apn2,tru1,tru2,tru3,tru4,tru5]", "pfile", 2),
        ("/sise/home/karato/work_tools/DomainsData/rovers_enhanced/", "rover_combined_domain.pddl",
         "[rover0,rover1,rover2,rover3,rover4,rover5,rover6,rover7,rover8,rover9]", "pfile", 2),
        # ("/sise/home/karato/work_tools/DomainsData/sokoban_enhanced/", "sokoban_combined_domain.pddl",
        #  "[player-01,player-02,player-03,player-04]", "pfile")
    ]

    for config in configurations:
        working_directory_path, domain_file_name, executing_agents, problems_prefix, policy = config
        os.system(f"nohup bash -c '{sys.executable} multi_agent_experiment_runner.py "
                  f"--working_directory_path {working_directory_path} --domain_file_name {domain_file_name} "
                  f"--executing_agents {executing_agents} --problems_prefix {problems_prefix} "
                  f"--negative_preconditions_policy {policy} "
                  f"> results-{domain_file_name}.txt ' &")
