# Multi-Agent SAM Documentation:

This contains the relevant information for the multi-agent version of the SAM-Learning framework.

## Supported Features
* Learning action models when the agents are acting concurrently in the environment.
* Learning simple collaborative actions when the algorithm cannot differentiate between the agents' actions.
* **Next**: Learning collaborative actions that may have different effects than the individual actions.

## Usage
Note - the experiment framework is a bit deprecated and needs to be updated.

* The main algorithm is [multi_agent_sam.py](..%2Fsam_learning%2Flearners%2Fmulti_agent_sam.py). 
* The code contains the algorithm for learning multi-agent action models.
* The algorithm heavily relies on [sam_learning.py](..%2Fsam_learning%2Flearners%2Fsam_learning.py) which is the lifted action model learning algorithm.
* There is some core functionality relevant to the CNF management is implemented in [literals_cnf.py](..%2Fsam_learning%2Fcore%2Fpropositional_operations%2Fliterals_cnf.py).

