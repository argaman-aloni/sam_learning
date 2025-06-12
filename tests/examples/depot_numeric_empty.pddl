(define (domain Depot)
(:requirements :typing :fluents)
(:types place locatable - object
	depot distributor - place
        truck hoist surface - locatable
        pallet crate - surface)

(:predicates (at ?x - locatable ?y - place) 
             (on ?x - crate ?y - surface)
             (in ?x - crate ?y - truck)
             (lifting ?x - hoist ?y - crate)
             (available ?x - hoist)
             (clear ?x - surface)
)

(:functions 
	(load_limit ?t - truck) 
	(current_load ?t - truck) 
	(weight ?c - crate)
	(fuel-cost)
)
	
(:action Drive
:parameters (?x - truck ?y - place ?z - place) 
:precondition (and )
:effect (and ))

(:action Lift
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and )
:effect (and ))

(:action Drop 
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and )
:effect (and ))

(:action Load
:parameters (?x - hoist ?y - crate ?z - truck ?p - place)
:precondition (and )
:effect (and ))

(:action Unload 
:parameters (?x - hoist ?y - crate ?z - truck ?p - place)
:precondition (and )
:effect (and ))

)
