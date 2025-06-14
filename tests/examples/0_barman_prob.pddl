(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot1 shot2 - shot
      ingredient1 ingredient2 ingredient3 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser1 dispenser2 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot1)
  (ontable shot2)
  (dispenses dispenser1 ingredient1)
  (dispenses dispenser2 ingredient2)
  (dispenses dispenser3 ingredient3)
  (clean shaker1)
  (clean shot1)
  (clean shot2)
  (empty shaker1)
  (empty shot1)
  (empty shot2)
  (handempty left)
  (handempty right)
  (shaker_empty_level shaker1 l0)
  (shaker_level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail_part1 cocktail1 ingredient1)
  (cocktail_part2 cocktail1 ingredient2)
  (cocktail_part1 cocktail2 ingredient1)
  (cocktail_part2 cocktail2 ingredient3)
)
 (:goal
  (and
      (contains shot1 cocktail1)
      (contains shot2 cocktail2)
)))
