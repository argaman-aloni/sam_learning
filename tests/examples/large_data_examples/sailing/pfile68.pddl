;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_13_12)

	(:domain sailing)

	(:objects
		b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - boat
		p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 - person
	)

  (:init
		(= (x b0) -8)
(= (y b0) 0)
(= (x b1) 2)
(= (y b1) 0)
(= (x b2) -1)
(= (y b2) 0)
(= (x b3) 9)
(= (y b3) 0)
(= (x b4) 5)
(= (y b4) 0)
(= (x b5) 7)
(= (y b5) 0)
(= (x b6) 7)
(= (y b6) 0)
(= (x b7) -4)
(= (y b7) 0)
(= (x b8) -1)
(= (y b8) 0)
(= (x b9) 10)
(= (y b9) 0)
(= (x b10) 4)
(= (y b10) 0)
(= (x b11) 9)
(= (y b11) 0)
(= (x b12) -7)
(= (y b12) 0)

		(= (d p0) 45)
(= (d p1) 1)
(= (d p2) 98)
(= (d p3) 55)
(= (d p4) 67)
(= (d p5) 21)
(= (d p6) 43)
(= (d p7) 7)
(= (d p8) 33)
(= (d p9) 96)
(= (d p10) 37)
(= (d p11) 26)

	)

	(:goal
		(and
			(saved p0)
(saved p1)
(saved p2)
(saved p3)
(saved p4)
(saved p5)
(saved p6)
(saved p7)
(saved p8)
(saved p9)
(saved p10)
(saved p11)
		)
	)
)
