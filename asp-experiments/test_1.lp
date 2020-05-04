entity(man).
entity(son).
nsubj(lift, man).
neg(lift, man).
goal :- goal_event(Event), -event(Event, _, _).
goal_event(lift).
dobj(lift, son).
neg(lift, son).
nsubj(be, pronoun).
acomp(be, weak).
-event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), neg(Event, Sub), not goal_event(Event).
event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), not neg(Event, Sub), not goal_event(Event).
property(Name, X) :- nsubj(Name, X).
property(Name, X) :- nsubj(Z, X), acomp(Z, Name).
:- not goal.



% if weak then cannot lift
-event(lift, S, Y) :- property(weak, S), entity(Y).
