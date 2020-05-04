nsubj(go, pronoun).
-event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), neg(Event, Sub), not goal_event(Event).
event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), not neg(Event, Sub), not goal_event(Event).
property(Name, X) :- nsubj(Name, X).
property(Name, X) :- nsubj(Z, X), acomp(Z, Name).
:- not goal.