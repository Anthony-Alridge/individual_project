%nsubj(lift, man).
%neg(lift, man).
%dobj(lift, son).
neg(lift, son).
nsubj(be, pronoun).
acomp(be, weak).
-event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), neg(Event, Sub).
event(Event, Sub, Obj) :- nsubj(Event, Sub), dobj(Event, Obj), not neg(Event, Sub).
property(Name, X) :- nsubj(Name, X).
property(Name, X) :- nsubj(Z, X), acomp(Z, Name).

%Defining a task
% goal is any generated event?
% how to stop that event from being generated, so that it can be
% explained.
% When generating rules, if event is a candidate antecedent,
% make a goal rule for it instead of a LexicalRule.
% -event(lift, man, son).
%property(weak, X) :- candidate(X).
% Needed background knowledge
% -event(lift, X, Y) :- property(weak, X).
%goal :- -event(lift, man, son).
%:- not goal.
