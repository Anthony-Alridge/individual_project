% BACKGROUND KNOWLEDGE
vocab(the).
vocab(man).
vocab(could).
vocab(not_res).
vocab(lift).
vocab(his).
vocab(son).
vocab(because).
vocab(he).
vocab(was).
vocab(so).
vocab(weak).
% An event which was prevented by a property can be explained by that property.
correct(X) :- property(P, X), solve(P).

% strong negation for not_... because ILASP doesn't seem to recognise strong negation.
:- event(X, Y, Z), no_event(X, Y, Z).

%% Modelling a sentence - extracting events and properties.
% event(V0, V1, V2) :- nsubj(V0, V1), dobj(V0, V2), not neg(V0, _).
%
% no_event(V0, V1, V2) :- nsubj(V0, V1), dobj(V0, V2), neg(V0, _).
% LANGUAGE BIAS
%%% Modelling what events can occur.
#bias("no_constraint.").
#bias(":- body(no_event(_, V0, V0)).").
#bias(":- body(event(_, V0, V0)).").
#bias(":- body(nsubj(V0, V0)).").
#bias(":- body(dobj(V0, V0)).").
#bias(":- body(neg(V0, V0)).").
#bias(":- head(no_event(V, V, V)).").
#bias(":- head(no_event(V, _, V)).").
#bias(":- head(no_event(_, V, V)).").
#bias(":- head(no_event(V, V, _)).").
#bias(":- head(no_event(V, _, _)), head(no_event(_, V, _)).").
#bias(":- head(no_event(V, _, _)), head(no_event(_, _, V)).").
#bias(":- head(no_event(_, V, _)), head(no_event(_, _, V)).").
#bias(":- head(property(_,_)), body(nsubj(_, _)).").
#bias(":- head(property(_,_)), body(dobj(_, _)).").
#bias(":- head(property(_,_)), body(neg(_, _)).").
#modeh(property(const(pr), var(entity))).
%%%% Model that properties can cause events, properties.
#modeb(1, event(const(ev), var(entity), var(entity))).
#modeb(1, no_event(const(ev), var(entity), var(entity))).

#modeh(no_event(var(vocab), var(vocab), var(vocab))).
#modeb(1, nsubj(var(vocab), var(vocab))).
#modeb(1, dobj(var(vocab), var(vocab))).
#modeb(1, neg(var(vocab), var(vocab))).

#modeb(1, entity(var(entity))).

#constant(pr, weak).
#constant(ev, lift).
#constant(pr, heavy).

% Examples
% For each sentence, the positive example says that the correct answer
% must be produced in at least one, and the negative example says the correct
% answer must be produced in none, given the correct background knowledge
% The ctx dependent knowledge is the sentence with associated dependencies,
% with the target pronoun subbed for the correct candidate in the +ve example
% and the incorrect candidate for the -ve example.
#pos(p1, {correct(man)}, {}, {
  nsubj(lift, man).
  aux(lift, could).
  neg(lift, not_res).
  dobj(lift, son).
  advcl(lift, was).
  mark(was, because).
  nsubj(was, man).
  acomp(was, weak).
  entity(man).
  entity(son).
  solve(weak).
  }).
#neg(n1, {correct(son)}, {}, {
  nsubj(lift, man).
  aux(lift, could).
  neg(lift, not_res).
  dobj(lift, son).
  advcl(lift, was).
  mark(was, because).
  nsubj(was, son).
  acomp(was, weak).
  entity(man).
  entity(son).
  solve(weak).
  }).

#pos(p2, {correct(son)}, {}, {
  nsubj(lift, man).
  aux(lift, could).
  neg(lift, not_res).
  dobj(lift, son).
  advcl(lift, was).
  mark(was, because).
  nsubj(was, son).
  acomp(was, heavy).
  entity(man).
  entity(son).
  solve(heavy).
  }).

#neg(n2, {correct(man)}, {}, {
  nsubj(lift, man).
  aux(lift, could).
  neg(lift, not_res).
  dobj(lift, son).
  advcl(lift, was).
  mark(was, because).
  nsubj(was, man).
  acomp(was, heavy).
  entity(man).
  entity(son).
  solve(heavy).
  }).
