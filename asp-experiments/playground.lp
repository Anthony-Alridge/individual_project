%%% Background knowledge
% We know how to reason given appropriate background knowledge.
correct :- holds(P, sentence), -occurs(E, sentence), -occurs(E, background).
% A simple way to start the search.
holds(P, background) :- holds(P, sentence).

%%% Background knowledge manually encoded from sentence.
-occurs(event(lift, man, son), sentence).
holds(property(weak, man), sentence).


% what should be learnt.
-occurs(event(lift, X, son), background) :- holds(property(weak, X), background).



%% Add to generated program. Run with both candidates, answer is one with answer sets.
% :- not correct.
