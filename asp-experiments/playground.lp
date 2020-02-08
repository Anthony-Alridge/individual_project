% A temporary file for testing random things.

person(ian).
person(dennis).

context(eats, munedo).
context(eats, intestine).
%
% any(X) :- person(X).
%
eats(ian, X) :- context(eats, X).

{-eats(X, Y) : person(X), context(eats, Y)}.

solution(ian) :- -eats(ian, intestine).
solution(dennis) :- -eats(dennis, intestine).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This simplifies the sentence into the central event (eat)
% It is arguably an easier problem because of the direct contradication
% between the pronoun event and the central event.
% The problem of not knowing the relation between intestine and munedo is
% handled by taking an assumption that all events not explicitly forbidden
% and possible are allowed.
% This could be automated without needed additional knowledge, just by parsing
% the sentence.
