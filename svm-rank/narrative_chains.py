# import re
# from spacy import displacy
#

# Add documentation and some tests

# class NarrativeChainsFeature():
#     def __init__(self):
#         with open('../data/narrative_chains.txt', 'r') as f:
#             self.chains = f.readlines()
#
#     # Returns the role the pronoun plays in matching event chains.
#     # Returns None if no chains match, or if the detected role for the pronoun
#     # is conflicting
#     def get_role(self, event_tuples):
#         role = None
#         for c_event, p_event in event_tuples:
#             for chain in self.chains:
#                 events = chain.split()
#                 # This chain matches the event_tuple
#                 if p_event in events:
#                     if c_event + '-o' in events:
#                         # pronoun plays object role
#                         if role == 's':
#                             return None  # Clashing roles
#                         role = 'o'
#                     if c_event + '-s' in events:
#                         if role == 'o':
#                             return None  # Clashing roles
#                         role = 's'
#         return role
#
#     """
#     Given a WSCProblem returns a (1, 1) vector
#     which has value (1, -1). The value represents
#     a prediction for the answer based on narrative chains.
#     """
#
#     def process(self, problem, debug=False):
#         pronoun = 'MASK_PRONOUN'
#         mask = re.compile('_')
#         sentence = mask.sub(pronoun, problem.sentence)
#         # tokens = model(sentence)
#         c1 = problem.candidate_1
#         c2 = problem.candidate_2
#         c1_events = []
#         c1_role = ''
#         c2_events = []
#         c2_role = ''
#         pronoun_events = []
#         pronoun_role = ''
#         if debug:
#             html = displacy.render(tokens, style='dep')
#             with open('parse.html', 'w') as f:
#                 f.write(html)
#         for token in tokens.noun_chunks:
#             if token.text == c1:
#                 c1_events = [token.root.head.lemma_]
#                 c1_role = token.root.dep_
#             elif token.text == c2:
#                 c2_events = [token.root.head.lemma_]
#                 c2_role = token.root.dep_
#             elif token.text == 'MASK_PRONOUN':
#                 pronoun_events = [token.root.head.lemma_]
#                 pronoun_role = token.root.dep_
#
#         for token in tokens:
#             if token.dep_ == 'xcomp':
#                 if token.head.lemma_ in pronoun_events:
#                     pronoun_events.append(token.lemma_)
#                 elif token.head.lemma_ in c1_events:
#                     c1_events.append(token.lemma_)
#                 elif token.head.lemma_ in c2_events:
#                     c2_events.append(token.lemma_)
#
#         # we've got the events and roles. now form the event tuples
#         event_tuples = []
#         if debug:
#             print(f'Role of C1: {c1_role}')
#             print(f'Role of C2: {c2_role}')
#             print(f'Role of Pronoun: {pronoun_role}')
#             print(f'Pronoun events: {str(pronoun_events)}')
#             print(f'C1 events: {str(c1_events)}')
#             print(f'C2 events: {str(c2_events)}')
#
#         def role_to_string(role):
#             if role == 'nsubj':
#                 return 's'
#             return 'o'
#         for p_event in pronoun_events:
#             p = p_event + '-' + role_to_string(pronoun_role)
#             for event in set(c1_events + c2_events):
#                 event_tuples.append((event, p))
#         if debug:
#             print(event_tuples)
#         role = self.get_role(event_tuples)
#         if role is None:
#             return 0  # Undecided
#         if role == role_to_string(c1_role):
#             return 1
#         if role == role_to_string(c2_role):
#             return -1
#         return 0  # Undecided (failed to identify role for c1 or c2)
