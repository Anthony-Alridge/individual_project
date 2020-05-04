# from textacy.extract import subject_verb_object_triples
# import spacy
# from textacy.spacier import utils as spacy_utils
#
# nlp = spacy.load('en_core_web_md')
# sentence = 'The delivery truck zoomed by the school bus because it is going so fast.'
# sentence_2 = 'The man could not lift his son because he was so weak.'
# for triple in spacy_utils.get_main_verbs_of_sent(nlp(sentence)):
#     for sub in spacy_utils.get_subjects_of_verb(triple):
#         print(sub)
#     print(triple)
#
# for triple in subject_verb_object_triples(nlp(sentence_2)):
#     print('hello2')
#     print(triple)
'''{
  "@context": [
    "http://api.conceptnet.io/ld/conceptnet5.7/context.ld.json"
  ],
  "@id": "/query?rel=/r/Antonym&start=/c/en/weakness",
  "edges": [
    {
      "@id": "/a/[/r/Antonym/,/c/en/weakness/n/,/c/en/strength/]",
      "@type": "Edge",
      "dataset": "/d/wiktionary/en",
      "end": {
        "@id": "/c/en/strength",
        "@type": "Node",
        "label": "strength",
        "language": "en",
        "term": "/c/en/strength"
      },
      "license": "cc:by-sa/4.0",
      "rel": {
        "@id": "/r/Antonym",
        "@type": "Relation",
        "label": "Antonym"
      },
      "sources": [
        {
          "@id": "/and/[/s/process/wikiparsec/2/,/s/resource/wiktionary/en/]",
          "@type": "Source",
          "contributor": "/s/resource/wiktionary/en",
          "process": "/s/process/wikiparsec/2"
        }
      ],
      "start": {
        "@id": "/c/en/weakness/n",
        "@type": "Node",
        "label": "weakness",
        "language": "en",
        "sense_label": "n",
        "term": "/c/en/weakness"
      },
      "surfaceText": null,
      "weight": 1.0
    },
    {
      "@id": "/a/[/r/Antonym/,/c/en/weakness/,/c/en/strength/]",
      "@type": "Edge",
      "dataset": "/d/verbosity",
      "end": {
        "@id": "/c/en/strength",
        "@type": "Node",
        "label": "strength",
        "language": "en",
        "term": "/c/en/strength"
      },
      "license": "cc:by/4.0",
      "rel": {
        "@id": "/r/Antonym",
        "@type": "Relation",
        "label": "Antonym"
      },
      "sources": [
        {
          "@id": "/s/resource/verbosity",
          "@type": "Source",
          "contributor": "/s/resource/verbosity"
        }
      ],
      "start": {
        "@id": "/c/en/weakness",
        "@type": "Node",
        "label": "weakness",
        "language": "en",
        "term": "/c/en/weakness"
      },
      "surfaceText": "[[weakness]] is the opposite of [[strength]]",
      "weight": 0.41500000000000004
    }
  ]
}'''.json()
