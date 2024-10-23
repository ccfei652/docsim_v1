import spacy
from string import punctuation
from spacy.lang.en import stop_words

stop_words = stop_words.STOP_WORDS
punctuations = list(punctuation)

# initialize language model
nlp = spacy.load("en_core_web_lg")

# add pipeline (declared through entry_points in setup.py)
entity_linker = nlp.add_pipe("entityLinker", last=True)

# Example sentences
sentence_1 = "Software Agents: Completing Patterns and Constructing User Interfaces To support the goal of allowing users to record and retrieve information, this paper describes an interactive note-taking system for pen-based computers with two distinctive features. First, it actively predicts what the user is going to write. Second, it automatically constructs a custom, button-box user interface on request. The system is an example of a learning-apprentice software- agent. A machine learning component characterizes the syntax and semantics of the user's information. A performance system uses this learned information to generate completion strings and construct a user interface. Description of Online Appendix: People like to record information. Doing this on paper is initially efficient, but lacks flexibility. Recording information on a computer is less efficient but more powerful. In our new note taking softwre, the user records information directly on a computer. Behind the interface, an agent acts for the user. To help, it provides defaults and constructs a custom user interface. The demonstration is a QuickTime movie of the note taking agent in action. The file is a binhexed self-extracting archive. Macintosh utilities for binhex are available from mac.archive.umich.edu. QuickTime is available from ftp.apple.com in the dts/mac/sys.soft/quicktime."

# Process the sentences
doc1 = nlp(sentence_1)

# Exibir entidades reconhecidas
print("Entidades reconhecidas:")
for ent in doc1.ents:
    print(f"Entidade: {ent.text}, Label: {ent.label_}")

# Exibir entidades linkadas
print("\nEntidades linkadas:")
for entity in doc1._.linkedEntities:
    print(f"Alias: {entity.original_alias}, Linked Label: {entity.label}")

# Extract linked entities from both sentences
entities_1 = [entity.get_id() for entity in doc1._.linkedEntities]
entities_2 = [entity.get_id() for entity in doc2._.linkedEntities]

# Compare the entities to check if there's a match
common_entities = set(entities_1).intersection(entities_2)

if common_entities:
    print(f"Common entity found: {common_entities}")
else:
    print("No common entities found")

#
# doc_2 = nlp("American Airlines from Bill Gates and Elon Musk")
#
# doc = nlp("Elon Musk, Steve Jobs, D&G and versace floral dresser")
#
#
#
# # returns all entities in the whole document
# all_linked_entities = doc._.linkedEntities
# # iterates over sentences and prints linked entities
# for sent in doc.sents:
#     sent._.linkedEntities.pretty_print()
#
# # def preprocess(text):
# #     sentence = nlp(text)
# #     # lemmatization
# #     sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
# #     # removing stop words
# #     sentence = [
# #         word for word in sentence
# #         if word not in stop_words
# #            and word not in punctuations and word.isalpha()
# #     ]
# #
# #     return list(dict.fromkeys(sentence))
# #
# #
# # preprocess("um texto qualquer qualquer")
