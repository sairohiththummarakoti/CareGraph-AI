# RAG vocabulary and the CareGraph example

A five-minute guide to connecting the words used in a retrieval system. Read the example first, then use the tables as a reference.

**The main idea:** prepare small pieces of reference text for searching; find the pieces relevant to a question; use their text to construct an answer.

## One example from beginning to end

Imagine a reference document with several sections about heart health. Someone asks, “What is a heart attack?”

**Before the question arrives:**

1. Split the reference document into smaller pieces of text. Each piece is a **chunk**, also called a **passage** here.
2. Give each passage to an **embedding model**. It produces a list of numbers representing aspects of the passage's meaning. This list is an **embedding vector**.
3. Put the vectors into a searchable **index**. Keep the original passage text and its source information linked to the corresponding index entry.
4. Save the index and the text information so they can be loaded later.

**When the question arrives:**

1. Use the same embedding model to create a vector for the question.
2. Compare that question vector with the saved passage vectors. This is **vector search**.
3. Rank the passages using **similarity scores**, apply the specialty filter and score cutoff, and keep up to the requested number of results.
4. Look up the original text belonging to each selected vector.
5. Use that text to construct the answer, with **citations** showing where it came from.

The saved vectors are not converted back into text. The original text was kept separately and is looked up through its connection to the matching index entry.

```text
PREPARE THE REFERENCES
Document → passages → embedding model → vectors → saved search index
                └──────────────────────────────→ saved text and source details

PROCESS A QUESTION
Question → embedding model → question vector
                                      ↓
                      compare with saved passage vectors
                                      ↓
                         ranked matching entries
                                      ↓
                    look up passage text and sources
                                      ↓
                         construct a cited answer
```

## Words for the text and its numerical representation

| Term | Plain meaning | Connection to the example |
|---|---|---|
| Document | A complete source file. | The heart-health reference file. |
| Chunk or passage | A small section of a document. | Several sentences about a heart-health topic. |
| Chunking | The process of splitting text into chunks. | Preparing smaller pieces before embedding them. |
| Token | A small unit of text used by a model: a word, part of a word, or punctuation. | One passage contains many tokens. Tokens are not passages. |
| Overlap | Text repeated between neighboring chunks. | The end of one chunk also appears at the beginning of the next, preserving some context across the boundary. |
| Embedding model | A trained model that represents text as numbers. | It can give related expressions, such as “heart attack” and “myocardial infarction,” similar representations. |
| Embedding or vector | The resulting list of numbers. In this discussion, both terms refer to that numerical representation. | CareGraph's model produces 384 numbers per passage or ordinary query. |
| Embedding process | Running text through the embedding model. | This creates vectors; it does not by itself store or search them. |

**Chunk size is not document size.** For example, 500-token chunks with 100-token overlap would cover tokens 1–500, then 401–900, then 801–1300 within one document.

CareGraph's reference chunker instead uses **up to 110 words with 20-word overlap**, within each reference section. The embedding model then tokenizes the text internally. Words and tokens are different units.

## Words for storage and search

| Term | Plain meaning | In CareGraph |
|---|---|---|
| Index | A structure organized for searching. | A FAISS index containing reference vectors. |
| Indexing | Preparing and adding data to that searchable structure. | The explicit ingestion step chunks references, embeds them, and builds the index. |
| FAISS | A library for searching numerical vectors. | It runs inside our Python application; we save its index to a local file. |
| Vector database | A database that supports storing and querying vectors, often with additional data-management features. | We have not set up a separate vector database server. We use FAISS plus local files. |
| Metadata | Information kept alongside a passage. | Its domain, filename, section, version, and chunk ID. |
| Chunk ID or text ID | An identifier used to distinguish a text passage. | Results include a chunk ID for citations. FAISS returns index positions, which our code maps to the passage records. |
| Vector search | Comparing a query vector with stored vectors. | It ranks numerical representations of passages. |
| Semantic search | Searching by meaning rather than requiring identical words. | Our text embeddings make vector similarity useful for finding related language. |
| Nearest neighbors or closest matches | The entries most similar under the chosen comparison measure. | Whole vectors are compared, not one matching number within each vector. |
| Similarity score | A measure calculated for the query–passage pair. | Higher scores rank higher in our implementation. A score is not a correctness percentage. |
| Threshold or cutoff | A minimum score a match must meet. | Our default cutoff is 0.25. It filters weak matches but does not guarantee relevance or truth. |
| Top K | The maximum number of highest-ranked qualifying results to return. | We request three per selected specialty, so both specialties can return up to six total. |
| Retrieval | Finding and returning the relevant source material. | The output includes actual passage text, source details, and scores—not only IDs. |

The developer chooses top K and the cutoff, then evaluates whether those choices return useful evidence. A search can return fewer than K results when too few qualify.

## Words for the answer

**RAG** means **retrieval-augmented generation**. A typical RAG application retrieves reference passages and supplies them, along with the question, to a generative model that writes an answer.

CareGraph implements the retrieval pipeline, but its current answer step uses **extractive synthesis**: it assembles exact retrieved excerpts and citations. It does not call an external generative LLM to write new prose.

**A citation** is the source reference displayed with a passage. It lets the reader trace the excerpt to its filename, section, version, and chunk ID.

**Synthetic data** means fictional demonstration data. **Semantic search** means search by meaning. These describe different things: CareGraph performs semantic search over synthetic reference data.

## What is actually saved in this project

| Local file | Contents |
|---|---|
| `indexes/references.faiss` | The reference vectors in a FAISS search index. |
| `indexes/metadata.json` | Original passage text, source information, and index-validation details. |

The model's downloaded weights are cached separately. They are the model itself, not the reference index.

Ordinary question answering reads the existing reference index. The question vector is temporary. Uploaded reports are not added to that index; confirmed report facts can help form the search query.

**LangGraph** coordinates the workflow around these operations. Nodes perform jobs, edges schedule the next jobs, and shared state carries data through one run. The FAISS reference index is separate from that per-question state.

## Where to look in the code

| Question | Starting point |
|---|---|
| Where does text become vectors? | [`SentenceTransformerEmbedder.encode()`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/retrieval.py#L94) |
| Where are passages created? | [`_chunks()`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/retrieval.py#L194) |
| Where is the library built and saved? | [`build_index()`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/retrieval.py#L320) |
| Where is the saved library loaded? | [`ReferenceIndex`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/retrieval.py#L375) |
| Where does similarity search happen? | [`search()`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/retrieval.py#L486) |
| Where do we choose three results per domain? | [The graph's retrieval node](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/graph.py#L71) |
| Where is the cited answer assembled? | [`synthesis.py`](https://github.com/sairohiththummarakoti/CareGraph-AI/blob/master/caregraph/synthesis.py) |

The application uses a small fictional educational corpus. Its score ranks similarity; its citations identify sources. Neither establishes clinical correctness.
