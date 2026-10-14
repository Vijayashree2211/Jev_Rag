# JEV-Powered RAG for FastAPI Documentation

A retrieval-augmented generation (RAG) project that answers questions
about the official FastAPI documentation. It combines Neon PostgreSQL
with `pgvector`, dense vector retrieval, BM25-style lexical retrieval,
Reciprocal Rank Fusion (RRF), JEV-based relevance judging, Qwen answer
generation through OpenRouter, and JEV answer evaluation.

## Project Overview

The system retrieves documentation chunks relevant to a user's question,
supplies the retrieved evidence to a language model, and evaluates the
generated answer against that evidence.

The project has two distinct uses of JEV:

1.  **Retrieval relevance judging:** JEV scores question--chunk pairs to
    help label retrieved chunks as relevant or not relevant for
    retrieval evaluation.
2.  **Answer evaluation:** JEV evaluates the generated answer for answer
    relevance, groundedness, and citation support.

JEV is used through its Decisions API. The project uses JEV's returned
scores; it does not train JEV or implement the RLCD training algorithm.

## Architecture and Workflow

``` text
Official FastAPI documentation
            |
            v
     Chunk documents
            |
            v
 Generate 384-dimensional embeddings
            |
            v
 Store chunks and embeddings in Neon PostgreSQL
                  + pgvector
            |
            v
        Retrieval experiments
       /         |          \
      v          v           v
   Dense        BM25       Hybrid RRF
       \         |          /
        \        |         /
         v       v        v
           Retrieval audit
                  |
                  v
       JEV relevance scores per
         question–chunk pair
                  |
                  v
     Binary relevance labels (0/1)
                  |
                  v
 Recall@5, MRR@5, and nDCG@5
                  |
                  v
        Dense retrieval baseline
                  |
                  v
       Qwen answer generation
                  |
                  v
        JEV answer evaluation
```

The retrieval audit and retrieval metrics are an evaluation workflow.
The answer-generation workflow uses Dense retrieval as the current
baseline.

## 1. Document Ingestion and Chunking

The source material consists of official FastAPI documentation pages.
Each document is divided into smaller chunks so that retrieval can
return focused passages rather than entire pages.

Each chunk stores fields such as:

-   `document_id`
-   `title`
-   `section`
-   `source_url`
-   `content`
-   `content_hash`
-   `metadata`
-   `embedding`

The current embeddings use `BAAI/bge-small-en-v1.5`, producing
**384-dimensional vectors**.

Chunking affects retrieval quality: chunks that are too large may
contain unrelated information, while chunks that are too small may omit
context needed to answer a question.

## 2. PostgreSQL and pgvector

The project uses **Neon PostgreSQL** as its database and the `pgvector`
extension to store and search embeddings.

The `document_chunks` table stores the documentation text, metadata, and
vector embedding. A vector index supports efficient similarity search.
The table also contains a generated search field and a GIN index for
lexical search.

`pgvector` is the vector-storage and similarity-search component; it
does not generate answers. It makes the stored embeddings searchable so
the retriever can find chunks semantically similar to a question.

## 3. Retrieval Methods

The project compares three retrieval approaches.

### 3.1 Dense Retrieval

**Function:** `dense_search(query, k=5)`

Dense retrieval embeds the query into a vector and compares it with the
stored chunk embeddings. It returns the top `k` chunks by vector
similarity.

Dense retrieval is useful when the question and documentation use
different wording but express similar meaning. For example, a question
about a "URL variable" may retrieve documentation that calls it a "path
parameter."

### 3.2 BM25 / Lexical Retrieval

**Function:** `bm25_search(query, k=5)`

BM25-style lexical retrieval ranks chunks using query-term matches. It
can work well when the question contains exact technical terms,
identifiers, or phrases found in the documentation.

Dense retrieval and lexical retrieval have different strengths: dense
retrieval emphasizes semantic similarity, while lexical retrieval
emphasizes word-level matching.

### 3.3 Reciprocal Rank Fusion (RRF)

**Function:**
`reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=5)`

RRF combines the ranked results from Dense retrieval and BM25. Instead
of directly comparing their raw scores, it uses each chunk's rank in the
result lists to calculate a fused ranking.

The current workflow retrieves a larger candidate list from both
methods, applies RRF, and returns the top five fused results. RRF can
combine complementary results, but it is not guaranteed to outperform
Dense retrieval on every dataset.

## 4. Retrieval Audit

The audit compares the results returned by Dense, BM25, and Hybrid RRF
for each question in the golden dataset.

For each question, the audit records the ranked chunks returned by each
approach. Duplicate question--chunk pairs are consolidated into a review
file so the same chunk can be judged once for that question, even if
multiple retrieval methods returned it.

The review file, `retrieval_relevance_review.csv`, includes fields such
as:

-   Question ID and question text
-   Chunk ID, title, and section
-   A content preview
-   JEV relevance score
-   Binary relevance label

The audit makes retrieval behavior inspectable: it shows which chunks
each method retrieved and provides a basis for relevance labeling and
metric calculation.

## 5. JEV-Based Chunk Relevance Labeling

JEV is used to judge whether a retrieved chunk provides useful evidence
for answering a specific question.

For each question--chunk pair, the Decisions API receives the question
and chunk content and returns a `noul` score. In this project, that
score is treated as JEV's estimated probability that the answer to the
relevance decision is "yes."

The workflow then converts scores into binary labels:

-   `1` --- relevant to the question
-   `0` --- not relevant to the question

The current experiment used a **provisional threshold of 0.70**: scores
at or above the threshold were labeled relevant; lower scores were
labeled not relevant. This threshold is an experimental choice, not a
universal JEV threshold.

Labels should be manually audited against the actual chunk content,
especially near the threshold. JEV-generated labels are not independent
ground truth, so retrieval metrics calculated from them measure
performance against the judge's labels.

## 6. Retrieval Evaluation Metrics

The project calculates retrieval metrics at a cutoff of five results.

### Recall@5

Measures the fraction of judged-relevant chunks for a question that
appear among the top five retrieved chunks.

A higher Recall@5 means the retriever finds more of the chunks labeled
relevant in the judged candidate pool.

### MRR@5 --- Mean Reciprocal Rank

Measures how highly the first relevant chunk appears in the top five. If
the first relevant chunk is ranked first, the reciprocal rank is `1`; if
it is ranked second, it is `1/2`.

### nDCG@5 --- Normalized Discounted Cumulative Gain

Measures the quality of the ranking, giving more credit when relevant
chunks appear nearer the top.

### Current Baseline Results

The initial experiment used eight golden questions. With the current JEV
labels and evaluation implementation, the reported average results were:

  Retrieval approach     Recall@5   MRR@5   nDCG@5
  -------------------- ---------- ------- --------
  Dense                     0.842   1.000    0.897
  Hybrid RRF                0.751   0.906    0.805
  BM25                      0.584   0.812    0.631

**Dense retrieval is the current baseline winner for this eight-question
experiment.** These results should not be generalized beyond the current
dataset. They depend on the relevance labels and provisional JEV
threshold, and should be re-evaluated if the corpus, questions, or
labeling policy changes.

## 7. Qwen Answer Generation

**Function:** `generate_rag_answer(question, k=5)`

The answer-generation stage uses Dense retrieval to obtain the top five
documentation chunks. It formats the retrieved content as evidence,
includes source identifiers, and sends the question and context to Qwen
through OpenRouter.

The configured model is:

`qwen/qwen3.8-2.4t-a95b`

The generation prompt instructs the model to:

-   Answer the exact question asked.
-   Use the supplied FastAPI documentation as evidence.
-   Avoid inventing unsupported facts.
-   State when the supplied context is insufficient.
-   Cite evidence using source labels such as `[Source 1]`.

The function returns the question, generated answer, source metadata,
model identifier, and token-usage/cost information when provided by the
API.

The source labels are generated from the retrieved context. They should
be checked to ensure that each citation actually supports the claim it
accompanies.

## 8. JEV Answer Evaluation

After Qwen generates an answer, JEV evaluates the answer against the
question and retrieved evidence. This is a separate task from judging
individual chunks during the retrieval audit.

The current evaluation checks three criteria:

### Answer Relevance

Does the generated answer address the user's actual question?

### Groundedness

Are the factual claims in the answer supported by the retrieved
documentation?

### Citation Support

Do the cited sources support the claims associated with those citations?

JEV returns a `noul` score for each criterion. These values are judge
estimates, not independently verified accuracy percentages. For example,
a score of `0.94` should not automatically be interpreted as a
demonstrated 94% accuracy rate.

The initial path-parameter test returned:

  Criterion            JEV score
  ------------------ -----------
  Answer relevance          0.96
  Groundedness              0.94
  Citation support          0.86

These are results from one test example, not an overall evaluation of
the eight-question dataset.

## 9. Configuration and Running the Notebook

The experiments are developed in Google Colab. The notebook uses
environment variables for credentials:

-   `OPENROUTER_API_KEY` --- OpenRouter access for Qwen and JEV
-   `DATABASE_URL` --- Neon PostgreSQL connection string

Configure secrets in Colab Secrets or the runtime environment. Do not
commit API keys, database URLs, or other credentials to GitHub. Remove
sensitive notebook outputs before publishing the notebook.

Run the notebook sections in order: document setup and ingestion,
retrieval functions, retrieval audit, JEV relevance labeling, retrieval
metrics, Qwen answer generation, and JEV answer evaluation.

## Limitations and Next Steps

-   The current golden dataset contains eight questions, so it is a
    small evaluation set.
-   JEV-generated relevance labels use a provisional threshold and
    require manual validation.
-   The reported retrieval metrics evaluate agreement with the current
    labels, not independently verified relevance.
-   A high JEV answer score is not proof of correctness; answer-level
    evaluation should be checked against human-reviewed examples.
-   The current project uses JEV for relevance judging and answer
    evaluation. It does **not** train JEV or implement the RLCD training
    algorithm.
-   Dense retrieval is the current baseline winner for the initial
    experiment. Larger datasets and additional questions may change the
    ranking of retrieval approaches.

## Technology Stack

-   **Language:** Python
-   **Notebook environment:** Google Colab
-   **Database:** Neon PostgreSQL
-   **Vector search:** `pgvector`
-   **Embeddings:** `BAAI/bge-small-en-v1.5`
-   **Retrieval:** Dense search, BM25-style lexical search, Reciprocal
    Rank Fusion
-   **Decision/judging model:** TypeSafe JEV Decisions API
    (`typesafe/jev-1.13`)
-   **Answer-generation model:** Qwen through OpenRouter
    (`qwen/qwen3.8-2.4t-a95b`)
-   **Evaluation:** Recall@5, MRR@5, nDCG@5, and JEV answer-level scores
