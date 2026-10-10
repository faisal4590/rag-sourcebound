# Chunk quality report, 2026-10-10

Issue #62. Chunk set `chunk_config_hash d4ebab9ed10ca7e7`, git SHA `63389e1`, 202 chunks, embed model `BAAI/bge-m3`. "Before fixes" is the 207-chunk set `e9ca297723b994f1`, rebuilt into `data/_before_e9ca2977` without the page-break paragraph merge and the overlap fix (changes 11 and 12); the 279-chunk set that opened the issue appears where its numbers were recorded.

## Verdict

**Freeze at chunk_config_hash `d4ebab9e`** (`d4ebab9ed10ca7e7`, git SHA `63389e1`). Levels 1 and 2 pass; dense retrieval over the structural chunks meets all three Level 3 targets (Recall@5 1.00, Recall@20 1.00, MRR@10 0.86); the flat 400/50 baseline wins on no metric.

The rank differences between structural and baseline are not significant on this question set: sign test p = 0.77 for dense (7 better, 18 equal, 5 worse) and p = 0.45 for BM25 (5 better, 23 equal, 2 worse). What structural chunking was designed for shows in answer completeness: 375 of 375 code blocks sit whole inside one chunk, against 369 of 375 for the baseline, which splits 6. None of those 6 blocks is one the golden questions quote, so at question level every system holds the whole evidence block in its top 5 for 9 of 9 code questions.

## Level 1: structural gate

| Check | Before fixes (207) | After (202) | Target | |
|---|---|---|---|---|
| size | 0 under 120, 0 over 900 (min 124, max 742) | 0 under 120, 0 over 900 (min 124, max 685) | 0 under 120, 0 over 900 | pass |
| header_noise | 31 header patterns, 0 chunks with a header line, 0 header words left in data/parsed | 31 header patterns, 0 chunks with a header line, 0 header words left in data/parsed | 0 lines, 0 words | pass |
| glyph_bug | counts {'=_construct': 0, '$this=>': 0, '=[': 0} | counts {'=_construct': 0, '$this=>': 0, '=[': 0} | 0 | pass |
| sections | 0 without section_title, 0 without section_titles, 0 with an unlisted heading, 0 spanning two chapters | 0 without section_title, 0 without section_titles, 0 with an unlisted heading, 0 spanning two chapters | 0 | pass |
| code_integrity | 1 chunks with unbalanced braces (review list) | 3 chunks with unbalanced braces (review list) | review | pass |
| coverage | 1.0000 of 62332 clean tokens covered, 0 blocks uncovered, 0 chapters under 0.99 | 1.0000 of 62324 clean tokens covered, 0 blocks uncovered, 0 chapters under 0.99 | 0.99 | pass |
| chapters | 33 of 33 chapters have a chunk | 33 of 33 chapters have a chunk | 33 | pass |

Brace mismatches reviewed: `20:0128` InvoiceDTO, partial by design (the PDF clips the line on printed page 219). After the fixes also `22:0145` and `22:0146`: the Preloader class is printed over PDF pages 247, 249, 251 with blank pages between, held as three code blocks; the merged lead-in paragraph pushed the third block into its own chunk.

## Level 2: content checks

| Check | 279-chunk set | Before fixes (207) | After (202) | Target | |
|---|---|---|---|---|---|
| Near-duplicate pairs, dense cosine > 0.95 | 0 (highest 0.901) | 0 (highest 0.903) | 0 (highest 0.903) | 0 | pass |
| Nearest neighbor in the same chapter | 0.742 | 0.841 | 0.822 | keep | |
| Chunks under 10 sparse terms | 4 (one with 0) | 0 (lowest 26) | 0 (lowest 26) | 0 | pass |
| Human read, 20 chunks seed 42: one sentence | - | 20 | 20 | 18 | pass |
| Human read: clean start and end | - | 20 | 20 | 18 | pass |
| Human read: code complete | - | 20 | 20 | 18 | pass |
| Human read: one sub-topic | - | 19 | 19 (18 counting `03:0028`, three merged sections) | 18 | pass |

The reviewer's second-round answers are in `eval/chunk_review.md`; the first round covered the 207-chunk set. In the first round the reviewer's `yes` on `16:0108` (started mid-sentence) and `08:0062`, `22:0151` (lead-in without its code) would have been `no` by the strict reading; those three defects are what changes 11 and 12 removed.

## Level 3: retrieval

30 questions, one per chapter 01-30: 8 definitions, 8 code requests, 7 how-to, 7 comparisons. Hit = normalized chunk text contains a normalized evidence quote. Baseline: RecursiveCharacterTextSplitter.from_tiktoken_encoder (cl100k_base, 400 tokens, 50 overlap), 185 chunks, embedded from the raw text with no header. "No header" embeds the structural chunks from `display_text` alone.

### Overall

| System | Set | Recall@5 | Recall@20 | MRR@10 | Mean rank of first hit | Evidence share | No hit |
|---|---|---|---|---|---|---|---|
| structural, dense | before fixes | 1.00 | 1.00 | 0.89 | 1.37 | 0.06 | 0 |
| structural, dense | after | 1.00 | 1.00 | 0.86 | 1.43 | 0.06 | 0 |
| structural, dense, no header | after | 1.00 | 1.00 | 0.83 | 1.43 | 0.06 | 0 |
| structural, BM25 | before fixes | 1.00 | 1.00 | 0.88 | 1.33 | 0.06 | 0 |
| structural, BM25 | after | 1.00 | 1.00 | 0.90 | 1.30 | 0.06 | 0 |
| baseline, dense | - | 0.97 | 1.00 | 0.82 | 1.67 | 0.05 | 0 |
| baseline, BM25 | - | 0.97 | 0.97 | 0.84 | 2.47 | 0.05 | 0 |

### Paired comparison per question (rank of the first hit)

| Pair | Before fixes | p | After | p |
|---|---|---|---|---|
| structural vs baseline, dense | 7 better, 19 equal, 4 worse | 0.55 | 7 better, 18 equal, 5 worse | 0.77 |
| structural vs baseline, BM25 | 4 better, 24 equal, 2 worse | 0.69 | 5 better, 23 equal, 2 worse | 0.45 |
| structural with header vs no header, dense | - | - | 3 better, 25 equal, 2 worse | 1.00 |

p is the two-sided exact sign test with ties dropped. One question is 3.3 points of any rate on 30 questions. The MRR@10 and mean-rank differences above rest on the counts in this table, not on the decimals, and none of them is significant.

### Answer completeness

Code blocks are what structural chunking protects: a block split across chunks cannot be quoted verbatim by the generator. Corpus level: clean code blocks that sit whole inside at least one chunk; blocks over `chunk.code_max_tokens` are split into a code group by design and counted apart. Question level: for the code questions whose evidence is a code line, the number whose top-5 holds the whole code block of that line.

| Measure | Before fixes (207) | After (202) | Baseline (185) |
|---|---|---|---|
| Code blocks in `data/clean` | 375 | 375 | 375 |
| Whole inside one chunk | 375 | 375 | 369 |
| Split into a code group by design (over 900 tokens) | 0 | 0 | 0 |
| Split otherwise | 0 | 0 | 6 |
| Code questions, whole block in top 5, dense | 9 of 9 | 9 of 9 | 9 of 9 |
| Code questions, whole block in top 5, BM25 | 9 of 9 | 9 of 9 | 9 of 9 |
| Code questions, whole block in top 5, dense, no header | - | 9 of 9 | - |

The code questions counted are those whose evidence quote sits inside a clean code block (the others quote prose about code).

### Per kind

| System | Set | Kind | n | Recall@5 | Recall@20 | MRR@10 | Evidence share |
|---|---|---|---|---|---|---|---|
| structural, dense | before fixes | code | 8 | 1.00 | 1.00 | 0.94 | 0.02 |
| structural, dense | before fixes | comparison | 7 | 1.00 | 1.00 | 0.72 | 0.08 |
| structural, dense | before fixes | definition | 8 | 1.00 | 1.00 | 1.00 | 0.07 |
| structural, dense | before fixes | how-to | 7 | 1.00 | 1.00 | 0.89 | 0.08 |
| structural, dense | after | code | 8 | 1.00 | 1.00 | 0.94 | 0.02 |
| structural, dense | after | comparison | 7 | 1.00 | 1.00 | 0.65 | 0.08 |
| structural, dense | after | definition | 8 | 1.00 | 1.00 | 0.94 | 0.06 |
| structural, dense | after | how-to | 7 | 1.00 | 1.00 | 0.89 | 0.07 |
| structural, dense, no header | after | code | 8 | 1.00 | 1.00 | 0.88 | 0.02 |
| structural, dense, no header | after | comparison | 7 | 1.00 | 1.00 | 0.65 | 0.08 |
| structural, dense, no header | after | definition | 8 | 1.00 | 1.00 | 0.85 | 0.06 |
| structural, dense, no header | after | how-to | 7 | 1.00 | 1.00 | 0.93 | 0.07 |
| structural, BM25 | before fixes | code | 8 | 1.00 | 1.00 | 0.94 | 0.02 |
| structural, BM25 | before fixes | comparison | 7 | 1.00 | 1.00 | 0.90 | 0.08 |
| structural, BM25 | before fixes | definition | 8 | 1.00 | 1.00 | 0.88 | 0.07 |
| structural, BM25 | before fixes | how-to | 7 | 1.00 | 1.00 | 0.81 | 0.08 |
| structural, BM25 | after | code | 8 | 1.00 | 1.00 | 0.94 | 0.02 |
| structural, BM25 | after | comparison | 7 | 1.00 | 1.00 | 0.90 | 0.08 |
| structural, BM25 | after | definition | 8 | 1.00 | 1.00 | 0.88 | 0.06 |
| structural, BM25 | after | how-to | 7 | 1.00 | 1.00 | 0.89 | 0.07 |
| baseline, dense | - | code | 8 | 1.00 | 1.00 | 0.85 | 0.02 |
| baseline, dense | - | comparison | 7 | 1.00 | 1.00 | 0.86 | 0.08 |
| baseline, dense | - | definition | 8 | 1.00 | 1.00 | 0.76 | 0.06 |
| baseline, dense | - | how-to | 7 | 0.86 | 1.00 | 0.80 | 0.06 |
| baseline, BM25 | - | code | 8 | 1.00 | 1.00 | 0.84 | 0.02 |
| baseline, BM25 | - | comparison | 7 | 1.00 | 1.00 | 0.90 | 0.08 |
| baseline, BM25 | - | definition | 8 | 1.00 | 1.00 | 0.76 | 0.06 |
| baseline, BM25 | - | how-to | 7 | 0.86 | 0.86 | 0.86 | 0.06 |

### Acceptance

- Recall@20 >= 0.90: 1.00, pass.
- Recall@5 >= 0.70: 1.00, pass.
- MRR@10 >= 0.55: 0.86, pass.
- Structural beats the baseline: not measurable on recall. Recall@5 is 1.00 for structural and 0.97 for the baseline, so the question set saturates recall. The relative result is MRR@10 (0.86 against 0.82 dense, 0.90 against 0.84 BM25), mean rank of the first hit (1.43 against 1.67 dense, 1.30 against 2.47 BM25), and the paired counts (7 better, 18 equal, 5 worse dense; 5 better, 23 equal, 2 worse BM25).

## Failed questions

No question missed the top 5 with dense retrieval over the structural chunks. The table lists every question whose first hit was not ranked first (the MRR@10 losses) after the fixes, with the rank before the fixes, then the baseline misses. Pages are printed pages of the top 3 chunks.

| System | Case | Kind | Question | Expected pages | Rank before | Rank after | Top 3 after (pages) | Cause |
|---|---|---|---|---|---|---|---|---|
| structural, dense | g010 | comparison | How do short closures differ from normal closures? | 127 | 2 | 2 | `10:0071` 129; `10:0069` 127-128; `10:0070` 128 | Both differences the question asks for are split over two chunks of the chapter: the `use` statement point sits in the neighbor section 'Values from Outer Scope', which ranks first. The quote is in the chapter intro, second. |
| structural, dense | g012 | how-to | How do you destructure an array with string keys into variables? | 140 | 4 | 4 | `12:0077` 141; `12:0075` 138-139; `12:0078` 142 | Three chunks of the 'Array Destructuring' section answer the question: 'In Practice' (parse_url, pathinfo keys) ranks first and is a correct answer; the quoted `['c' => $c, 'a' => $a]` example sits in the 'Skipping Elements' chunk, fourth. |
| structural, dense | g013 | comparison | How does type checking differ between match and switch? | 145 | 3 | 3 | `13:0079` 143-144; `13:0082` 148; `13:0080` 145-146 | The chapter intro (the switch and match comparison) and 'So, Switch or Match?' rank above the 'No Type Coercion' chunk that holds the strict-check sentence. All three compare match with switch. |
| structural, dense | g014 | code | Show a constructor that uses new in a parameter default value for a State object. | 150 | 2 | 2 | `14:0085` 152-153; `14:0083` 149-151; `14:0084` 151-152 | The 'Not in class properties' chunk ranks first and also shows `State $state = new InitialState()` in the transpiled constructor; it is a correct answer. The quoted line is in the chapter intro, second. |
| structural, dense | g015 | definition | What does composition over inheritance mean in the poem generator example? | 173 | 1 | 2 | `15:0094` 166-167; `15:0096` 171-173; `15:0093` 165-166 | The chunk that states the problem (RandomEchoPoemGenerator, three subclasses) ranks above the chunk with the composition conclusion. The whole section is about the poem generator, as the question says. |
| structural, dense | g024 | comparison | How was the #[] attribute syntax chosen over the alternatives? | 262 | 5 | 5 | `08:0053` 97-98; `24:0158` 261; `24:0157` 260 | The Chapter 8 callout 'What about the syntax?' tells the same story in two sentences and ranks first; the Chapter 24 RFC account with the quote is fifth. Two chapters answer the question, see term_elsewhere. |
| structural, dense | g025 | comparison | What is the difference between covariance and contravariance in PHP type variance? | 267 | 1 | 2 | `25:0163` 268-269; `25:0162` 266-268; `03:0023` 41-43 | The second half of the 'Type Variance' section (the contravariance example) ranks first; the covariance definition sits in the first half, second. The question names both terms. |
| baseline, dense | g020 | how-to | Where should the opening curly bracket go in a multi-line method signature, and why? | 224 | - | 9 | `base:0120`; `base:0063`; `base:0117` | Baseline only (rank 9 dense, 33 BM25; structural rank 1). The flat splitter cut 'On a new line!' away from the X-visualisation paragraphs that give it meaning; the section-aware chunk keeps them together. |
| baseline, BM25 | g020 | how-to | Where should the opening curly bracket go in a multi-line method signature, and why? | 224 | - | 33 | `base:0120`; `base:0063`; `base:0044` | Baseline only (rank 9 dense, 33 BM25; structural rank 1). The flat splitter cut 'On a new line!' away from the X-visualisation paragraphs that give it meaning; the section-aware chunk keeps them together. |

## Limits

- The 30 questions were drafted from the book's own sentences. Mean overlap between question content stems and evidence stems is 0.44 (`eval/golden_v1.draft.jsonl`, field `overlap`); four questions share every content word with their evidence. This favors lexical retrieval and saturates recall for both chunkers.
- 30 questions, one per chapter. One question moves any rate by 3.3 points, so differences under about 0.07 are within one question.
- Comparison questions have unlabeled alternative answers: in four of the seven MRR losses the top-ranked chunk answers the question but holds no evidence quote. The hit rule counts only the quoted sentence. `term_elsewhere` in the draft file lists the chapters that also answer d005, d023, and d024.
- The baseline has no chunk header. Part of the structural advantage on dense retrieval is the header, not the boundaries; the no-header row isolates it.
- Evidence share compares one quoted sentence or code line with a 300-token chunk. It reads as quote length here, not as chunk size.
- Follow-up: Milestone 6, 100 golden cases (60 answerable, 20 in-domain unanswerable, 20 out-of-domain), with at least a third of the answerable questions under overlap 0.3 and answers that span a section.

## Changes during this issue

From `docs/spec-changes-pending.md`. Before and after on the shipped PDF.

| # | Change | Before | After |
|---|---|---|---|
| 1 | Glyph tables in Stage 1 (issue #48, verified here) | pdfplumber alone: about 640 ligature glyphs read as `=` | 1,607 glyphs repaired; `=_construct`, `$this=>`, `=[` each 0; PyMuPDF oracle agrees on every body page |
| 2 | Inline code rule in Stage 2 | 31 code blocks off a rectangle, 17 code blocks with 3 spurious leading spaces | 1 (`opcache.preload`), 0 |
| 3 | Word gap join in Stage 2 | 127 paragraphs with `word .` gaps | 11, all real text |
| 4 | Small-section merge in Stage 4 | 52 chunks were sections under 120 tokens | 0 |
| 5 | Small-callout merge in Stage 4 | 26 callout chunks under 120 tokens, one with 0 sparse terms | 0; lowest sparse term count 26 |
| 6 | `chunk.small_merge_max_tokens: 900` | 1 chunk under 120 tokens left (42-token callout beside a 569-token chunk) | 0; three chunks between 600 and 685 tokens |
| 7 | `chunk_config_hash` scope | config `chunk` section only; code changes kept the hash | parse, structure, clean, chunk sections plus Stages 1-5 source; hash changed on every rule change (a8ea0fc5 -> 06a108b2 -> e9ca2977 -> d4ebab9e) |
| 8 | Per-chapter coverage in Stage 8 | chunks per page flagged Chapters 5 and 11 while fully covered | every chapter at 1.0 coverage, no exceptions |
| 9 | Header noise rule | literal `Front Line PHP` failed on a Foreword sentence | 31 running-header patterns, 0 matches, 0 header words left in `data/parsed` |
| 10 | `section_title` always set, `section_titles` list | 73 chunks with an empty section title | 0; every merged heading listed |
| 11 | Paragraph merge across a page break | 43 paragraphs starting mid-sentence, 2 chunks starting mid-sentence | 1 (a sentence starting with `match`), 0 |
| 12 | No overlap after a chunk that ends in code | 44 chunks opening with a colon lead-in (28 with no code following) | 2, both followed by the quote they introduce |

Level 3 before and after changes 11 and 12 (structural, dense): Recall@5 1.00 -> 1.00, MRR@10 0.89 -> 0.86, mean rank 1.37 -> 1.43; BM25: MRR@10 0.88 -> 0.90. Chunk counts along the way: 279 -> 209 -> 207 -> 207 -> 202.
