# Front Line PHP RAG

Question answering over one book, *Front Line PHP*. The system turns the PDF into typed units,
retrieves them, and answers with printed page citations or the exact text `No information found`.

## Language

### The source document

**PDF page**:
The 1-based index of a page in the PDF file. Pages 1-324.
_Avoid_: page index, sheet

**Printed page**:
The number printed on a page of the physical book. Always PDF page minus 2. Citations use this.
_Avoid_: book page, page

**Running header**:
The 8 pt line at the top of every body page that repeats the printed page number and the book or
chapter title. Deleted before any other processing.
_Avoid_: page header, masthead

**Part title page**:
A page that holds only the title of one of the three parts (PDF pages 12, 156, 232). Skipped.
_Avoid_: divider, section page

**Front matter**:
The Foreword and the Preface. Both have chapter number 0 and distinct chapter titles.
_Avoid_: preamble, intro

### Structure

**Chapter**:
One bookmark entry of the book that holds body text: Foreword, Preface, the 30 numbered chapters,
and In Closing. 33 in total. Numbered chapters are 1-30 from bookmark order, never from the
printed `CHAPTER NN` label. Foreword and Preface are 0. In Closing is 31.
_Avoid_: section, part

**Part**:
One of the three top-level bookmark groups: Part 1 "PHP, the Language", Part 2 "Building With
PHP", Part 3 "PHP In Depth". Front matter belongs to no part. In Closing belongs to Part 3.
_Avoid_: book, volume

**Manifest**:
The record of one completed ingestion: document hash, chunking hash, index version, counts, model
names, and the verification result. One per index version. The "already indexed" check reads it.
_Avoid_: index metadata, run record

**Chapter table**:
The ordered list of the 33 chapters with number, title, part, and PDF page range. Built once from
the bookmarks in Stage 1 and read by every later stage.
_Avoid_: table of contents, outline, TOC

### Parsing units

**Word**:
One token from the PDF text layer with its font name, size, and position on the page. The font
name has no subset prefix.
_Avoid_: token, glyph run

**Parsed page**:
One PDF page after header removal: its words plus the count of header words deleted.
_Avoid_: page record
