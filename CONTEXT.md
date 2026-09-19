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

**Shaded rectangle**:
A filled rectangle drawn under text on a body page. Its fill color gives its kind: code
background, callout box, or highlight. Exported by Stage 1 next to the words of the page.
_Avoid_: rect, box, panel

**Code background**:
The light grey shaded rectangle under one line of a code block. A code block is a stack of them
with no gap.
_Avoid_: code box, grey box

**Callout box**:
The dark purple shaded rectangle around a boxed sidebar: one title rectangle and one or more body
rectangles. Example: "PHP Compiler" on PDF page 61.
_Avoid_: sidebar box, note box, aside

**Highlight**:
The light blue shaded rectangle that marks one emphasized line inside a code block. Carries no
structure of its own.
_Avoid_: selection, marker

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

**Ligature glyph**:
A JetBrains Mono glyph that draws an operator such as `->`, `::`, `__`, or `!==` as one shape.
The PDF text layer reads it as `=` plus a tail, so code text is wrong until glyph ids are mapped
back (issue #48). Stage 2 output still carries the corruption.
_Avoid_: font bug, encoding error

### Parsing units

**Word**:
One token from the PDF text layer with its font name, size, and position on the page. The font
name has no subset prefix.
_Avoid_: token, glyph run

**Line**:
Words on one page whose `top` values lie within the line tolerance of each other, ordered by
`x0`. The unit that gets a block type, by its dominant font.
_Avoid_: row, text line

**Block**:
One typed unit of the book: chapter title, heading, callout, code, or paragraph. Built from
consecutive lines of one type on one page, then merged across a page break for code and callouts.
_Avoid_: segment, element, node

**Callout**:
A block built from one callout box: the title line first, then the body lines. Keeps its own
type so retrieval can tell a sidebar from the main text.
_Avoid_: sidebar, note, aside, tip

**Inline code**:
A mono-font word on a prose line. The line stays a paragraph and the word keeps its printed text
with no added markup.
_Avoid_: code span, backtick code

**Parsed page**:
One PDF page after header removal: its words plus the count of header words deleted.
_Avoid_: page record
