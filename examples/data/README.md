# Stored example data

`makelov_read_source.csv` holds the measured IO-minus-subject logit margins of the Makelov
read-source pilot: 32 base prompt pairs (`unit`), both swap directions (`repeat`), and four
conditions (`baseline`, `full`, `read_row`, `read_null`) at MLP 8 of GPT-2 Small. It was
built by `make_makelov_read_source_csv.py` from the pilot's `records.jsonl` (sha256
`5e63b3b9…`, branch `applications/makelov-2311.17030`). It holds numbers only: no prompts,
no model weights, no upstream vectors.

`makelov_read_source_candidates.json` declares the two explanations compared there.
Both are anchored to measured endpoints, so each reproduces the full patch by
construction; they differ only in what they predict for the two read conditions.

This is development data. The contract of a confirmation (loss, scope and tolerance) has
to be declared before fresh data are seen. `examples/from_data.py` therefore shows the
pilot's own loss at two illustrative tolerances instead of claiming one.
