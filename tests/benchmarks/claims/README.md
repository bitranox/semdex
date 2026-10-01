# Prose claim files

`scripts/gen_bench_tables.py` makes every published benchmark TABLE a rendering of
`tests/benchmarks/raw/*.json`, so a table cannot drift from its data. The sentences around the
tables had no such gate. Rewriting `docs/benchmarks/03-chunking.md` published 42 hand-transcribed
figures and one of them was already wrong, which is the same drift arriving through the prose.

A claim file closes that. One file per documentation page, named after it, pairing each published
figure with the derivation that reproduces it. `scripts/check_bench_claims.py` recomputes every
derivation and `tests/test_bench_claims_current.py` gates the result.

Run it by hand with:

    python scripts/check_bench_claims.py

## What a claim guarantees, and what it does not

A passing claim says four things: the quoted sentence is still in the page's prose, it appears
there exactly once, it still contains the figure, and the derivation still produces that figure to
the precision the page prints.

It does not say the derivation is the RIGHT reading of the sentence. Nothing can check that, which
is why every claim carries a `why` naming what it reduced and over which rows. Read the `why`
against the sentence when reviewing.

It also cannot see a figure nobody wrote a claim for. Coverage is answered by reading this file
beside its page, never by a green run.

## File layout

    doc = "docs/benchmarks/03-chunking.md"

    [sets]
    fit_corpora = ["gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice"]

    [rowsets.effects]
    source = "chunk-knob-effects.json"   # a file in tests/benchmarks/raw
    path = "effects"                     # the key holding the list; omit if the file IS a list.
                                         # A key holding a single table is a one-row rowset, for
                                         # a scalar fact kept beside the rows
    orient = true                        # paired comparisons, see below

    [[claim]]
    id = "headline-max-tokens-resolved"
    quote = "all 41 of its resolved"
    published = "41"
    rows = "effects"
    reduce = "count"
    where = { axis = "max_tokens", resolved = true, corpus = "@fit_corpora" }
    why = "resolved chunk-size comparisons on the corpora that can carry a chunking claim"

## Claim keys

- `id` - unique across every claim file; it is what a failure names.
- `quote` - the sentence fragment as the page prints it, on ONE line, unique in the page's prose,
  and containing `published` verbatim.
- `published` - the figure exactly as printed, sign, comma and trailing zero included
  (`"+0.0143"`, `"12,298"`, `"36.0"`).
- `rows` - the rowset to reduce.
- `reduce` - `count`, `sum`, `mean`, `min`, `max`, `value` (refuses anything but a single row), or
  the two-selection kinds `ratio`, `difference` and `percent_change`.
- `field` - what to reduce; required for everything but `count`. A dotted name reaches into a
  nested table (`held_fixed.strategy`).
- `where` - the filter. A scalar means equals; a list means one of; `"@name"` expands a list
  declared under `[sets]`; a filter against a list-valued field asks whether it contains the value.
- `against`, `against_rows`, `against_field` - the second selection for `ratio`, `difference` and
  `percent_change`; each defaults to the first selection's value, so two fields of ONE row is
  written by giving only `against_field`.
- `transform` - `abs` or `negate`, applied before the reduction.
- `scale` - multiplier applied after it. A stored fraction published as a percentage is `100`.
- `tolerance` - an explicit absolute tolerance. Leave it out unless you need it.
- `why` - what this derivation means. Not optional: it is the only record of the reading.

## Tolerance

By default a claim asserts that the printed figure is the CORRECT ROUNDING of the derived one, so
the tolerance is half of the last printed place: `"36.0"` allows 0.05, `"+0.0143"` allows 0.00005,
`"41"` allows 0.5. A page that rounds more coarsely than its digits suggest needs `tolerance` set
explicitly, which forces whoever writes the claim to say so out loud.

## Oriented rowsets

`chunk-knob-effects.json` stores the HIGHER level as `from_level` in about half its rows, so an
unoriented filter would miss them and an unoriented sign would read backwards. `orient = true`
runs each row through the table generator's own `_oriented`, so a filter can be written straight
off the published table (`from_level = 0, to_level = 51`), and adds four fields:

- `favoured_level` - the level a resolved comparison came out in favour of.
- `unfavoured_level` - the other one.
- `favoured_end` - `"low"` or `"high"`, for a claim about the direction rather than the level.
- `levels` - both levels, for a claim about comparisons INVOLVING one of them.
- `material` - carried from the file, not derived: `true` when the comparison resolves AND its
  absolute `mean_delta` is at or above the file's `material_floor` (0.005 nDCG@10 today). A claim
  about a finding filters on `material = true`; one about what the query set can merely see
  filters on `resolved = true`.

## Adding a claim

1. Find the sentence and copy the fragment and the figure out of it verbatim.
2. Work out the derivation against the raw file, and check it produces the printed number.
3. Add the claim and run `python scripts/check_bench_claims.py`.
4. If it fails, the derivation is wrong or the page is. Decide which before changing either.

Two guards exist to stop a broken claim reading as a passing one: a filter key that no row in the
rowset carries is refused rather than silently matching nothing, and an unknown key anywhere in a
claim is refused rather than ignored. Both are misspellings that would otherwise leave a claim
evaluated over the wrong rows.
