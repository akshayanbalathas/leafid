# assets/

## species_info.csv

Facts for the species card: one row per class in `splits/classes.txt` (185 rows, same order).
The LLM only rewords these; it is told never to contradict them, so **they must be correct**.

| Column | Meaning |
|---|---|
| `species` | class name exactly as in `splits/classes.txt` (including LeafSnap's spellings, e.g. `aesculus_hippocastamon`) |
| `common_name` | English name shown on the card |
| `ontario_status` | `native` · `non-native` · `invasive` (the app maps these to the status line) |
| `notes` | leaf identification tip + Ontario context (range, species at risk, pests) |
| `verified` | `no` · `first-pass` · `yes` (see below) |
| `to_check` | for `first-pass` rows: the source used, or what still needs confirming |

> **⚠️ Side note: verification is not finished (as of Nov 23, 2026).**
> All 185 rows were drafted from general knowledge. The 24 least certain rows had a first pass against
> sources (Ontario species-at-risk pages, conservation-authority invasive lists, Ontario tree pages):
> `verified = first-pass`, with the source in `to_check`. **The other 161 rows are `verified = no`.**
> Akshayan will spot-check the 24 sources and skim the 161 on a separate branch (`species-info-verify`)
> before the demo. Weakest first-pass sources: `amelanchier_canadensis` and `crataegus_pruinosa`
> (no Ontario government page found). Tracked in [`BACKLOG.md`](../BACKLOG.md).

**How to verify a row:** look the Latin name up on [VASCAN](https://data.canadensys.net/vascan) (Ontario =
native / introduced), the Ontario Invasive Plant Council (invasive?) and Ontario's Species at Risk list
(at-risk wording). Fix the row, set `verified = yes`, clear `to_check`. Edit in Google Sheets or a text editor,
not Excel (it can change the encoding and turn values like `1-2` into dates).
