# Learnings

A running log of what worked, what broke, and what to do differently on this
project. **Read it before starting any implementation step. Update it right
after.** Keep entries short and concrete; delete or rewrite entries that turn
out to be wrong instead of piling on contradictions.

Format: `- [area] lesson — why / evidence (date)`

## Rules I follow on this project

1. Read this file before each implementation step; add what I learned right after the step.
2. Verify a claim against a primary source before writing it down as fact. When I can't, label it "unverified" in the same sentence.
3. Build parsers from real downloaded files, not from memory of a format. Save a trimmed copy as a test fixture.
4. Every fact stored carries source, `as_of` (period it describes), `fetched_at`, and the vehicle it applies to.
5. Check the rights registry before collecting, keeping, embedding, sending to an AI provider, showing or publishing any source's data.
6. Never let the chatbot produce a number that no tool returned.
7. Keep sources organized: every data source has a `type` in `sources.yaml`; every publisher (specialist media, creator) lives in `publishers.yaml` with dated evidence for each channel; new platforms get a `Fetcher`, new datasets a `Connector` — never a one-off script.
8. After tests pass, use the product on real data (CLI + live run) before calling a step done.

## Research and sources

- [youtube] Derived metrics such as sentiment need YouTube's "Analytics & Reporting" approval; comment text must be refreshed or deleted within 30 days even with approval; derived metrics may be kept up to 36 months. I first rated YouTube "Low risk" and a second review caught it — check the platform's *developer policies*, not just API docs (2026-09-27).
- [fipe] fipe.org.br says lookups are "oficialmente e exclusivamente" through its site, no API, no downloads. veiculos.fipe.org.br returns 403 from this environment. A second analysis claimed a complete-table contract offer; unconfirmed. Treat FIPE collection as blocked until a contract exists (2026-09-27).
- [complaints] Complaint counts, even per 10,000 registered vehicles, are not failure rates. Use them to decide what to investigate; tell consumers "recurs in N reports we examined" (2026-09-27).
- [identity] Use an internal vehicle ID (brand → model → generation → version → model year → configuration revision). FIPE codes are a mapping, not the backbone: they lag launches and don't mark in-year changes (2026-09-27).
- [consumidor] dados.mj.gov.br did not resolve from this environment, so the Consumidor.gov.br file schema is unverified. A 2020 "Relato do Consumidor" file exists (2026-09-27).
- [autoseg] SUSEP AutoSeg's latest download is 2nd half 2020 — stale for insurance costs (2026-09-27).

## Environment

- [network] Reachable: api.bcb.gov.br, apisidra.ibge.gov.br, www.gov.br (ANP, INMETRO pages), dados.transportes.gov.br, fenabrave.org.br, anfavea.com.br, parallelum.com.br. Blocked: veiculos.fipe.org.br (403), ANP `ca-*.zip` semiannual files (403 via proxy), dados.mj.gov.br (DNS) (2026-09-27).
- [anp] Weekly ANP file names are inconsistent (`resumo_semanal_lpc_2026-08-30-2026-09-5.xlsx` vs `..._2026-09-06_2026-09-12.xlsx`). Discover links from the listing page; never build the URL from dates (2026-09-27).
- [repo] This project lives in `carbrain/` inside the crossby repo on branch `claude/brazil-car-market-data-g6b51a`. crossby's CI only checks `src/` and `tests/`, so `carbrain/` is invisible to it; run carbrain's own checks with `./scripts/check.sh` from `carbrain/` (2026-09-27).

## Source formats (verified from real files, 2026-09-27)

- [senatran] Fleet by brand/model: CKAN dataset `registro-nacional-de-veiculos-automotores-renavam` on dados.transportes.gov.br; resources named `i_frota_por_uf_municipio_marca_e_modelo_ano_<mes>_<ano>.zip`. 136 MB zip → one 1.2 GB **UTF-8** (CRLF) `;`-separated TXT: `UF;Município;Marca Modelo;Ano Fabricação Veículo CRV;Qtd. Veículos`, quantities like ` 1.0`. The UF column also holds `Sem Informação`, `Não se Aplica`, `Não Identificado` (stored as UF `XX`). Stream it; never load it. Full parse ≈ 25 s CPU; the run is dominated by the ~2.5 min download.
- [encoding] **Correction:** I first wrote "Latin-1" for the fleet file because I decoded it with `iconv -f latin1` and misread the mojibake (`MunicÃ­pio`) as correct text. `Ã` followed by a symbol means UTF-8 bytes read as Latin-1. Detect the encoding (try UTF-8 on the first MiB) instead of assuming (2026-09-27).
- [senatran] Downloads get cut mid-transfer (51 MB of 136 MB). The server honors `Range`, so resume instead of restarting.
- [registry strings] Imports are `I/BRAND MODEL` or `IMP/BRAND MODEL` (no slash between brand and model). Names are truncated around 20–24 chars (`CRETA1TA PLTINUM`, `ENDURAN CS13`). Traps: `DOLPHIN` vs `DOLPHIN MINI`; `HILUX SW…`/`HILUXSW4` is the SW4 SUV, not the Hilux pickup; Corolla Cross is `CCROSS`; T-Cross is `T CROSS`; `ONIX PLUS` is the sedan; `HB20S` is the sedan; Chevrolet appears as both `CHEV/` and `GM/`; CAOA Chery as `CAOACHERY/`.
- [anp] Weekly summary xlsx: sheets CAPITAIS, MUNICIPIOS, ESTADOS, REGIOES, BRASIL. Header row starts with `DATA INICIAL`; header names differ per sheet (`ESTADO`+`MUNICÍPIO` vs `REGIAO`+`ESTADOS`). Names are upper case without accents (`SAO PAULO`).
- [anfavea] `siteautoveiculos<ano>.xlsx`, sheet `I. Emplacamento`: blocks titled "Emplacamento de autoveículos nacionais/importados/…", a year row, a `Jan…Dez, Total Ano` row, then labels in column B (level 1) or C (level 2). Level-2 names repeat under different parents (`Leves` under trucks), so segment IDs include the parent. **Unpublished months are empty in the 2025 file but `0` in the 2026 file** — found only in a live run. A month counts as published when the block's Total row is positive; real zeros in sub-rows are kept.
- [bcb] `api.bcb.gov.br/dados/serie/bcdata.sgs.<code>/dados?formato=json` → `[{"data":"01/07/2026","valor":"1.98"}]` (dd/mm/yyyy, value as string).
- [ibge] SIDRA `/values/t/1737/n1/all/v/2266/p/all` → first element is a header dict; `D3C` is `YYYYMM`, `V` the index as a string.
- [fipe] Value JSON: `{"Valor":"R$ 74.594,00","Marca":"VW - VolksWagen","Modelo":"Polo Track 1.0 Flex 12V 5p","AnoModelo":2025,"Combustivel":"Flex","CodigoFipe":"005540-9","MesReferencia":"setembro de 2026","SiglaCombustivel":"F"}`. Zero km uses year `32000`. Captured with one lookup for the parser fixture.
- [pbev] INMETRO publishes the PBEV table only as a PDF. The file named `mascara-pbev-2026_19_jan-rev01.pdf` holds the 2026 table *updated 14 Aug 2026* (976 rows, 9 landscape pages): file names lie about dates, so pick the link with the highest year and read the date from page 1.

## Implementation

- [resolve] Use `(?![A-Z])` after a family name, not `\b`: truncated registry labels glue digits to names (`CRETA1TA`), and `\b` would miss them, while `(?![A-Z])` still rejects `HB20S` and `GOLF` (2026-09-27).
- [resolve] "Longest match wins" only protects against lookalikes we also track. A lookalike we don't track (Onix Plus, Hilux SW4) needs an explicit negative lookahead in the tracked family's pattern, plus a NOT_TRACKED test using the real label (2026-09-27).
- [tests] I wrote a wrong expectation (`ONIX HATCH` → no match) from memory. Derive test labels from real files, and when a test fails, first check whether the test or the code is wrong (2026-09-27).
- [bcb] `ultimos/N` is capped at 20 and errors come back as HTTP 400 with `{"erro": {...}}`. Query by date range instead (2026-09-27).
- [anfavea] The Excel page renders its links with JavaScript, so there's nothing to discover in the HTML. The per-year URL `anfavea.com.br/docs/siteautoveiculos<ano>.xlsx` works (2023, 2025, 2026 verified). This is the one place I build a URL, and it must tolerate a 404 for a year not yet published (2026-09-27).
- [tooling] `ruff format` rewrites files; re-read before editing them with exact-match tools (2026-09-27).
- [live runs] Fixture tests passed while two real-data bugs existed (fleet encoding, ANFAVEA zero months). Always finish a connector with a live run (`pytest -m live`), and turn every live surprise into a fixture-based regression test (2026-09-27).
- [resolve] Auditing all 40k real fleet labels (not a sample) found silent misses worth ~600k vehicles: `VW/NOVO GOL …` (marketing prefix NOVO/NOVA/NEW, now stripped before matching) and `HYUNDAI/HB2010TA …` (digits glued after HB20). Re-run the full-label audit whenever patterns change (2026-09-27).
- [resolve] `fuzz.partial_ratio` made review suggestions useless (GOLF→Gol, DUCATO **CARGO**→Argo). Compare the label's first model word with the family name (prefix or `fuzz.ratio` ≥ 85) and suppress each family's declared `lookalikes`. After the fix, 6 labels ≥1,000 vehicles need review — all Marcopolo bus bodies (`VW/MPOLO …`), a true lookalike for a person to reject once (2026-09-27).
- [pytest] Live tests must be excluded by default (`addopts = -m "not live"`); otherwise a plain `pytest` downloads 136 MB and blows the command timeout (2026-09-27).
- [archive] I "optimized" parse-skipping to key on content hash alone. Wrong: parsers read meaning from the URL (SGS series code, SENATRAN month), so the same bytes from two URLs are different facts. A test fixture that served identical bytes for two series exposed it. Dedupe on (source, URL, hash, parser version) (2026-09-27).
- [fixtures] Serving one fixture for two different endpoints hid the bug above and produced a fake "annual rate = 1.98%". Give every endpoint its own real fixture (2026-09-27).
- [chat guard] The number guard exempted "small numbers ≤ 31", which let `13,4 km/l` and the English reading of `3.500` (3.5) through. Exempt only whole numbers (counts, days) and years. Numbers are read both PT-BR (`74.594,00`) and EN style, and "120 mil" also counts as 120000 (2026-09-27).
- [rights] Rights are enforced at three points: collection (`run_connector`), serving (`_Cite.allowed` → `restricted` result, even when data exists, as for FIPE), and AI processing (`ChatSession` refuses to start if a tool source disallows it). Test each point (2026-09-27).
- [cli run] Running the tools on the real database found three things the tests didn't: YoY was always empty (sync fetched only the current ANFAVEA year), 9,666 Dolphin Minis had manufacture year `0` (missing in the registry; now shown as "unknown"), and the "several families match" note misread deliberate comparisons. Use the product on real data after the tests pass (2026-09-27).
- [daily cost] `sync` re-downloaded the 136 MB fleet file every day only to find it unchanged. Monthly files never change under the same URL, so `immutable_urls` skips URLs already parsed by the current parser: the daily check went from ~3 min to 1.6 s (2026-09-27).
- [chat] No Anthropic credentials in this environment, so no live chat. Instead `tests/test_chat_sdk.py` runs the loop through the real SDK with `httpx2.MockTransport` (anthropic 1.x uses httpx2), which checks the real request serialization: `fallbacks="default"`, the `server-side-fallback-2026-07-01` beta header and the cached system prompt (2026-09-27).
- [pdf tables] Plain text extraction scrambles the PBEV table. Rebuild rows from positions: column boundaries from vertical ruling lines (33 columns), words grouped into lines with a 2.5 pt gap (rows sit ~3 pt apart), each word assigned by its x-center. Fake-bold glyphs are drawn twice ~0.3 pt apart; pdfplumber's `dedupe_chars()` handles that but took 58 s for 9 pages, while a 1-pt tolerance dedupe keyed by (glyph, line) takes 12 s (2026-09-27).
- [coverage guard] Compare parsed rows with the document's own declared totals (976 rows; per-type counts on page 1). The guard caught my own regex bug on the first run (it read "5976" after I squashed spaces). Parsed 963/976; per-type counts match exactly for gasoline, diesel, hybrid and plug-in (2026-09-27).
- [source gaps] The Aug 2026 PBEV table prints no km/l for the Polo Track (`\` and blanks), only 1.48 MJ/km. Keep gaps as gaps; the tool says "not published" (2026-09-27).
- [pbev] INMETRO puts trim words in the version column (model `POLO`, version `TRACK 1.0 MPI`), so match on model + version. 27 rows repeat another row's exact label with different figures (e.g. two `TIGGO 7 PHEV`), so keys get a `#2` suffix instead of overwriting (2026-09-27).
- [catalog] Creator handles come from third-party lists dated 2022–2023, so each channel stores its evidence URL and date and a status ladder (`web_evidence` → `feed_verified` → `api_verified`). Nothing is presented as verified until an official interface confirms it (2026-09-27).
- [youtube] youtube.com pages are reachable from here, but YouTube's terms forbid automated access outside the API (and search engines). No page scraping: verification and uploads go through the Data API (`channels.list forHandle`, `playlistItems.list`, 1 quota unit each). With no key in this environment, the YouTube adapter is tested with fixtures shaped from the API reference — the one exception to rule 3. Replace them with captured responses on the first run with a key (2026-09-27).
- [feeds] Autoesporte dates items with `-0000` ("zone unknown" in RFC 5322), which Python parses as a naive datetime; mixing naive and aware timestamps breaks string comparisons in SQL. All feed dates are normalized to UTC (2026-09-27).
- [feeds] Quatro Rodas and AutoPapo put the full article in `content:encoded`. Only title, link, date and id are kept; a test dumps the whole database to prove body text never lands in it, and fixtures were stripped of article text. Untrusted XML goes through defusedxml (entity declarations refused, tested) (2026-09-27).
- [mentions] First live run: 5 of 6 "Tera" mentions were the verb "terá" (accent stripped → TERA). Families that are also common words carry `text_rules`: `accent_sensitive` (never match accented text) and `proper_noun` (in editorial text, need a capital or the brand). Known limit: a capital at sentence start proves nothing ("Gol de placa: ..." still matches), but rejecting it would lose real headlines like "Compass cai a R$ 119.990". Mentions are recomputed on every fetch, and `relink_mentions` fixes history after a rules change (2026-09-27).
- [cli] `catalog list` crashed on a channel known only by its YouTube channel ID. The CLI now has its own tests (`typer.testing.CliRunner`) (2026-09-27).

