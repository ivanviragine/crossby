# carbrain

A versioned evidence database about the Brazilian car market, with a cited tool layer for a
chatbot and for posts. It implements the first phase of the *Brazil Car Data Blueprint*
(revision 2).

Every fact records its source, the period it describes (`as_of`), when it was fetched, and
the vehicle it applies to. Corrections become new revisions and nothing is overwritten. A
rights registry decides what may be collected, kept, embedded, sent to an AI provider,
shown and published, per source.

## Quick start

```bash
cd carbrain
uv sync --all-extras
export CARBRAIN_DATA_DIR=$PWD/var        # where the database and raw files live
uv run carbrain init                      # schema, 20 vehicle families, events, 30 publishers
uv run carbrain sync                      # all dataset connectors (fleet file: ~3 min first time)
uv run carbrain sync-content              # latest headlines from specialist media
uv run carbrain status                    # freshness per source
uv run carbrain tool fuel_prices '{"place": "Curitiba, PR"}'
```

Run `./scripts/daily.sh` once a day (cron or any scheduler). Unchanged sources cost almost
nothing: files are content-hashed, and monthly files are never downloaded twice.

## Sources, by type

`uv run carbrain sources` lists them with what their rights allow.

| Type | Source | Status |
|---|---|---|
| official_statistics | Banco Central SGS (vehicle loan rates) | implemented |
| official_statistics | IBGE IPCA (deflator) | implemented |
| official_statistics | ANP weekly fuel prices (state, city) | implemented |
| industry_association | ANFAVEA registrations by segment | implemented |
| vehicle_registry | SENATRAN fleet by brand/model label | implemented |
| lab_testing | INMETRO PBEV consumption (PDF table) | implemented |
| specialist_media | Publisher RSS/Atom feeds (headlines only) | implemented |
| social_platform | YouTube Data API | implemented; needs `YOUTUBE_API_KEY` |
| social_platform | Instagram Graph API | implemented; blocked until rights are confirmed |
| price_reference | FIPE | blocked until a contract; dev sample only (`fipe-sample`) |
| industry_association | Fenabrave | registered, not implemented |
| consumer_complaints | ReclameAqui | registered; needs an agreement |
| specialist_database | Carros na Web | registered; needs permission |

## Publishers and creators

`data/publishers.yaml` catalogs specialist media (Quatro Rodas, Autoesporte, Motor1,
AutoPapo and others) and creators, with their website, RSS, YouTube and Instagram channels.
Each channel cites dated evidence and carries a status: `web_evidence`, `feed_verified` or
`api_verified`. Browse it with `uv run carbrain catalog list --focus ev` (or `--kind`,
`--platform`). With a YouTube key, `uv run carbrain catalog verify --platform youtube`
confirms handles through the API.

## Chatbot tools

| Tool | What it answers |
|---|---|
| `find_vehicle` | Identify which vehicle families (and model year, if given) the user means. |
| `fleet` | Registered fleet of a family: total, by state and by year of manufacture. |
| `registrations` | Monthly new-vehicle registrations by segment (ANFAVEA), with year-on-year change. |
| `fuel_prices` | Latest weekly average pump prices by fuel for a city, state or Brazil (ANP). |
| `loan_rate` | Latest average interest rate on vehicle loans to individuals (Banco Central). |
| `deflate_price` | An amount from one month in reais of another month, using IPCA. |
| `flex_fuel_choice` | Cheaper fuel for a flex car at local prices, using the car's own consumption. |
| `ownership_cost` | Cost of owning a car for N years: low/base/high scenarios with every assumption. |
| `consumption` | Standardized consumption per version (INMETRO PBEV): km/l, energy use, EV range. |
| `fipe_price` | FIPE reference price. Restricted until FIPE confirms display rights. |
| `events` | Tax changes, rating-protocol changes and other breaks that explain market moves. |
| `information_sources` | Specialist media and creators worth following, by kind, focus or platform. |
| `expert_content` | Recent headlines and videos about a vehicle, with links to the originals. |
| `data_freshness` | When each source was last checked and the latest period its data describes. |

Every result has a status (`ok`, `no_data`, `restricted`), citations with reference periods,
and notes, including a warning when a source is stale. `uv run carbrain ask "..."` runs the
Claude chat loop over these tools (needs the `chat` extra and Anthropic credentials). It
uses server-side fallbacks, and a guard sends the answer back once if it contains a number
that no tool returned.

## Development

- `./scripts/check.sh` runs ruff, the format check, strict mypy and the offline tests.
- `uv run pytest -m live -s` runs smoke tests against the real sources.
- Read `LEARNINGS.md` before changing anything and update it after (see `CLAUDE.md`).

## Not done yet

- Version-level catalog (trims, model years) and list prices from automaker sites.
- Fenabrave PDFs (registrations by model) and the retail vs direct-sales split.
- Recalls (Senacon), Latin NCAP, Consumidor.gov.br: their open-data portal was unreachable
  from the build environment.
- Owner-evidence pipeline (aspect sentiment, issue reports) on top of `text_item`.
- Licensed data (FIPE contract, JATO/Molicar, marketplace listings, social listening).
