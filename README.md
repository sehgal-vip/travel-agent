# Travel Planning Assistant

A production-grade, destination-agnostic multi-agent travel planning system. Works for **any country, any cities, any trip duration**.

## Tech Stack

- **LangGraph** — agent orchestration
- **Claude API** — LLM (Sonnet for most agents, Opus for planning)
- **Telegram Bot** — user interface
- **Markdown Library** — local knowledge base
- **SQLite** — state persistence
- **Tavily** — web research

## Quick Start

1. **Clone and install**
```bash
cd travel-agent
pip install -e ".[dev]"
```

2. **Configure environment**
```bash
cp .env.example .env
# Fill in your API keys
```

3. **Run**
```bash
python -m src.main
```

4. **Chat** — Open Telegram, find your bot, send `/start`

## Deploy with Docker

```bash
docker compose up -d
```

## Deploy to Railway

```bash
railway up
```

See the Hosting Guide for detailed deployment instructions.

## Architecture

9 specialized agents orchestrated by LangGraph:

| Agent | Role |
|-------|------|
| Orchestrator | Routes messages, manages flow |
| Onboarding | Conversational trip setup |
| Research | Destination intel + city research |
| Librarian | Markdown knowledge base management |
| Prioritizer | Tier assignment (must-do / nice-to-have / if-nearby) |
| Planner | High-level day-by-day itinerary |
| Scheduler | Detailed 2-day agendas with logistics |
| Feedback | End-of-day check-ins and adjustments |
| Cost | Dynamic budget tracking |

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Begin planning a new trip |
| `/research <city>` | Research a city |
| `/research all` | Research all cities |
| `/library` | Sync markdown knowledge base |
| `/priorities` | View/adjust priority tiers |
| `/plan` | Generate itinerary |
| `/agenda` | Detailed 2-day agenda |
| `/feedback` | End-of-day check-in |
| `/costs` | Budget breakdown |
| `/status` | Planning progress |
| `/trips` | List all trips |
| `/help` | Show commands |

## Testing

```bash
pytest tests/
```
