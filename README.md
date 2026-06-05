# Neural Service Mesh — Phase 2

> API + Database + AI Intelligence Layer

## What's New in Phase 2

| Feature | Phase 1 | Phase 2 |
|---|---|---|
| Storage | JSON files | JSON + **SQLite** |
| API | Basic Flask | Full REST API (CRUD + AI endpoints) |
| AI Layer | ❌ | ✅ Rules + Heuristics (ML-ready for Phase 3) |
| Execution | BFS path only | BFS + **AI path selection** + **fallback** |
| Logging | Console | Console + **rotating file logs** |
| Node creation | Code only | Code + **via API** |

## Project Structure

```
project/
├── core/
│   ├── node.py          ← BaseNode, NodeSchema (Phase 1, unchanged)
│   ├── graph.py         ← ServiceGraph, BFS/DFS (Phase 1, unchanged)
│   ├── registry.py      ← NodeRegistry (Phase 1, unchanged)
│   └── engine.py        ← ExecutionEngine (Phase 2: + AI, fallback, DB)
├── api/
│   └── app.py           ← Flask REST API (Phase 2: full CRUD + AI endpoints)
├── ai/
│   └── decision.py      ← AIDecisionLayer (Phase 2: rules + heuristics)
├── storage/
│   ├── file_storage.py  ← JSON storage (Phase 1, compatibility)
│   └── db.py            ← SQLiteStorage (Phase 2: nodes, edges, logs)
├── logs/
│   └── mesh_logger.py   ← Structured logging (Phase 2)
├── services/
│   ├── input_service.py
│   ├── processor_service.py
│   ├── output_service.py
│   └── dynamic_node.py  ← PassThroughNode for API creation (Phase 2)
├── connectors/
│   ├── base_connector.py
│   └── data_transformer.py
├── data/                ← JSON + SQLite files
├── main.py              ← Entry point
└── requirements.txt
```

## Quick Start

```bash
pip install flask

# Run demo
python main.py --mode demo

# Start API server
python main.py --mode api --port 5000
```

## API Endpoints

### Nodes
| Method | Endpoint | Description |
|---|---|---|
| GET | `/nodes` | List all nodes |
| POST | `/nodes` | Create dynamic node |
| GET | `/nodes/<id>` | Get node details |
| DELETE | `/nodes/<id>` | Remove node |

### Graph & Connections
| Method | Endpoint | Description |
|---|---|---|
| GET | `/graph` | Full graph (nodes + edges) |
| GET | `/graph/stats` | Node/edge counts |
| POST | `/connect` | Connect two nodes |
| DELETE | `/connect` | Disconnect nodes |
| GET | `/connections` | List all connections |

### Execution
| Method | Endpoint | Description |
|---|---|---|
| POST | `/run` | Run between start → end (AI-assisted) |
| POST | `/run/path` | Run explicit path |
| POST | `/run/full` | Run entire graph |
| GET | `/runs` | List execution history |
| GET | `/runs/<id>` | Get single run details |

### AI Layer
| Method | Endpoint | Description |
|---|---|---|
| POST | `/ai/paths` | Rank all paths between two nodes |
| POST | `/ai/suggest` | Suggest next node |
| GET | `/ai/insights` | Performance insights + recommendations |

### Storage & Monitoring
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/status` | Full system status |
| GET | `/storage/stats` | Storage statistics |
| GET | `/logs/stats` | Execution log stats |

## Example: POST /run

```json
{
  "start_id": "<node_id>",
  "end_id": "<node_id>",
  "data": { "text": "Hello world" },
  "use_ai": true
}
```

## Phase 3 Preview (Next)

- ML-based path optimization (learning from history)
- Auto-scaling nodes
- Real-time WebSocket events
