# Neural Service Mesh — Phase 7 Setup Guide

## Folder Structure on GitHub

After downloading all files, recreate this structure:

```
Neural-Service-Mesh/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore                    ← rename gitignore.txt → .gitignore
│
├── ai/
│   ├── __init__.py               ← ai___init__.py
│   ├── code_generator.py         ← ai_code_generator.py        [NEW Phase 7]
│   ├── evolution_pipeline.py     ← ai_evolution_pipeline.py    [NEW Phase 7]
│   ├── governance_p7.py          ← ai_governance_p7.py         [NEW Phase 7]
│   ├── objectives.py             ← ai_objectives.py            [NEW Phase 7]
│   ├── phase6_validator.py       ← ai_phase6_validator.py      [NEW Phase 7]
│   ├── sandbox_lab.py            ← ai_sandbox_lab.py           [NEW Phase 7]
│   ├── self_awareness.py         ← ai_self_awareness.py        [NEW Phase 7]
│   ├── agent_factory.py
│   ├── capability_marketplace.py
│   ├── decision.py
│   ├── discovery_engine.py
│   ├── economic_engine.py
│   ├── evolution_engine.py
│   ├── gap_detector.py
│   ├── goal_planner.py
│   ├── governor.py
│   ├── learning_validator.py
│   ├── memory_engine.py
│   ├── meta_reasoner.py
│   ├── multi_goal_planner.py
│   ├── optimization_engine.py
│   ├── reputation_engine.py
│   ├── routing_engine.py
│   ├── scoring_engine.py
│   ├── self_optimizer.py
│   ├── semantic_matcher.py
│   ├── service_generator.py
│   ├── simulation_engine.py
│   ├── simulation_lab.py
│   ├── swarm_coordinator.py
│   ├── system_dna.py
│   └── validator.py
│
├── sensors/                      ← [NEW Phase 7 — full folder]
│   ├── __init__.py               ← sensors___init__.py
│   ├── api_sensor.py
│   ├── base_sensor.py
│   ├── filesystem_sensor.py
│   ├── log_sensor.py
│   ├── sensor_hub.py
│   └── webhook_sensor.py
│
├── world_model/                  ← [NEW Phase 7 — full folder]
│   ├── __init__.py               ← world_model___init__.py
│   └── environment_model.py
│
├── api/
│   ├── __init__.py               ← api___init__.py
│   └── app.py
│
├── core/
│   ├── __init__.py               ← core___init__.py
│   ├── engine.py
│   ├── graph.py
│   ├── node.py
│   └── registry.py
│
├── services/
│   ├── __init__.py               ← services___init__.py
│   ├── dynamic_node.py
│   ├── input_service.py
│   ├── output_service.py
│   └── processor_service.py
│
├── connectors/
│   ├── __init__.py               ← connectors___init__.py
│   ├── base_connector.py
│   └── data_transformer.py
│
├── storage/
│   ├── __init__.py               ← storage___init__.py
│   ├── db.py
│   └── file_storage.py
│
├── knowledge/
│   ├── __init__.py               ← knowledge___init__.py
│   └── knowledge_store.py
│
└── logs/
    ├── __init__.py               ← logs___init__.py
    └── mesh_logger.py
```

## Run Phase 7

```bash
pip install flask
python main.py --mode evolve7 --cycles 3
```
