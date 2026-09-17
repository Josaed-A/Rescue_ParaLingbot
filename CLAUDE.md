# Instrucciones para Claude Code

La documentación del proyecto y la bitácora técnica completa viven en el README:

@README.md

## Reglas de trabajo en este repo

- **Bitácora append-only:** la sección "Bitácora técnica de la investigación" de `README.md` es acumulativa. Cada experimento nuevo va como sección fechada al final de la bitácora (antes de "Filosofía de la investigación"); los resultados anteriores no se editan ni se borran. En el "Estado actual" de la bitácora solo se agregan punteros a las secciones nuevas.
- **Presentación del README:** la parte de arriba (hasta "Bitácora técnica") es el resumen para quien llega al repo. Actualizarla cuando cambie el mejor resultado o el uso recomendado.
- **No tocar `demo.py` ni `lingbot_map/` para experimentos u optimizaciones:** van en scripts aparte (`scripts_context/`, `scripts_gpu/`, `scripts_seq/`, `scripts_webcam/`).
- **Commits solo cuando el usuario los pida,** con la identidad local del repo (`Pc_semillero`). Nunca cambiar la configuración global de git.
- **Fuera de git:** `captures/`, checkpoints y modelos (`env/assets.json` los describe).
- **GPU:** lanzar las corridas con `scripts_gpu/run_gpu.sh` (chequeo previo y bloqueo de suspensión). Los flags validados para 8 GB están en el README.
