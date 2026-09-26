.PHONY: up down up-local down-local status ingest serve eval test calibrate

# ---- services: Docker path (spec default) ----------------------------------
up:            ## start Qdrant and Phoenix with docker compose
	docker compose up -d

down:
	docker compose down

# ---- services: native path (no Docker) -------------------------------------
# Qdrant: native binary in .local/qdrant/ (download from github.com/qdrant/qdrant/releases).
# Web UI at /dashboard: Qdrant serves ./static relative to its working directory. Unzip
# dist-qdrant.zip from github.com/qdrant/qdrant-web-ui/releases so that
# .local/qdrant/static/index.html exists. Optional.
# Phoenix: `uv run --extra local` installs the server extra on demand, so a plain `uv sync`
# that drops the extra cannot break `make up-local`.
# Phoenix: `uv sync --extra local` installs the server; `phoenix serve` runs it.
LOCAL   := .local
QDRANT  := $(LOCAL)/qdrant/qdrant

up-local:      ## start Qdrant (:6333) and Phoenix (:6006) as background processes
	@test -x $(QDRANT) || { echo "missing $(QDRANT); see README"; exit 1; }
	@mkdir -p $(LOCAL)/qdrant/storage $(LOCAL)/phoenix
	@cd $(LOCAL)/qdrant && nohup ./qdrant > qdrant.log 2>&1 & echo $$! > $(LOCAL)/qdrant.pid
	@PHOENIX_WORKING_DIR=$(abspath $(LOCAL)/phoenix) nohup uv run --extra local phoenix serve > $(LOCAL)/phoenix/phoenix.log 2>&1 & echo $$! > $(LOCAL)/phoenix.pid
	@echo "qdrant pid $$(cat $(LOCAL)/qdrant.pid), phoenix pid $$(cat $(LOCAL)/phoenix.pid)"

down-local:    ## stop the native processes
	@-for p in qdrant phoenix; do test -f $(LOCAL)/$$p.pid && kill $$(cat $(LOCAL)/$$p.pid) 2>/dev/null; rm -f $(LOCAL)/$$p.pid; done
	@-pkill -f "phoenix serve" 2>/dev/null; true

status:        ## check both services answer
	@curl -sf http://localhost:6333/readyz >/dev/null && echo "qdrant  :6333 up" || echo "qdrant  :6333 DOWN"
	@curl -sf http://localhost:6006/healthz >/dev/null && echo "phoenix :6006 up" || echo "phoenix :6006 DOWN"

# ---- pipeline --------------------------------------------------------------
ingest:        ## run Stages 1-8, print the verification report
	uv run flp ingest

serve:         ## start the API on port 8000
	uv run uvicorn flp_rag.api.app:app --reload --port 8000

eval:          ## run the harness, compare with eval/baseline.json, exit 1 on regression
	uv run flp eval

test:          ## run the "Done when" tests
	uv run pytest -q

calibrate:     ## threshold sweep of spec Section 8.4
	uv run flp calibrate
