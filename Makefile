.PHONY: up down ingest serve eval test calibrate

up:            ## start Qdrant and Phoenix
	docker compose up -d

down:
	docker compose down

ingest:        ## run Stages 1-8, print the verification report
	flp ingest

serve:         ## start the API on port 8000
	uvicorn flp_rag.api.app:app --reload --port 8000

eval:          ## run the harness, compare with eval/baseline.json, exit 1 on regression
	flp eval

test:          ## run the "Done when" tests
	pytest -q

calibrate:     ## threshold sweep of spec Section 8.4
	flp calibrate
