PY ?= python3

.PHONY: test data all clean
test:
	$(PY) -m pytest -q

# Filled in as steps are completed
data:
	@echo "simulator not implemented yet (Step 2)"

all: test data

clean:
	rm -rf data/generated/* results/tables/* results/figures/*
