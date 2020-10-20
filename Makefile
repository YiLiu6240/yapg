PYTHON_FILES=$$(git ls-files -- . | grep "\.py$$")

# -------- tang poetry --------

## tang poetry: process vocab
tang-poetry-vocab:
	python -m scripts.process_tang_poetry_vocab

## tang poetry: train model
tang-poetry-train:
	python -m scripts.tang_poetry_train \
		--fp16

## tang poetry: generate poetry (remember to have a checkpoint model in place)!
tang-poetry-gen:
	python -m scripts.tang_poetry_gen \
		--starting-text "数据挖掘哪家强" \
		--max-doc-length 21

# -------- english haiku poetry --------

## english haiku: process source data
haiku-process-corpus:
	python -m scripts.process_haiku_corpus

## english haiku: train model
haiku-train:
	python -m scripts.haiku_train \
		--batch_size 48 \
		--num_train_epochs 20 \
		--fp16

## english haiku: generate poetry (remember to have a checkpoint model in place)!
haiku-gen:
	python -m scripts.haiku_gen \
		--starting-text "how to do data mining"


# -------- utils --------

## Lint codebase
lint:
	python -m flake8 ${PYTHON_FILES}
	python -m mypy ${PYTHON_FILES}

## Format codebase
fmt:
	python -m autoflake --in-place --remove-all-unused-imports --recursive ${PYTHON_FILES}
	python -m isort ${PYTHON_FILES}
	python -m black ${PYTHON_FILES}

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Params:$$(tput sgr0)"
	@echo
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'
