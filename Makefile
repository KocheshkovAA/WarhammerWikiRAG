export DOCKER_API_VERSION := 1.44

.PHONY: help debug-up up down eval 

help:                   ## Показать все доступные команды
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

debug-up:               ## Запустить стек с профилем debug (Langfuse + Clickhouse + Minio)
	docker compose --profile debug up -d

up:                     ## Запустить основной стек без debug
	docker compose up -d

down:                   ## Остановить и удалить все контейнеры
	docker compose down --remove-orphans --volumes

eval:                   ## Запустить оценку ретривера
	docker compose exec api python /app/app/eval/evaluate_retrieval.py

logs:                   ## Показать логи
	docker compose logs -f