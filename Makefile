.PHONY: setup run

setup:
	python3 -m venv server/venv
	./server/venv/bin/pip install -r requirements.txt
	cd client && npm install

run:
	cd client && npm run build && cd ..
	./server/venv/bin/python3 server/server.py
