.PHONY: setup run

# Installs requirements for both server and client
setup:
	python3 -m venv server/venv
	./server/venv/bin/pip install -r requirements.txt
	./server/venv/bin/python -m spacy download en_core_web_sm
	cd client && npm install

run:
	cd client && npm run build && cd ..
	./server/venv/bin/python3 server/server.py
