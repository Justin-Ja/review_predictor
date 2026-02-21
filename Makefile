.PHONY: setup run

# Installs requirements for both server and client
setup:
	python3 -m venv server/venv
	./server/venv/bin/pip install -r requirements.txt
	cd client && npm install

run:
#God i fucked something up. apps broken if using the venv python. im working on it.  sorry
	cd client && npm run build && cd ..
	# ./server/venv/bin/python3 server/server.py
	python3 server/server.py
