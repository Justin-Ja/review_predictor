#!/bin/bash

#Execute at the root directory of the project
cd client
npm run build
cd ..
python3 server/server.py