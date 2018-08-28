#!/usr/bin/env bash

LIBRARY_PATH="$(find $HOME -name "OpenNMT-tf")"
MODEL_SERVER_PORT=9000
MODEL_NAME=es-en

export MODEL_NAME=${MODEL_NAME}
export TIME_OUT=30
export SERVER_NAME=localhost
export SERVER_PORT=${MODEL_SERVER_PORT}

python ${LIBRARY_PATH}/examples/serving/server.py