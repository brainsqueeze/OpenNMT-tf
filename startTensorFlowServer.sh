#!/usr/bin/env bash

LIBRARY_PATH="$(find $HOME -name "OpenNMT-tf")"
MODEL_SERVER_PORT=9000
MODEL_NAME=es-en
MODEL_PATH=${HOME}/Documents/trained_models/translations/${MODEL_NAME}/v2

tensorflow_model_server --port=${MODEL_SERVER_PORT} \
    --enable_batching=true \
    --batching_parameters_file=${LIBRARY_PATH}/examples/serving/batching_parameters.txt \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_PATH}/export/latest