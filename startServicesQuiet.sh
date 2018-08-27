#!/usr/bin/env bash

LIBRARY_PATH="$(find $HOME -name "OpenNMT-tf")"
MODEL_SERVER_PORT=9000
MODEL_NAME=es-en

nohup tensorflow_model_server --port=${MODEL_SERVER_PORT} \
    --enable_batching=true \
    --batching_parameters_file=${LIBRARY_PATH}/examples/serving/batching_parameters.txt \
    --model_name=${MODEL_NAME} \
    --model_base_path=${LIBRARY_PATH}/${MODEL_NAME}/export/latest & > nohup_tf_serving.out&

export MODEL_NAME=${MODEL_NAME}
export TIME_OUT=30
export SERVER_NAME=localhost
export SERVER_PORT=${MODEL_SERVER_PORT}

nohup python ${LIBRARY_PATH}/examples/serving/server.py & > nohup_flask.out&