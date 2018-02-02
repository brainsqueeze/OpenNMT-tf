#!/usr/bin/env bash
nohup tensorflow_model_server --port=9000 \
    --enable_batching=true \
    --batching_parameters_file=examples/serving/batching_parameters.txt \
    --model_name=hi-en \
    --model_base_path=$HOME/Documents/OpenNMT-tf/hi-en/export/latest & > nohup_tf_serving.out&

nohup python examples/serving/server.py --model_name hi-en \
    --timeout 30 \
    --host localhost \
    --port 9000 & > nohup_flask.out&