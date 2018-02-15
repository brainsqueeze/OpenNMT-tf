"""Example of a translation client."""

from __future__ import print_function

import argparse
import json

import tensorflow as tf

from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from flask import Flask, request, Response
from flask_cors import cross_origin
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.options import parse_command_line

from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)


def parse_translation_result(result):
    """Parses a translation result.

    Args:
      result: A `PredictResponse` proto.

    Returns:
      A list of tokens.
    """
    lengths = tf.make_ndarray(result.outputs["length"])[0]
    hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]

    # Only consider the first hypothesis (the best one).
    best_hypothesis = hypotheses[0]
    best_length = lengths[0]

    return best_hypothesis[0:best_length - 1]  # Ignore </s>


def translate(stub, model_name, tokens, timeout=5.0):
    """Translates a sequence of tokens.

    Args:
      stub: The prediction service stub.
      model_name: The model to request.
      tokens: A list of tokens.
      timeout: Timeout after this many seconds.

    Returns:
      A future.
    """
    length = len(tokens)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto([tokens], shape=(1, length)))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto([length], shape=(1,)))

    return stub.Predict.future(request, timeout)


def main():
    parser = argparse.ArgumentParser(description="Translation client example")
    parser.add_argument("--model_name", required=True,
                        help="model name")
    parser.add_argument("--host", default="localhost",
                        help="model server host")
    parser.add_argument("--port", type=int, default=9000,
                        help="model server port")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="request timeout")
    args = parser.parse_args()

    channel = implementations.insecure_channel(args.host, args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return args.model_name, args.timeout, stub


name, to, st = main()


@app.route('/translate', methods=['POST', 'GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def translate_api():
    j = request.get_json()
    if j is None:
        j = request.args
    if not j:
        j = request.form

    if 'text' not in j:
        return Response(response=json.dumps({"errorMessage": "No 'text' parameter"}, indent=2),
                        status=500,
                        mimetype="application/json")

    text = j['text']
    batch_tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    print(batch_tokens)

    futures = []
    for tokens in batch_tokens:
        future = translate(st, name, tokens, timeout=to)
        future = parse_translation_result(future.result())
        futures.append(future)

    translation = " ".join([" ".join(future) for future in futures])
    results = {
        "originalText": text,
        "translatedText": translation
    }

    return Response(response=json.dumps(results, indent=2),
                    status=200,
                    mimetype="application/json")


def run_server():
    """
    This initializes the Tornado WSGI server to allow for
    asynchronous request handling
    """

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(6006)

    parse_command_line()

    io_loop = IOLoop.instance()

    try:
        io_loop.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_server()
