"""Example of a translation client."""

from __future__ import print_function

import json
import os

from grpc import insecure_channel

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf

from flask import Flask, request, Response
from flask_cors import cross_origin
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.options import parse_command_line

from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)
MODEL_NAME = os.environ["MODEL_NAME"]
TIME_OUT = os.environ.get("MODEL_NAME", 30)
SERVER_NAME = os.environ.get("SERVER_NAME", "localhost")
SERVER_PORT = os.environ.get("SERVER_PORT", 9000)


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


def translate(stub, tokens, timeout=5.0):
    """Translates a sequence of tokens.

    Args:
      stub: The prediction service stub.
      tokens: A list of tokens.
      timeout: Timeout after this many seconds.

    Returns:
      A future.
    """
    length = len(tokens)

    req = predict_pb2.PredictRequest()
    req.model_spec.name = MODEL_NAME
    req.inputs["tokens"].CopyFrom(tf.make_tensor_proto([tokens], shape=(1, length)))
    req.inputs["length"].CopyFrom(tf.make_tensor_proto([length], shape=(1,)))

    # return stub.Predict.future(request=req, timeout=timeout)
    return stub.Predict.future(request=req)


@app.route('/translate', methods=['POST', 'GET'])
@cross_origin(origins=['*'], allow_headers=['Content-Type', 'Authorization'])
def translate_api():
    j = request.get_json()
    if j is None:
        j = request.args
    if not j:
        j = request.form

    if 'text' not in j:
        return Response(
            response=json.dumps({"errorMessage": "No 'text' parameter"}, indent=2),
            status=500,
            mimetype="application/json"
        )

    text = j['text']
    batch_tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]

    futures = []
    for tokens in batch_tokens:
        future = translate(stub=st, tokens=tokens, timeout=TIME_OUT)
        future = parse_translation_result(future.result())
        futures.append(future)

    translation = " ".join([" ".join(token.decode('utf8') for token in future) for future in futures])
    results = {
        "originalText": text,
        "translatedText": translation
    }

    return Response(response=json.dumps(results, indent=2),
                    status=200,
                    mimetype="application/json")


def run_server(port=6006):
    """
    This initializes the Tornado WSGI server to allow for
    asynchronous request handling
    """

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)

    parse_command_line()

    io_loop = IOLoop.instance()
    print("Listening to port", port)

    try:
        io_loop.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    channel = insecure_channel(target="{0}:{1}".format(SERVER_NAME, SERVER_PORT))
    st = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    run_server(port=6006)
