# SPDX-License-Identifier: Apache-2.0
import argparse
import runpy
import os
import sys

from offline_inference_tt import check_tt_model_supported, register_tt_models

register_tt_models()  # Import and register models from tt-metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Model name")
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to be processed in a single iteration"
    )
    # Add host/port for vLLM OpenAI server exposure
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("SERVER_HOST", "0.0.0.0"),
        help="Host interface for the OpenAI-compatible server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SERVER_PORT", "8000")),
        help="Port for the OpenAI-compatible server (default: 8000)",
    )
    args, _ = parser.parse_known_args()

    check_tt_model_supported(args.model)

    sys.argv.extend([
        "--model",
        args.model,
        "--block_size",
        "64",
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--num_scheduler_steps",
        "10",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
