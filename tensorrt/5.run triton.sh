#!/bin/bash

# Author: Murat Tezgider
# Date: 2024-03-18
# Description: This script run triton server to use created model repository. Before running this script, run the following scripts sequentially: 1. install_git_and_lfs.sh 2. install_tensorrt_llm.sh 3.trendyol_llm_tensorrt_engine_build_and_test.sh 4.create_triton_model_repository.sh

tritonserver --model-repository=/tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0 --model-control-mode=explicit --load-model=preprocessing --load-model=postprocessing --load-model=tensorrt_llm --load-model=tensorrt_llm_bls --load-model=ensemble  --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1