#!/bin/bash

# Author: Murat Tezgider
# Date: 2024-03-18
# Description: This script automates the creating triton repos and model config. Before running this script, run the following scripts sequentially: 1. install_git_and_lfs.sh 2. install_tensorrt_llm.sh 3.trendyol_llm_tensorrt_engine_build_and_test.sh


mkdir -p  /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/
cp /tensorrt/v0.8.0/tensorrtllm_backend/all_models/inflight_batcher_llm/* /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/ -r


python3 /tensorrt/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/preprocessing/config.pbtxt tokenizer_dir:/tensorrt/models/Trendyol-LLM-7b-chat-v1.0,tokenizer_type:llama,triton_max_batch_size:64,preprocessing_instance_count:1
python3 /tensorrt/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/postprocessing/config.pbtxt tokenizer_dir:/tensorrt/models/Trendyol-LLM-7b-chat-v1.0,tokenizer_type:llama,triton_max_batch_size:64,postprocessing_instance_count:1
python3 /tensorrt/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 /tensorrt/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/ensemble/config.pbtxt triton_max_batch_size:64
python3 /tensorrt/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:/tensorrt/tensorrt-models/Trendyol-LLM-7b-chat-v1.0/v0.8.0/trt-engines/fp16/1-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600


