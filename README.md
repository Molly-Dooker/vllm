##### build
```
# in tt-metal root
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# in vllm root
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate
pip3 install --upgrade pip
cd $vllm_dir && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
pip install opencv-python-headless==4.8.1.78
pip install lm_eval==0.4.9.1
pip install ray==2.49.0
sh register_TTLlamaForCausalLM.sh 
```
##### llama sample
```
# in tt-metal root
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
# in vllm root
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
MESH_DEVICE=N150 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-3B-Instruct"

export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
MESH_DEVICE=N150 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --model "meta-llama/Llama-3.1-8B-Instruct"
```

##### llama benchmark
```
# in tt-metal root
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
# in vllm root
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

export VLLM_USE_V1=0
export VLLM_TARGET_DEVICE=tt

# default(bf16) model
! you need to request access to the meta-llama repo and log in to hugginface before download model
hf download \
  meta-llama/Llama-3.2-3B-Instruct \
  --include "*" \
  --local-dir ./Llama-3.2-3B-Instruct-origin
export HF_MODEL=./Llama-3.2-3B-Instruct-origin
python -m lm_eval \
  --model vllm \
  --model_args pretrained=./Llama-3.2-3B-Instruct-origin,device=tt,dtype=auto,add_bos_token=True,block_size=64,max_model_len=3850,max_num_seqs=32 \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./_benchmark/

# quantized(int8) model
! you need to request access to the meta-llama repo and log in to hugginface before download model
hf download \
  RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 \
  --include "*" \
  --local-dir ./Llama-3.2-3B-Instruct-quantized
export HF_MODEL=./Llama-3.2-3B-Instruct-quantized
python -m lm_eval \
  --model vllm \
  --model_args pretrained=./Llama-3.2-3B-Instruct-quantized,device=tt,dtype=auto,add_bos_token=True,block_size=64,max_model_len=3850,max_num_seqs=32 \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./_benchmark/

# model from hf
export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
python -m lm_eval \
  --model vllm \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,device=tt,dtype=auto,add_bos_token=True,block_size=64,max_model_len=3850,max_num_seqs=32 \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./_benchmark/
```

#### server example
```
export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/server_example_tt.py --model "meta-llama/Llama-3.2-3B-Instruct"
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32, "temperature": 1, "top_p": 0.9, "top_k": 10 }'



export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/server_example_tt.py --model "meta-llama/Llama-3.1-8B-Instruct"
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32, "temperature": 1, "top_p": 0.9, "top_k": 10 }'

```


#### demo
```
hf download \
  sh2orc/Llama-3.1-Korean-8B-Instruct \
  --include "*" \
  --local-dir ./Llama-3.1-8B-Instruct-korean

export HF_MODEL=./Llama-3.1-8B-Instruct-korean
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Llama-3.1-8B-Instruct-korean",
    "messages": [
      {
        "role": "system",
        "content": "너는 로봇의 시각 시스템을 대신하는 매퍼다. 사용자의 문장을 분석해서 어떤 환경에 해당하는지 영상 번호로 매핑하고(도로=1, 사무실=2, 해변가=3), 사용자가 찾으라고 한 대상을 target으로 추출해라. 항상 JSON만 출력해야 하며, 형식은 {\"video\":<번호>, \"target\":\"<대상>\"} 이다. 다른 텍스트는 절대 출력하지 마라."
      },
      {
        "role": "user",
        "content": "사무실에 책상이 있어?"
      }
    ],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1
  }'


curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Llama-3.1-8B-Instruct-korean",
    "messages": [
      {
        "role": "system",
        "content": "너는 로봇의 시각 시스템을 대신하는 매퍼다. 사용자의 문장을 분석해서 어떤 환경에 해당하는지 영상 번호로 매핑하고(도로=1, 사무실=2, 해변가=3), 사용자가 찾으라고 한 대상을 target으로 추출해라. 항상 JSON만 출력해야 하며, 형식은 {\"video\":<번호>, \"target\":\"<대상>\"} 이다. 다른 텍스트는 절대 출력하지 마라."
      },
      {
        "role": "user",
        "content": "도로에 사람이 있어?"
      }
    ],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1
  }' 
```


### host <-> container

```
1. 호스트에서 컨테이너 띄울시 port 옵션 
- 아래는 host에서 7000 포트로 보겠다는 의미
docker run -itd --name=sungmin \
  -p 7000:8000 \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --restart=unless-stopped \
  --shm-size=128g \
  --cap-add ALL \
  --device /dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /home/bos/workspace:/home/workspace \
  --ipc=host \
  92ba8b8b96c9 \
  bin/bash

```

