#!/bin/bash

pip install -e .

sed -i '1161s/^[ \t]*logits = logits.float()/#&/' /usr/local/lib/python3.12/dist-packages/transformers/models/llama/modeling_llama.py
sed -i '1044s/^[ \t]*logits = logits.float()/#&/' /usr/local/lib/python3.12/dist-packages/transformers/models/mistral/modeling_mistral.py
sed -i '1069s/^[ \t]*logits = logits.float()/#&/' /usr/local/lib/python3.12/dist-packages/transformers/models/qwen2/modeling_qwen2.py