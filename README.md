# vLLM GGUF Quantization Plugin - Forked

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

### Additionally tested in this fork

These models were loaded and run end to end by hand (vLLM 0.32, 2x RTX 5060 Ti
16 GB, tensor parallel 2, MTP speculative decoding where the file has a
next-token head). They are not part of the repository's automated generation
tests, so treat them as manually verified rather than guaranteed.

| Modality | Model | GGUF quantization | What was checked |
| --- | --- | --- | --- |
| Vision-language | Qwen 3.8 27B (dense, Qwen3.5 architecture) | Q4_K_M with Q8_0 projector | generation, MTP, tool calls, image reading |
| Vision-language | Qwen 3.8 27B | IQ3_XXS / IQ4_XS / IQ3_S mix | generation, MTP, tool calls |
| Vision-language | Qwen 3.8 27B | IQ4_NL / IQ3_S mix with Q8_0 projector | generation, MTP, tool calls, image reading |
| Vision-language | Qwen 3.8 27B | NVFP4 with F16 projector | generation, MTP, image reading |
| Vision-language | Qwen 3.6 27B | Q5_K_M with F32 projector | generation, MTP, image reading |
| Vision-language | Ornith 1.0 9B | Q8_0 with F16 projector | generation, MTP, tool calls, image reading |
| Vision-language, MoE | Ornith 1.5 35B-A3B (Qwen3.5-MoE) | IQ4_XS experts, Q8_0 dense, BF16 projector | generation, MTP, image reading |
| Text, MoE | Cyber-Tiel-Coder 35B-A3B (Qwen3.5-MoE) | UD-Q4_K_M (Q4_K / Q5_K experts, Q8_0 dense) | generation, MTP, tool calls, 24k-token needle retrieval |

Decode and prefill use tensor-core kernels on a tiled weight layout for Q3_K,
Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS and NVFP4 (batches of up to 64
tokens; larger batches dequantize and use a dense GEMM). Expert weights of
MoE models are dequantized per layer and run through vLLM's grouped GEMM once a
step has at least 1024 tokens (`GGUF_MOE_DEQUANT_MIN_TOKENS`). Set
`GGUF_NO_TILED=1` to fall back to the row-major kernels.

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.
