"""Model adapters for each arm of the benchmark.

Heavy dependencies (torch, transformers, funasr, speechbrain, requests) are
imported lazily inside each adapter so the base ``oruk-bench`` install stays
light. Install extras to run a given arm:

  pip install "oruk-bench[open]"      # open-source SER models
  pip install "oruk-bench[audiollm]"  # open-weight audio-LLMs
  pip install "oruk-bench[api]"       # OpenAI / Anthropic API arms
"""
