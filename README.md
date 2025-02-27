# CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences

**This is the official implementation of the paper** CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences.

We conduct extensive experiments to evaluate CAKE's ability to retain accuracy and improve performance on long context LLM generation under limited cache sizes.

# Introduction

Large language models (LLMs)' proficiency in handling long sequences boosts Key-value (KV) caching demand. Recent efforts to evict KV cache have alleviated the inference burden, but they often fail to allocate resources rationally across layers with different attention patterns. In this paper, we introduce **C**ascading and **A**daptive **K**V cache **E**viction (**CAKE**), a method that significantly improves LLM inference efficiency by optimizing KV cache eviction through an adaptive cache allocation strategy implemented via a cascading cache management and an innovative eviction indicator.

We approach KV cache eviction as a "cake-slicing problem," assessing each layer's KV cache needs by considering attention dynamics in both spatial and temporal dimensions. During prompt prefilling, CAKE allocates rational cache size for layers by analyzing layer-specific KV cache preferences and manages the memory budgets with the guidance of these preferences in a cascading manner. This approach allows for a global view of cache size allocation, distributing resources adaptively based on the diverse attention mechanisms across layers.

Also, we've designed a new eviction indicator that considers the shifting importance of tokens over time, addressing a limitation in existing methods that often overlook temporal dynamics. Our comprehensive experiments on the LongBench and NeedleBench datasets show that CAKE is capable of preserving the performance of models when retaining only 3.2% KV cache and consistently outperforms current baselines across various models and memory constraints, especially in low-memory situations.

# News

# Quick Start
