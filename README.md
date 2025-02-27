<<<<<<< HEAD
# CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences

**This is the official implementation of the paper** CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences.

We conduct extensive experiments to evaluate CAKE's ability to retain accuracy and improve performance on long context LLM generation under limited cache sizes.

# Introduction

Large language models (LLMs)' proficiency in handling long sequences boosts Key-value (KV) caching demand. Recent efforts to evict KV cache have alleviated the inference burden, but they often fail to allocate resources rationally across layers with different attention patterns. In this paper, we introduce **C**ascading and **A**daptive **K**V cache **E**viction (**CAKE**), a method that significantly improves LLM inference efficiency by optimizing KV cache eviction through an adaptive cache allocation strategy implemented via a cascading cache management and an innovative eviction indicator.

We approach KV cache eviction as a "cake-slicing problem," assessing each layer's KV cache needs by considering attention dynamics in both spatial and temporal dimensions. During prompt prefilling, CAKE allocates rational cache size for layers by analyzing layer-specific KV cache preferences and manages the memory budgets with the guidance of these preferences in a cascading manner. This approach allows for a global view of cache size allocation, distributing resources adaptively based on the diverse attention mechanisms across layers.

Also, we've designed a new eviction indicator that considers the shifting importance of tokens over time, addressing a limitation in existing methods that often overlook temporal dynamics. Our comprehensive experiments on the LongBench and NeedleBench datasets show that CAKE is capable of preserving the performance of models when retaining only 3.2% KV cache and consistently outperforms current baselines across various models and memory constraints, especially in low-memory situations.

# News

# Quick Start
=======
## README的意义

README 文件通常是项目的第一个入口点。你应该通过 README 明确地告诉大家，为什么他们应该使用你的项目，以及安装和使用的方法。

如果在仅仅看文档而不看代码的情况下就可以使用你的项目，该文档就完成了。 这个非常重要，因为这将使项目的文档接口与其内部实现分开，只要接口保持不变，就可以自由更改项目的内部结构。 

**文档，而不是代码定义了项目的使用方式。**

一个规范的README文档能减少用户检索信息的时间。

## 标准 README

一个标准的README文件应当至少包含以下的内容：

- 项目背景：说明创建本项目的背景与动机，创建本项目试图解决的问题 
- 安装方法：说明如何快速上手使用该项目
- 使用方法：列出本项目能够提供的功能以及使用这些功能的方法
- 文档：现阶段antcode鼓励用户使用语雀组织项目文档，在README上应当放入项目的语雀文档链接

## 附加内容

视项目的实际情况，同样也应该包含以下内容：

- 项目特性：说明本项目相较于其他同类项目所具有的特性
- 兼容环境：说明本项目能够在什么平台上运行
- 使用示例：展示一些使用本项目的小demo
- 主要项目负责人：使用“@”标注出本项目的主要负责人，方便项目的用户沟通
- 参与贡献的方式：规定好其他用户参与本项目并贡献代码的方式
- 项目的参与者：列出项目主要的参与人
- 已知用户：列出已经在生产环境中使用了本项目的全部或部分组件的公司或组织
- 赞助者：列出为本项目提供赞助的用户
>>>>>>> bc352a0debb332bfa1cc7c84a1cc7f039231d116
