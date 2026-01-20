# 16-Week AI & LLM Engineering Masterclass

**Legend:** 📄 = Article/Blog | 🎥 = Video/Playlist/Lecture | 📖 = Course | 🎓 = Research Paper | 🔧 = Tool/Framework

---

## Week 0 – Setup + Core AI Math

### Linear Algebra Fundamentals
- [ ] 🎥 **3Blue1Brown – Essence of Linear Algebra** (15+ videos)  
  https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab  
  *Perfect intro to vectors, matrices, eigenvalues, determinants*

- [ ] 📖 **Coursera – Mathematics for Machine Learning: Linear Algebra**  
  https://www.coursera.org/learn/mathematics-machine-learning-linear-algebra  
  *Practical course with assignments and quizzes*

- [ ] 📖 **Dive into Deep Learning (D2L.ai) – Ch.2: Preliminaries**  
  http://www.d2l.ai/chapter_preliminaries/index.html  
  *Matrix operations, automatic differentiation, code examples*

### Probability & Statistics
- [ ] 🎥 **StatQuest with Josh Starmer – Statistics Fundamentals**  
  https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzn7p-6E04CmiyGIgg  
  *Expectations, variance, distributions, Bayes rule for ML*

- [ ] 📄 **Google Developers – Machine Learning Glossary (Probability section)**  
  https://developers.google.com/machine-learning/glossary  
  *Quick reference for probability concepts*

### Practical Setup (Python + GPU + Jupyter)
- [ ] 🔧 **PyTorch Installation Guide with GPU Support**  
  https://pytorch.org/get-started/locally/  
  *CUDA setup for Linux/WSL (relevant for your Arch Linux setup)*

- [ ] 📄 **GPU-Enabled Jupyter Notebooks Guide (2025)**  
  https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html  
  *Conda/uv environment management*

- [ ] 📄 **Linux GPU Troubleshooting (nvidia-smi verification)**  
  https://docs.nvidia.com/datacenter/tesla/tesla-installation-checklist/  
  *Verify GPU setup with `nvidia-smi`*

---

## Week 1 – AI Terminology + MNIST

### Machine Learning Fundamentals
- [ ] 📄 **Google ML Glossary – Core Concepts**  
  https://developers.google.com/machine-learning/glossary  
  *Loss, optimizer, epoch, overfitting, batch size, learning rate*

- [ ] 🎥 **Andrew Ng – Machine Learning Crash Course (Google)**  
  https://developers.google.com/machine-learning/crash-course  
  *~15 hour course covering supervised/unsupervised learning*

- [ ] 📖 **Fast.ai – Practical Deep Learning for Coders**  
  https://course.fast.ai/  
  *Top-down approach: build models first, understand later*

### MNIST Implementation (Hands-On)
- [ ] 📄 **PyTorch Official Tutorial – Training a Classifier (MNIST)**  
  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
  *Step-by-step PyTorch guide using MNIST*

- [ ] 🎥 **Make Language Model from Scratch Like MNIST**  
  https://tree.rocks/make-language-model-from-scratch-like-mnist-5ed59aeb538d  
  *Bridges MNIST concepts to language modeling*

- [ ] 🎥 **Coding MNIST from Scratch in PyTorch**  
  https://www.youtube.com/watch?v=Q_2PYrWdPEw  
  *Interactive coding walkthrough (Transformers & Tokenizers Explained)*

- [ ] 📖 **Hugging Face Course – NLP Fundamentals**  
  https://huggingface.co/learn/nlp-course  
  *Intro to tokenization and embeddings*

---

## Week 2 – Basics of LLMs: Tokenization, Vectorization, Attention

### High-Level LLM Overview
- [ ] 📖 **DeepLearning.AI – How Transformer LLMs Work (short course)**  
  https://www.deeplearning.ai/short-courses/how-transformer-llms-work/  
  *Tokenization, embeddings, self-attention, transformer blocks*

- [ ] 📄 **Codecademy – Transformer Architecture & Self-Attention Mechanism**  
  https://www.codecademy.com/article/transformer-architecture-self-attention-mechanism  
  *Clear visual explanation of attention mechanism*

- [ ] 🎥 **Stanford CME295 – Transformers & LLMs (Autumn 2025) – Lecture 1**  
  https://www.youtube.com/watch?v=AIiwuClvH6k  
  *Introduction to transformers and tokenization*

### Tokenization Deep Dive
- [ ] 📖 **Hugging Face LLM Course – Chapter 2: Tokenizers**  
  https://huggingface.co/learn/llm-course/en/chapter2/4  
  *BPE tokenization, WordPiece, SentencePiece explained*

- [ ] 📖 **Hugging Face LLM Course – Chapter 6: Training a New Tokenizer**  
  https://huggingface.co/learn/llm-course/en/chapter6/2  
  *Practical guide to training custom tokenizers*

- [ ] 📄 **OpenAI – Tokenization and Embeddings Overview**  
  https://platform.openai.com/docs/guides/tokens  
  *How tokens work in modern LLMs*

### Embeddings & Attention Basics
- [ ] 📄 **Sebastian Raschka – Understanding the Self-Attention Mechanism**  
  https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html  
  *Math-driven explanation with code examples*

- [ ] 🎥 **3Blue1Brown – Attention in Transformers, Step-by-Step (Deep Learning Chapter 6)**  
  https://www.youtube.com/watch?v=eMlx5fFNoYc  
  *Intuitive visual explanation of attention*

- [ ] 📄 **TrueFoundry – Transformer Architecture Deep Dive**  
  https://www.truefoundry.com/blog/transformer-architecture  
  *Encoder, decoder, residual connections, normalization*

---

## Week 3 – Deep Dive: QKV, Self/Cross/Multi-Head Attention

### Q, K, V Matrix Math
- [ ] 📄 **Writing an LLM from Scratch, Part 5 – More on Self-Attention**  
  https://www.gilesthomas.com/2025/01/llm-from-scratch-5-self-attention  
  *QKV shapes, scaling factor, attention computation*

- [ ] 🎥 **Visualizing Transformers and Attention – TNG Big Tech Day Talk**  
  https://www.youtube.com/watch?v=KJtZARuO3JY  
  *Interactive visualization of Q/K/V transformations*

### Multi-Head Attention (MHA)
- [ ] 📄 **LLM Transformer Model Visually Explained (Interactive)**  
  https://poloclub.github.io/transformer-explainer/  
  *Play with attention heads interactively*

- [ ] 🎥 **Deep Learning Fundamentals – Multi-Head Attention Implementation**  
  https://www.youtube.com/watch?v=eMlx5fFNoYc  
  *Coding multi-head attention in PyTorch*

- [ ] 📖 **Hugging Face Course – Chapter 3: Fine-Tuning (attention sections)**  
  https://huggingface.co/learn/llm-course/en/chapter3  
  *Hands-on attention implementation*

### Cross-Attention (Encoder-Decoder)
- [ ] 📄 **DataCamp – How Transformers Work (encoder-decoder section)**  
  https://www.datacamp.com/tutorial/how-transformers-work  
  *Cross-attention in seq2seq and multimodal models*

- [ ] 🎥 **Attention Explained: Self vs Cross Attention**  
  https://www.youtube.com/results?search_query=self+attention+vs+cross+attention  
  *Short explainer comparing attention types*

### Foundational Paper
- [ ] 🎓 **"Attention Is All You Need" (Vaswani et al., 2017)**  
  https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf  
  *The original Transformer paper – foundational reading*

---

## Week 4 – LLM Coding: Causal Masking + Code GPT

### Causal (Autoregressive) Masking
- [ ] 📄 **From-Scratch Language Model Tutorial – Causal Masking**  
  https://tree.rocks/make-language-model-from-scratch-like-mnist-5ed59aeb538d  
  *Next-token prediction, causal masking explained*

- [ ] 🎥 **Implementing Causal Self-Attention in PyTorch**  
  https://www.youtube.com/watch?v=Q_2PYrWdPEw  
  *Step-by-step code walkthrough*

### Build a Tiny GPT-Like Model
- [ ] 📖 **Hugging Face LLM Course – Building Language Models from Scratch**  
  https://huggingface.co/learn/llm-course/en  
  *Full LM training pipeline (tokenization → training → evaluation)*

- [ ] 🎥 **Andrej Karpathy – Let's Build GPT: From Scratch in Code**  
  https://www.youtube.com/watch?v=kCc8FmEb1nY  
  *1h56m walkthrough building GPT-2 from scratch*

- [ ] 🔧 **NanoGPT: Minimal GPT Implementation**  
  https://github.com/karpathy/nanoGPT  
  *Tiny, clean PyTorch implementation (130 lines)*

### Code LLMs Overview
- [ ] 📄 **Code LLM Training Practices & Tokenization Differences**  
  https://huggingface.co/blog/code-models  
  *How code models differ from text models*

- [ ] 🎥 **Using Code Llama & GPT-4 for Code Generation (Practical)**  
  https://www.youtube.com/results?search_query=code+llama+tutorial  
  *Hands-on setup and usage*

---

## Week 5 – Think Like an Engineer: Training Massive Models

### Large-Scale Pretraining Pipeline
- [ ] 📄 **Aahil Mehta – Training LLMs from Scratch (10B+ scale)**  
  https://www.aahilm.com/blog/training-llms-from-scratch  
  *Data pipeline, compute requirements, checkpointing*

- [ ] 📖 **Hugging Face Course – Chapter 9: Training Large Models**  
  https://huggingface.co/learn/llm-course/en/chapter9  
  *Multi-GPU training, distributed strategies*

- [ ] 🎥 **How Large Language Models Are Trained (Full Talk)**  
  https://www.youtube.com/results?search_query=how+large+language+models+trained  
  *Infrastructure, data, scaling laws explained*

### Distributed Training & Systems
- [ ] 📄 **OpenAI – Techniques for Training Large Neural Networks**  
  https://openai.com/index/techniques-for-training-large-neural-networks/  
  *Data parallelism, model parallelism, pipeline parallelism, Megatron-LM*

- [ ] 🔧 **DeepSpeed Tutorial – ZeRO, Offloading, Pipeline Parallelism**  
  https://www.deepspeed.ai/  
  *Industry-standard distributed training library*

- [ ] 📄 **Scaling Laws & Compute-Optimal Training**  
  https://arxiv.org/abs/2203.15556  
  *Chinchilla scaling laws (tokens vs. parameters tradeoff)*

---

## Week 6 – Optimization Hacks: KV Caching, Quantization, LoRA

### KV Caching & Inference Optimization
- [ ] 📄 **Sebastian Raschka – Understanding & Coding the KV Cache**  
  https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms  
  *KV cache mechanics, implementation, speed gains*

- [ ] 📄 **DataScienceDojo – KV Cache: How to Speed Up LLM Inference**  
  https://datasciencedojo.com/blog/kv-cache-how-to-speed-up-llm-inference/  
  *Practical optimization techniques for production*

- [ ] 📄 **RedHat – Master KV Cache Aware Routing (llm-d)**  
  https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference  
  *Advanced: KV cache-aware load balancing*

- [ ] 📄 **NVIDIA – Mastering LLM Techniques: Inference Optimization**  
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/  
  *GPU-level optimization strategies*

- [ ] 🎥 **How KV Caching Makes LLMs Fast (Visual Explanation)**  
  https://www.youtube.com/results?search_query=kv+cache+llm+inference  
  *Quick video walkthrough*

### Quantization (4-bit, 8-bit, GGUF)
- [ ] 📄 **Maarten Grootendorst – A Visual Guide to Quantization**  
  https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization  
  *60+ illustrations explaining quantization methods*

- [ ] 📄 **The Register – Beginner's Guide to Quantization**  
  https://www.theregister.com/2024/07/14/quantization_llm_feature/  
  *Hands-on: GGUF, llama.cpp, Q4_0 quantization*

- [ ] 📄 **Symbl.ai – A Guide to Quantization in LLMs**  
  https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/  
  *GPTQ vs GGUF vs BitsAndBytes comparison*

- [ ] 🎥 **Quantize Any LLM with GGUF and llama.cpp**  
  https://www.youtube.com/watch?v=wxQgGK5K0rE  
  *Step-by-step conversion to 4-bit GGUF*

- [ ] 📄 **Reddit – Overview of GGUF Quantization Methods**  
  https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/  
  *Community guide to different quant methods*

### LoRA & Parameter-Efficient Fine-Tuning
- [ ] 📖 **Hugging Face LLM Course – Chapter 11: LoRA**  
  https://huggingface.co/learn/llm-course/en/chapter11/4  
  *LoRA configuration, rank, alpha parameters*

- [ ] 📄 **Databricks – Efficient Fine-Tuning with LoRA for LLMs**  
  https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms  
  *QLoRA + bitsandbytes practical guide*

- [ ] 📄 **Philipp Schmid – How to Fine-Tune Open LLMs in 2025**  
  https://www.philschmid.de/fine-tune-llms-in-2025  
  *QLoRA, Spectrum, distributed training with DeepSpeed*

- [ ] 🎥 **Fine-Tune LLMs with Hugging Face & PyTorch**  
  https://www.youtube.com/watch?v=bZcKYiwtw1I  
  *Complete tutorial: setup → training → evaluation*

- [ ] 🔧 **Hugging Face TRL – Supervised Fine-Tuning Trainer**  
  https://huggingface.co/docs/trl/sft_trainer  
  *SFTTrainer for efficient fine-tuning*

---

## Week 7 – The RAG Problem: Chunking, Reranking, Vector DBs

### RAG Conceptual Overview
- [ ] 📄 **SingleStore – RAG Tutorial: Beginner's Guide**  
  https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/  
  *RAG architecture, retrieve → augment → generate flow*

- [ ] 📄 **AMD ROCm Blog – RAG Pipeline with vLLM, LangChain, Chroma**  
  https://rocm.blogs.amd.com/artificial-intelligence/rag-pipeline-vllm/README.html  
  *End-to-end RAG implementation with code*

- [ ] 🎥 **RAG Systems Explained (High-Level Overview)**  
  https://www.youtube.com/results?search_query=retrieval+augmented+generation+tutorial  
  *Architecture walkthrough*

### Chunking & Semantic Search
- [ ] 📄 **LangChain – Document Splitting Strategies**  
  https://python.langchain.com/docs/modules/data_connection/document_loaders/  
  *Semantic chunking, fixed-size chunking, recursive splitting*

- [ ] 📄 **Chunking Strategies for RAG Systems**  
  https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/  
  *When to use different chunking methods*

- [ ] 📖 **Hugging Face Course – Chapter 7: Semantic Search & Retrieval**  
  https://huggingface.co/learn/llm-course/en/chapter7/7  
  *Embeddings, similarity search, hybrid retrieval*

### Vector Databases
- [ ] 🔧 **Chroma – Vector DB for RAG**  
  https://www.trychroma.com/  
  *Lightweight, in-memory vector store (good for learning)*

- [ ] 🔧 **FAISS – Facebook AI Similarity Search**  
  https://github.com/facebookresearch/faiss  
  *Production-scale vector search library*

- [ ] 🔧 **Milvus – Open-source Vector Database**  
  https://milvus.io/  
  *Enterprise-ready, distributed vector DB*

- [ ] 📄 **Pinecone – What is a Vector Database?**  
  https://www.pinecone.io/learn/vector-database/  
  *Conceptual overview of vector DBs*

### Reranking & Retrieval Quality
- [ ] 📄 **LangChain – Rerankers & Query Optimization**  
  https://python.langchain.com/docs/modules/data_connection/retrievers/  
  *Cross-encoder rerankers, hybrid search*

- [ ] 📖 **LlamaIndex – Advanced Retrieval & Reranking**  
  https://docs.llamaindex.ai/en/latest/module_guides/retrieval/retrieval/  
  *Reranking with cross-encoders (BGE-Reranker, etc.)*

---

## Week 8 – The RAG Code: Safety, Guardrails, Code RAG

### Safety & Guardrails
- [ ] 📄 **François Chollet – How I Think About LLM Prompt Engineering**  
  https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering  
  *Safety, reliability, and guardrails*

- [ ] 📄 **LangChain – Safety & Guardrails for LLM Apps**  
  https://python.langchain.com/docs/guides/safety/  
  *Input filtering, output validation, policy enforcement*

- [ ] 📖 **Anthropic – Constitutional AI for Safety**  
  https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback  
  *Safety alignment methods beyond RLHF*

- [ ] 📄 **Anthropic's Approach to Safe AI (Blog)**  
  https://aishwaryasrinivasan.substack.com/p/anthropics-approach-to-safe-ai  
  *Guardrails, jailbreaking resistance, content moderation*

### Code RAG Implementation
- [ ] 📄 **LangChain RAG Tutorial**  
  https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/langchain-rag-implementation  
  *Complete RAG implementation walkthrough*

- [ ] 📖 **LangChain Docs – RAG Agents**  
  https://docs.langchain.com/oss/python/langchain/rag  
  *Building Q&A over documents (code repos, PDFs, etc.)*

- [ ] 🎥 **Building Code RAG with LangChain**  
  https://www.youtube.com/results?search_query=code+rag+codebase+langchain  
  *Query codebase with natural language*

- [ ] 📄 **Together AI – Building RAG with LangChain**  
  https://www.together.ai/blog/rag-tutorial-langchain  
  *Integration patterns and best practices*

---

## Week 9 – AI Agents: ReAct, Tool Calling, LangChain, LangGraph

### ReAct Pattern & Tool Calling
- [ ] 📄 **Machine Learning Mastery – Building ReAct Agents with LangGraph**  
  https://machinelearningmastery.com/building-react-agents-with-langgraph-a-beginners-guide/  
  *ReAct cycle (Reason → Act → Observe) explained*

- [ ] 📖 **LangChain Docs – Agents Overview**  
  https://docs.langchain.com/oss/python/langchain/agents  
  *Tool binding, agent types, execution flow*

- [ ] 🎥 **ReAct Agents and Tool Calling with OpenAI / LangChain**  
  https://www.youtube.com/results?search_query=react+agent+tool+calling+langchain  
  *Practical implementation walkthrough*

### LangChain Basics
- [ ] 📖 **LangChain Documentation – Getting Started**  
  https://python.langchain.com/docs/get_started/introduction  
  *Prompts, chains, tools, memory*

- [ ] 📖 **LangChain Course (official)**  
  https://learn.langchain.com/  
  *Structured learning path with exercises*

- [ ] 🎥 **LangChain Basics: Prompts, Chains, Tools**  
  https://www.youtube.com/results?search_query=langchain+tutorial+basics  
  *Short courses on fundamentals*

### LangGraph & State Management
- [ ] 📖 **LangGraph Documentation – How to Create a ReAct Agent from Scratch**  
  https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/  
  *StateGraph, nodes, edges, conditional routing*

- [ ] 🎥 **LangGraph Advanced – Flow Engineering & Agentic RAG**  
  https://www.youtube.com/watch?v=5TLQdNM5pHg  
  *Databricks talk on complex agent workflows*

- [ ] 📖 **Google – ReAct Agent with Gemini 2.5 & LangGraph**  
  https://ai.google.dev/gemini-api/docs/langgraph-example  
  *Modern agent implementation with latest APIs*

- [ ] 🎥 **LangGraph Complete Implementation Guide 2025**  
  https://www.youtube.com/results?search_query=langgraph+tutorial+2025  
  *State persistence, checkpointing, debugging*

- [ ] 📖 **LangChain & LangGraph Tutorials Playlist**  
  https://www.youtube.com/playlist?list=PLAMHV77MSKJ7Pn_OwuGzbDPs_MOibBRP-  
  *Curated playlist for RAG to Multi-Agent systems*

---

## Week 10 – Context Engineering: Memory, MCP, Multi-Agents

### Memory Systems for Conversational AI
- [ ] 📄 **PromptEngineering.org – Memory, Context, and Cognition in LLMs**  
  https://promptengineering.org/memory-context-and-cognition-in-llms/  
  *Context window mechanics, sliding windows, memory strategies*

- [ ] 📄 **Content Whale – Context Engineering for LLM Memory**  
  https://content-whale.com/blog/llm-context-engineering-information-retention/  
  *Priority-based hierarchies, token management, adaptive allocation*

- [ ] 📄 **Mezmo – Context Engineering: Best Practices**  
  https://www.mezmo.com/learn-observability/context-engineering-how-to-deliver-the-right-data-to-llms  
  *Prompt engineering vs context engineering distinction*

- [ ] 📖 **LangChain Docs – Context Engineering in Agents**  
  https://docs.langchain.com/oss/python/langchain/context-engineering  
  *System prompts, memory management, state handling*

- [ ] 🎥 **Building Memory into Chatbots (LangChain/LangGraph)**  
  https://www.youtube.com/results?search_query=chatbot+memory+langchain  
  *Conversation history, long-term memory patterns*

### Multi-Agent Patterns
- [ ] 📄 **LangChain – Multi-Agent Systems & Orchestration**  
  https://python.langchain.com/docs/modules/agents/agent_types/openai_multi_functions_agent  
  *Router agents, hierarchical agents, specialist agents*

- [ ] 🎥 **Basics to Advanced Multi-Agent AI Chatbot (with Code)**  
  https://www.youtube.com/watch?v=60XDTWhklLA  
  *Complete multi-agent implementation walkthrough*

- [ ] 📖 **LangGraph – Multi-Agent Workflows**  
  https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/  
  *Coordinating multiple agents, shared context*

### MCP (Model Context Protocol)
- [ ] 🔧 **Anthropic – Model Context Protocol Spec**  
  https://modelcontextprotocol.io/  
  *Standard for LLM-tool interaction*

- [ ] 📖 **MCP Documentation & Examples**  
  https://github.com/modelcontextprotocol/python-sdk  
  *Building MCP servers and clients*

---

## Week 11 – AI Engineering: Evals, Tradeoffs, Fine-Tuning vs RAG vs Prompting

### LLM Evaluation Frameworks
- [ ] 📄 **Codecademy – Build an LLM Evaluation Framework**  
  https://www.codecademy.com/article/build-an-llm-evaluation-framework  
  *Metrics (BLEU, ROUGE, BERTScore), benchmarks, evaluation methods*

- [ ] 📖 **LangSmith Evaluation Documentation**  
  https://docs.langchain.com/langsmith/evaluation  
  *Datasets, evaluators, experiments, comparison*

- [ ] 📖 **LangSmith – How to Evaluate an LLM Application**  
  https://docs.langchain.com/langsmith/evaluate-llm-application  
  *Code examples for defining custom evaluators*

- [ ] 🎥 **How to Evaluate Your LLM Application with LangSmith**  
  https://www.youtube.com/watch?v=9pvSREkUXCk  
  *Walkthrough of LangSmith evaluation workflow*

- [ ] 📖 **LangChain – Harden Your Application with Evals**  
  https://www.langchain.com/evaluation  
  *Offline evaluation, regression testing*

- [ ] 🎥 **365 Data Science – AI Engineer Job Outlook 2025**  
  https://365datascience.com/career-advice/career-guides/ai-engineer-job-outlook-2025/  
  *Industry metrics and evaluation trends*

### Fine-Tuning vs RAG vs Prompting Tradeoffs
- [ ] 📄 **François Chollet – How I Think About LLM Prompt Engineering**  
  https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering  
  *When prompting is enough vs when to use RAG/fine-tuning*

- [ ] 📖 **Hugging Face Course – Chapter 11: Choosing Your Method**  
  https://huggingface.co/learn/llm-course/en/chapter11  
  *Decision framework for fine-tuning, RAG, prompting*

- [ ] 📄 **Blog: Fine-Tuning vs RAG vs Prompting Decision Matrix**  
  https://www.youtube.com/results?search_query=fine+tuning+vs+rag+vs+prompting+decision  
  *Practical decision framework*

- [ ] 🎥 **Short Talk: When to Fine-Tune vs RAG vs Prompt**  
  https://www.youtube.com/results?search_query=fine+tuning+vs+rag+comparison  
  *Quick comparison video*

### MLOps & Production Considerations
- [ ] 📖 **Hugging Face Course – Chapter 13: MLOps for LLMs**  
  https://huggingface.co/learn/llm-course/en/chapter13  
  *Monitoring, versioning, deployment*

- [ ] 📄 **AI Engineering Production Checklist**  
  https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/  
  *Cost controls, latency targets, reliability*

---

## Week 12 – Thinking Models: Reasoning, Chain of Thought

### Chain of Thought & Reasoning Strategies
- [ ] 📄 **PromptEngineering.org – Leveraging CoT for Deeper Insights**  
  https://promptengineering.org/memory-context-and-cognition-in-llms/  
  *Step-by-step reasoning, guided problem-solving*

- [ ] 📖 **Hugging Face Course – Chapter 12: Reasoning & CoT**  
  https://huggingface.co/learn/llm-course/en/chapter12  
  *Chain of Thought, self-consistency, tree of thought*

- [ ] 🎥 **Reasoning with LLMs: Chain of Thought & Beyond**  
  https://www.youtube.com/results?search_query=chain+of+thought+llm+reasoning+tutorial  
  *Practical implementation examples*

### Self-Critique & Iterative Refinement
- [ ] 📄 **Building Reasoning-Centric Agents with CoT**  
  https://www.youtube.com/results?search_query=self+critique+agent+llm  
  *Self-checking, error correction patterns*

- [ ] 🎥 **Building Reasoning Agents with LangGraph**  
  https://www.youtube.com/results?search_query=reasoning+agent+langgraph+tutorial  
  *Multi-step reasoning workflows*

### Thinking Models (O1 Style)
- [ ] 📄 **OpenAI – o1 Model & Reasoning Capabilities**  
  https://openai.com/o1/  
  *Extended thinking, complex problem-solving*

- [ ] 🎥 **How to Use OpenAI's o1 Model**  
  https://www.youtube.com/results?search_query=openai+o1+reasoning+model+tutorial  
  *Practical API usage*

---

## Week 13 – Multimodal: Images, Video, CLIP, Diffusion

### Vision Transformers (ViT)
- [ ] 📄 **V7 Labs – Vision Transformer: What It Is & How It Works**  
  https://www.v7labs.com/blog/vision-transformer-guide  
  *ViT architecture, patch embeddings, self-attention*

- [ ] 📖 **Building a Vision Transformer from Scratch: Comprehensive Guide**  
  https://atalupadhyay.wordpress.com/2025/03/01/building-a-vision-transformer-from-scratch-a-comprehensive-guide/  
  *SigLIP, patch embeddings, multi-head attention implementation*

- [ ] 🎥 **Vision Transformer Quick Guide – Theory & Code (15 min)**  
  https://www.youtube.com/watch?v=j3VNqtJUoz0  
  *Image patching, positional embeddings, transformer encoder*

- [ ] 🎥 **Vision Transformer from Scratch Tutorial (59 min)**  
  https://www.youtube.com/watch?v=4XgDdxpXHEQ  
  *CLIP, SigLIP, image preprocessing, patch embeddings*

- [ ] 📖 **DataCamp – Vision Transformers Tutorial**  
  https://www.datacamp.com/tutorial/vision-transformers  
  *ViT architecture and when to use ViTs*

### CLIP & Text-Image Alignment
- [ ] 📄 **LearnOpenCV – Training a CLIP Model from Scratch**  
  https://learnopencv.com/clip-model/  
  *CLIP architecture, text-image pairing, contrastive learning*

- [ ] 📄 **Huyen Chip – Multimodality and Large Multimodal Models**  
  https://huyenchip.com/2023/10/10/multimodal.html  
  *CLIP embedding space, text-to-image applications*

### Diffusion Models
- [ ] 📄 **SuperAnnotate – Introduction to Diffusion Models**  
  https://www.superannotate.com/blog/diffusion-models  
  *Forward/reverse process, noise scheduling, stability*

- [ ] 📄 **Encord – Stable Diffusion 3: Multimodal Diffusion Transformer**  
  https://encord.com/blog/stable-diffusion-3-text-to-image-model/  
  *SD3 architecture, flow matching, MMDiT*

- [ ] 🎥 **Stable Diffusion from First Principles (Engineering-Focused)**  
  https://www.youtube.com/results?search_query=stable+diffusion+from+scratch+tutorial  
  *Practical diffusion model explanation*

- [ ] 📄 **Milvus – What is Multi-Modal Diffusion Modeling?**  
  https://milvus.io/ai-quick-reference/what-is-multimodal-diffusion-modeling  
  *Cross-attention for multimodal generation*

---

## Week 14 – Capstone Project: Build Your Own AI Project

### Project Ideation & Scoping
- [ ] 📄 **How to Scope an LLM Project from Idea to MVP**  
  https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/  
  *Feasibility, MVP definition, success metrics*

- [ ] 📖 **Hugging Face Course – Chapter 14: Building LLM Applications**  
  https://huggingface.co/learn/llm-course/en/chapter14  
  *Project templates and examples*

### End-to-End LLM App Build
- [ ] 🎥 **End-to-End LLM App Build (RAG or Agent)**  
  https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/langchain-rag-implementation  
  *Choose RAG, agent, or hybrid approach based on your idea*

- [ ] 📖 **LangChain – Building with LangChain**  
  https://python.langchain.com/docs/use_cases  
  *Pre-built templates: Q&A, RAG, agents, multimodal*

### Production Deployment
- [ ] 🎥 **Deploying an LLM/RAG App to Production**  
  https://www.youtube.com/results?search_query=deploy+llm+docker+kubernetes+production  
  *Docker containerization, cloud deployment (GCP/AWS/Azure)*

- [ ] 📄 **FastAPI + Docker for LLM Services**  
  https://fastapi.tiangolo.com/  
  *API framework for LLM applications*

- [ ] 📖 **Hugging Face Spaces – Deploy for Free**  
  https://huggingface.co/spaces  
  *Quick deployment option for demos*

- [ ] 📄 **Production Checklist (Infrastructure, Monitoring, Cost)**  
  https://huggingface.co/learn/llm-course/en/chapter13  
  *Ops considerations for production systems*

### Documentation & Sharing
- [ ] 📄 **How to Write Technical Documentation**  
  https://www.writethedocs.org/  
  *Best practices for project documentation*

- [ ] 🔧 **GitHub + README Best Practices**  
  https://www.makeareadme.com/  
  *Creating compelling project READMEs*

---

## Week 15 – Career Goals: Moving into AI Engineering

### Understanding the AI Engineer Role
- [ ] 📄 **Yochana – AI Career Guide 2025**  
  https://www.yochana.com/ai-career-guide-2025/  
  *Top roles: ML Engineer, AI Research Scientist, AI Product Manager*

- [ ] 📄 **CultivatedCulture – AI Career Path Guide**  
  https://cultivatedculture.com/ai-career-path-guide/  
  *Entry → Mid → Senior progression, salary benchmarks*

- [ ] 📄 **Pluralsight – AI Career Paths: 2026 Job Guide**  
  https://www.pluralsight.com/resources/blog/ai-and-data/ai-career-guide-2025  
  *ML Engineering, Data Science, AI Research, MLOps, AI Ethics*

- [ ] 📄 **365 Data Science – AI Engineer Job Outlook 2025**  
  https://365datascience.com/career-advice/career-guides/ai-engineer-job-outlook-2025/  
  *Skills breakdown (NLP 19.7%, Prompt Engineering 8.9%, Speech 2%)*

### Building Your AI Portfolio
- [ ] 📄 **François Chollet – Building an LLM Portfolio**  
  https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering  
  *Public projects, technical writeups, open-source contributions*

- [ ] 📄 **How to Get an AI Engineering Job**  
  https://www.youtube.com/results?search_query=ai+engineering+job+portfolio+github  
  *Portfolio building strategies*

- [ ] 🔧 **GitHub – Showcase Your Projects**  
  https://github.com/  
  *Make your capstone project public with strong documentation*

- [ ] 🎥 **Building a Strong AI/ML Portfolio in 2025**  
  https://www.youtube.com/results?search_query=ai+engineer+portfolio+projects+2025  
  *Portfolio project ideas and strategies*

### Interview Preparation
- [ ] 📄 **Designing Your Roadmap: Software/DevOps → AI Engineer**  
  https://www.youtube.com/results?search_query=software+engineer+transition+to+ai+engineer  
  *Leveraging your existing skills*

- [ ] 🎥 **LLM & AI System Design Interview Questions**  
  https://www.youtube.com/results?search_query=ai+engineer+system+design+interview  
  *Common interview patterns*

- [ ] 📖 **LeetCode + Interview Prep (ML-focused)**  
  https://leetcode.com/  
  *Coding interview preparation*

### Continuous Learning & Community
- [ ] 🔧 **Reddit – r/learnmachinelearning, r/LanguageTechnology**  
  https://www.reddit.com/r/learnmachinelearning/  
  *Community Q&A, project ideas, job discussions*

- [ ] 📖 **Papers with Code – SOTA Leaderboards**  
  https://paperswithcode.com/  
  *Latest research implementations*

- [ ] 📖 **Hugging Face Hub – Model & Dataset Exploration**  
  https://huggingface.co/models  
  *Trending models, datasets, community projects*

- [ ] 🎥 **ArXiv Sanity – Latest ML Papers**  
  https://arxiv-sanity.com/  
  *Curated recent research papers*

---

## Supplementary: Foundational Resources (Throughout Course)

### Andrej Karpathy – Neural Networks: Zero to Hero (Highly Recommended)
- [ ] 🎥 **The Spelled-Out Intro to Neural Networks & Backpropagation**  
  https://www.youtube.com/watch?v=VMj-3S1tku0  
  *2h25m – Deep understanding of gradient descent (micrograd)*

- [ ] 🎥 **The Spelled-Out Intro to Language Modeling**  
  https://www.youtube.com/watch?v=PaCmpygxKno  
  *Building character-level language models (makemore)*

- [ ] 🎥 **Let's Build GPT: From Scratch in Code**  
  https://www.youtube.com/watch?v=kCc8FmEb1nY  
  *1h56m – Full GPT-2 implementation explained*

- [ ] 📖 **Neural Networks: Zero to Hero Playlist**  
  https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ  
  *Complete series (lectures 1-6)*

### DeepMind & Anthropic Research
- [ ] 🎥 **DeepMind x UCL – Attention & Memory in Deep Learning**  
  https://www.youtube.com/watch?v=AIiwuClvH6k  
  *1h36m – Alex Graves on attention mechanisms*

- [ ] 🎓 **Anthropic – Constitutional AI: Harmlessness from AI Feedback**  
  https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback  
  *Core safety alignment methodology*

- [ ] 🎓 **Anthropic – Core Views on AI Safety (PDF)**  
  https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf  
  *Deep dive into RLHF and Constitutional AI methods*

---

## Learning Tracker Template

Copy this into your notes file for weekly tracking:

```
# Learning Progress Log

## Week X: [Topic]

**Target**: [Main goal]
**Completed Resources**: 
- [ ] Resource 1: [Link] ✅ [Date]
- [ ] Resource 2: [Link] ⏳ [Date started]

**What I Built**:
[Describe project/code/experiment]

**Key Takeaways**:
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

**Open Questions/Gaps**:
- [ ] [Question 1]
- [ ] [Question 2]

**Next Week Preview**:
[What's coming next]
```

---

## Pro Tips for Your Learning Path

### 1. **Focus on Hands-On Coding**
- Don't just watch videos; implement everything yourself
- Start small (MNIST → tiny GPT → full RAG system)
- Use your Linux/Git skills to track experiments

### 2. **Prioritize Depth Over Breadth**
- Complete one resource thoroughly rather than skimming many
- Implement concepts in PyTorch before moving on
- Build projects that combine multiple concepts

### 3. **Join Communities**
- Participate in r/learnmachinelearning, r/LanguageTechnology
- Share your projects on GitHub
- Engage with Hugging Face Hub discussions

### 4. **Stay Current (2025)**
- Follow arxiv-sanity.com for latest papers
- Subscribe to Hugging Face Blog
- Watch for new tools (LangGraph, MCP evolving rapidly)

### 5. **Leverage Your Background**
- You have strong Linux/DevOps skills → focus on MLOps
- Use your networking knowledge → understand distributed training
- Your Docker/containerization expertise → production deployment

### 6. **Build in Public**
- Document your learning journey
- Push code to GitHub weekly
- Write blog posts explaining what you learned
- This becomes your portfolio for AI jobs

---

## Repository Structure Recommendation

```
ai-learning-journey/
├── week_0_math/
│   ├── linear_algebra_exercises.ipynb
│   ├── notes.md
│   └── resources_completed.txt
├── week_1_mnist/
│   ├── mnist_classifier.py
│   ├── experiments/
│   └── README.md
├── week_2_tokenization/
│   ├── tokenizer_exploration.ipynb
│   └── custom_tokenizer.py
├── ...
├── week_14_capstone/
│   ├── project_idea.md
│   ├── src/
│   ├── notebooks/
│   ├── README.md
│   └── deployment/
└── learning_log.md
```

**Use Git to track your progress, push weekly, and build credibility for job applications.**

---

## Estimated Time Investment

- **Week 0-2**: 40-50 hours (foundations)
- **Week 3-6**: 50-60 hours (core transformer concepts + optimization)
- **Week 7-10**: 60-70 hours (RAG + agents + advanced patterns)
- **Week 11-13**: 50-60 hours (evals, reasoning, multimodal)
- **Week 14-15**: 80-100 hours (capstone project + job prep)

**Total: ~500 hours of focused learning over 16 weeks**

---

**Last Updated:** January 2026 | **Resources Verified:** Week 0 through Week 15
