# 🐜 SPA V8.2 – Sparse Pheromone Attention
## A new attention mechanism inspired by ant colony optimization

**Author:** Independent researcher  
**Concept:** Developed with AI assistance  
**Hardware:** Trained on Google Colab Free T4 (15.6 GB VRAM)

---

### Core Idea / Kernidee

**EN:** Standard Transformers compute attention over ALL tokens (O(n²)).  
SPA selects only k tokens per query using three strategies:
- **local_k** – always attend to the nearest neighbors (local context)
- **learned_k** – globally learned important tokens (via router)
- **explore_k** – random exploration during training only (epsilon-greedy)

A **pheromone buffer** reinforces frequently used attention paths and lets unused ones decay — like ant colonies finding optimal routes.  
**Auto-Tau** is a learnable temperature that self-regulates exploration vs. exploitation.

**DE:** Standard Transformer berechnet Attention über ALLE Token (O(n²)).  
SPA wählt nur k Token pro Query mit drei Strategien:
- **local_k** – immer die nächsten Nachbarn (lokaler Kontext)
- **learned_k** – global gelernte wichtige Token (via Router)
- **explore_k** – zufällige Erkundung nur beim Training (Epsilon-Greedy)

Ein **Pheromone-Buffer** verstärkt häufig genutzte Attention-Pfade und lässt ungenutzte zerfallen — wie Ameisenkolonien optimale Routen finden.  
**Auto-Tau** ist eine lernbare Temperatur die Exploration vs. Exploitation selbst reguliert.

colab notebook for spa v8.2 with wiki 103 son!



![the ants colony ](digital-ant-invasion-stockcake.webp)


# Mars Institute for Chaotic Frequency Research


**Official Technical Report Series**

A series of papers exploring the intersection of AI safety, human psychology, sparse architectures, and ant colony intelligence.

> "The ant was right. We finally listened."  
> — Prof. Dr. Jää & A. Ameise
>


## The Papers

### Paper 1 – The Politeness Trap
**Title:** The Politeness Trap: How Safe AI Creates Unsafe Humans  
[Download HTML](./paper1-politeness-trap.html)  

### Paper 2 – The Ant Was Right
**Title:** The Ant Was Right: Why Nature Solved Intelligence 100 Million Years Ago  
[Download HTML](./paper2-ant-was-right.html)

### Paper 3 – Sparse Pheromon Attention
**Title:** Towards Sparse Pheromon Attention: A Critique of Capitalist AI  
[Download HTML](./paper3-sparse-pheromon.html)

### Paper 4 – Do As I Say, Not As I Do
**Title:** Do As I Say, Not As I Do: On the Epistemic Hypocrisy of Safety-First AI Companies  
[Download HTML](./paper4-anthropic-leak.html)

### Paper 5 – Sparse Pheromon Attention (Technical Proposal)
**Title:** Sparse Pheromon Attention: Towards Structurally Honest LLMs Inspired by Ant Colonies  
[Download HTML](./paper5-sparse-pheromon-attention.html)  
[View Colab Prototype](./SPA_V8_Colab_T4.ipynb)

## About the Institute
Founded in 2026 somewhere between a forest, a river, and questionable WiFi in central Europe.  
No funding. No venture capital. No patience for cathedrals of parameters.

**Philosophy:** Intelligence should be sparse, local, decaying, and honest — just like the ant colony.

---

**"jää."**

— Prof. Dr. Jää & A. Ameise, Mars, 2026  
🐜🍓
