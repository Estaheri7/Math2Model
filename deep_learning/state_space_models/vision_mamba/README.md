# Vision Mamba: A Structured State Space Model for Vision

Vision Mamba is a neural architecture designed to combine the global modeling capability of Transformers with the efficiency and recurrence of structured state space models (SSMs). It is part of the growing family of Mamba models, which leverage selective, long-sequence modeling using SSM-based recurrence instead of expensive full attention.

## Comparison with Transformers

| Feature                | Transformers     | Vision Mamba         |
|------------------------|------------------|-----------------------|
| Token Mixing           | Full Attention   | SSM-based Recurrence  |
| Inductive Bias         | None             | Recurrent + Locality  |
| Computational Cost     | Quadratic (O(L²))| Linear (O(L))         |
| Long-Range Modeling    | Explicit         | Implicit via SSM      |

## References And Orginal papers used in this notebook

- [Mamba: Linear-Time Sequence Modeling with Selective SSMs](https://arxiv.org/abs/2312.00752)
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)