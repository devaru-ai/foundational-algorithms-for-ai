# **Phase 1: NumPy for Concepts**


1.  **Scaled Dot-Product Attention Implementation:**
    * **Goal:** Write a Python function using NumPy that takes $Q$, $K$, and $V$ matrices and returns the attention output. You must implement the $QK^T$ dot product, the scaling by $\sqrt{d_k}$, the `softmax`, and the multiplication by $V$.

2.  **Positional Encoding Implementation:**
    * **Goal:** Write a function that generates the Positional Encoding (PE) matrix using NumPy's `sin` and `cos` functions, based on the formulas in the paper.
    * **Focus:** Ensure you correctly handle the dimension index ($2i$ and $2i+1$) and the position index ($\text{pos}$). Verify that the output matrix has the shape `(sequence_length, model_dimension)`.

# **Phase 2: PyTorch/TensorFlow for Modern Structure**

3.  **Multi-Head Attention Implementation (PyTorch `nn.Module`):**
    * **Goal:** Implement a class that takes $Q, K, V$ and performs the three steps of multi-head attention:
        * **Linear Projection:** Use `nn.Linear` layers to project $Q, K, V$ before splitting.
        * **Splitting:** Reshape the projected vectors into multiple heads.
        * **Concatenation & Final Linear Layer:** Combine the outputs of the heads and pass them through a final linear layer.

4.  **Full Transformer Layer Integration:**
    * **Goal:** Combine your custom Multi-Head Attention, Positional Encoding, and Feed-Forward Network components into a single **Transformer Layer** class.
