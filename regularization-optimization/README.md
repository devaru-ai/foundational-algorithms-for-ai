# **I. L2 Regularization (Weight Decay) Implementation**

**Focus:** Understanding the dual implementation methods: automatic via the optimizer (PyTorch) versus declarative via the layer (TensorFlow/Keras).

1.  **PyTorch (Automatic/Optimizer Control):**
    * **Goal:** Build a training loop and apply L2 regularization by initializing the `torch.optim.Adam` optimizer with a specified `weight_decay` parameter (e.g., `weight_decay=0.001`).
    * **Key Learning:** Confirming that the L2 penalty is automatically applied to the parameter gradients by the optimizer's internal logic, thereby keeping the loss function code clean.

2.  **TensorFlow/Keras (Declarative/Layer Control):**
    * **Goal:** Define a `tf.keras.Sequential` model and add L2 regularization by setting the `kernel_regularizer` argument directly within a `tf.keras.layers.Dense` layer (e.g., `kernel_regularizer=regularizers.l2(0.01)`).
    * **Key Learning:** Understanding how TensorFlow manages the regularization penalty by calculating and adding the L2 loss component behind the scenes based on layer declarations.

# **II. L1 Regularization (Lasso) Implementation**

**Focus:** Mastering the manual calculation of the L1 norm ($\sum |w|$) and integrating it directly into the loss function, as this is typically *not* automated.

1.  **PyTorch (Manual Loss Calculation):**
    * **Goal:** In the training loop, manually iterate over all trainable parameters (model.parameters()), calculate the sum of their absolute values (torch.abs().sum()), and then add the scaled penalty term ($\lambda \cdot \text{L1\ norm}$) to the standard loss before the loss.backward() call.
    * **Key Learning:** Proving you know the exact mathematical term for L1 regularization and how to manually inject it into the computation graph for autodifferentiation.

2.  **TensorFlow/Keras (Declarative/Layer Control):**
    * **Goal:** Implement L1 regularization by using the `kernel_regularizer` argument within a `tf.keras.layers.Dense` layer, specifying `regularizers.l1(0.01)`.
    * **Key Learning:** Observing the declarative simplicity in Keras for L1/L2 and contrasting it with the manual work required in PyTorch.


# **III. Dropout Implementation**

**Focus:** Correctly placing the Dropout layer in the network architecture and managing its behavior during training vs. evaluation.

1.  **Layer Injection (PyTorch & Keras):**
    * **Goal:** Build a multi-layer model and insert the framework's `Dropout` layer (`nn.Dropout` or `tf.keras.layers.Dropout`) between two hidden layers.
    * **Key Learning:** Ensuring you call `model.train()` before the training loop and `model.eval()` before the testing/validation loop, confirming that the random dropping of neurons only occurs during training.

# **IV. Adam Optimization Formula Implementation**

**Focus:** Implementing the full Adam weight update algorithm to understand the mechanism behind adaptive learning rates.

1.  **Custom Adam Optimizer (PyTorch):**
    * **Goal:** Create a class (`CustomAdam`) inheriting from `torch.optim.Optimizer`. Manually override the `@torch.no_grad() def step(self, ...)` method to implement:
        * Tracking the first moment estimate, $m_t$ (momentum).
        * Tracking the second moment estimate, $v_t$ (RMSprop/AdaGrad squared gradient).
        * Applying the **Bias Correction** formulas ($\hat{m}_t$ and $\hat{v}_t$).
        * Performing the final parameter update using the full Adam equation: $p_{new} = p_{old} - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$. 
    * **Key Learning:** This exercise solidifies your understanding of state management (tracking $m_t$ and $v_t$) and the critical role of bias correction in the early stages of training.
