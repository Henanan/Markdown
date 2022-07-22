# Markdown入门
## 代码块：
~~~java
Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the sequence.
