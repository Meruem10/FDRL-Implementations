from typing import Dict

import torch


class Optimizer(object):
    def __init__(
        self,
        model,
        type: str = "Adam",
        lr: float = 0.001,
        optimizer_kw: Dict = {},
    ) -> None:
        self.type = type
        self.lr = lr
        self.optimizer = eval(
            f"torch.optim.{self.type}(model.parameters(), lr={self.lr}, **optimizer_kw)"
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()


if __name__ == "__main__":
    # example
    from ann import ANN

    torch.manual_seed(42)

    # define model
    batch = 4
    num_inputs = 7
    num_outputs = 2
    hidden_layer_dims = [4, 4]

    dropout = 0.1
    use_batch_norm = True
    weight_init = "xavier_normal"
    weight_init_kw = {"gain": 1.0}

    net = ANN(
        num_inputs,
        num_outputs,
        hidden_layer_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        weight_init=weight_init,
        weight_init_kw=weight_init_kw,
    )

    # define optimizer
    optimizer = Optimizer(
        net, "SGD", 0.1, {"momentum": 0.9, "weight_decay": 1e-4}
    )

    # forward-pass
    x = torch.randn(
        (batch, num_inputs), device="cpu", dtype=torch.float32
    )  # batch x dim
    y = net(x)
    loss = torch.mean((1.0 - y) ** 2)

    # model-parameters before backpropagation
    print("Before:")
    params = list(net.parameters())
    print(params)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # model-parameters before backpropagation
    print("\nAfter:")
    params = list(net.parameters())
    print(params)

    print("All tests have passed successfully!")
