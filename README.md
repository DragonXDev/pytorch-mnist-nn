# MNIST PyTorch Neural Network

A MNIST CNN developed to recognize handwritten digits; grayscale input channel of 1 accepted, ```1x28x28``` input.

![image](https://github.com/DragonXDev/pytorch-mnist-nn/assets/99859617/f77a5f9d-09bd-4624-bb14-cf64ca919dd3)


## Usage & Information
The CNN is trained in batches of ```32``` from the DB. Each of the 32 filters is a 2D ```3x3``` and extracted from the ```1x28x28```

A Visual of an MNIST CNN

![image](https://github.com/DragonXDev/pytorch-mnist-nn/assets/99859617/0f4a9be1-07e3-4632-9048-8a741a7fdeef)

To train the CNN, 

```bash
poetry run python src/train_model.py
```

This will contact the MNIST DB & run 10 epochs to train the model. The appropriate state after the model is finished executing will be created
as the ```digital_model.pt``` file.

Once completed, run the model with the binary generated using

```bash
poetry run python src/main.py
```

## Optimizers

For the purpose of MNIST, the Adam optimizer with ```lr=1e-3``` performs best

![image](https://github.com/DragonXDev/pytorch-mnist-nn/assets/99859617/8ac61337-e8e0-45aa-9383-35fb368731fa)

### Adam

Starting with Stochastic Gradient Descent

```math
g = \frac{1}{m} \nabla_{\theta}\sum_{i}L(f(x^{(i)};\theta),y^{(i)})
```
Where $\theta$ is the models' params, $g$ is the negative direction of the gradient, $m$ is the size of the mini-batches of data, $f(x^{(i)}; \theta)$ is the neural network, $x^{(i)}$ is the training data, $y^{(i)}$ are the training labels, and $L()$ is the loss function.

The Adam optimizer redefines SGD's params as such:

```math
m = \beta_{1}m + (1-\beta_{1})g
```
```math
s = \beta_{2}s + (1-\beta_{2})g^{T}g
```
```math
\theta = \theta - \epsilon_{k}\cdot\frac{m}{\sqrt{s+eps}}
```

For this project, $\beta_{1} = 0.9$, $\beta_{2} = 0.999$, and $eps$ (learning rate) $=$ 1e-3.

### Nesterov Momentum $\nabla$

Nesterov's Momentum Acceleration $\nabla$ (NAG) performs significantly worse than Adam.

```math
\nu = \alpha\nu - \epsilon\nabla_{\theta}(\frac{1}{m}\sum_{i}L(f(x^{(i)};\theta+\alpha\cdot\nu),y^{(i)})
```
```math
\theta = \theta + \nu
```

The loss / cost after ```Epoch 10``` ended at ~ ```0.01253``` on average.

## Contributors

I am the sole contributor of this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)

