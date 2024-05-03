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

## Contributors

I am the sole contributor for this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)

