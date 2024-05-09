from digitnn import DigitClassifier, dataset
from torch import save, optim
from torch.nn import CrossEntropyLoss

# Instance of NN, calculate cost, optimize
classifier = DigitClassifier().to('cuda')
optimizer = optim.Adadelta(classifier.parameters(),lr=1e-3)
cost_fn = CrossEntropyLoss()

# Training Model
if __name__ == "__main__":
    for epoch in range(10):
        for batch in dataset:  # 32 chunked batches
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda') 
            yhat = classifier(X)
            cost = cost_fn(yhat, y)
            # Apply backpropagation
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Cost is {cost.item()}")

    with open('digit_model.pt', 'wb') as file:  # Save the model state
        save(classifier.state_dict(), file)
