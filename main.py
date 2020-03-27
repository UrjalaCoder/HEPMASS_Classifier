from network import Network
import torch
from data_parser import load_from_file, generate_dataset

def parse_prediction(pred_tensor):
    value = pred_tensor.item()
    correct_val = 0 if value < 0.5 else 1
    return torch.tensor([value], dtype=torch.float32)

def train(net, dataset, epochs=1, lr=1):
    for e in range(epochs):
        total_loss = 0
        for sample in dataset:
            input, correct_out = sample
            input_tensor = input.to(dtype=torch.float32)
            prediction = net(input_tensor)
            # print(prediction)
            correct_out_tensor = torch.tensor([1.0 - correct_out, correct_out])
            correct_out_tensor = correct_out_tensor.to(dtype=torch.float32)
            # print(correct_out_tensor)
            # test_input = dataset[0][0].to(dtype=torch.float32)
            # test_output = dataset[0][1].unsqueeze(dim=0).to(dtype=torch.float32)

            loss = net.loss(prediction, correct_out_tensor)
            loss.backward()

            # Training loop:
            with torch.no_grad():
                for p in net.parameters():
                    p -= p.grad * lr
                net.zero_grad()
                total_loss += loss.item()


        avg_loss = total_loss / len(dataset)
        print(f"LOSS: {avg_loss}")

def evaluate_net(net, dataset):
    correctly_evaluated = 0
    for sample in dataset:
        input, correct_out = sample
        # print(sample)
        input_tensor = input.to(dtype=torch.float32)
        prediction = net(input_tensor)
        correct_value = correct_out.item()
        pred_value = 0 if prediction[0] > prediction[1] else 1
        # print("CORRECT ", correct_value)
        # print(prediction)
        # print(pred_value)

        if int(round(correct_value)) == pred_value:
            correctly_evaluated += 1
    print(f"CORRECT {correctly_evaluated} / {len(dataset)} = {(float(correctly_evaluated) / float(len(dataset))) * 100}%")


def main():
    raw_data = load_from_file(filename="1000_train.csv")
    # print(raw_data)
    test_data = load_from_file(filename="1000_test.csv")
    # print(test_data)
    test_dataset = generate_dataset(test_data)
    dataset = generate_dataset(raw_data)

    in_size = dataset[0][0].shape[0]
    out_size = 2
    net = Network(in_size, out_size, hidden_size=6)

    train(net, dataset, epochs=5, lr=0.2)

    print("Training done, evaluating...")
    evaluate_net(net, test_dataset)

if __name__ == '__main__':
    main()
