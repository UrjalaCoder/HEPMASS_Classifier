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
            correct_out_tensor = torch.tensor([1.0 - correct_out, correct_out])
            correct_out_tensor = correct_out_tensor.to(dtype=torch.float32)

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
        input_tensor = input.to(dtype=torch.float32)
        prediction = net(input_tensor)
        correct_value = correct_out.item()
        pred_value = 0 if prediction[0] > prediction[1] else 1

        if int(round(correct_value)) == pred_value:
            correctly_evaluated += 1
    p = float(correctly_evaluated) / float(len(dataset))
    print(f"CORRECT {correctly_evaluated} / {len(dataset)} = {p * 100}%")
    return p

def main():
    raw_data = load_from_file(filename="1000_train.csv", skip=100)
    test_data = load_from_file(filename="1000_test.csv", skip=50)
    test_dataset = generate_dataset(test_data)
    dataset = generate_dataset(raw_data)
    print("LOADED DATA")
    in_size = dataset[0][0].shape[0]
    out_size = 2
    net = Network(in_size, out_size, hidden_size=6)
    print("Training: ")
    train(net, dataset, epochs=12, lr=0.2)

    print("Training done, evaluating...")
    p = evaluate_net(net, test_dataset)
    if p > 0.8:
        filename_p = int(round(p * 100))
        file_pt = f"nets/{filename_p}-network-{len(dataset)}-rows.pt"
        print(f"Saving to {file_pt}")
        torch.save(net, file_pt)

if __name__ == '__main__':
    main()
