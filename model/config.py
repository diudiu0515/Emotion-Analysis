import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model configuration')
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--maxlen", type=int, default=80)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patient", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()
