#!/usr/bin/env python

import genomenet

if __name__ == '__main__':
    seed = 42
    EPOCHS = 2
    network = genomenet.GenomeNet(seed, EPOCHS)
    network.info()
    network.load("../../data")
    for data, label in network.train_loader:
        print("Loaded data details:")
        print("\tdata:", data)
        print("\tlabel:", label)
        print("\tdata shape:", data.shape)
        print("\tlabel shape:", label.shape)
        print("\tdata length:", len(data))
        print("\tlabel length:", len(label))
        break
    for current_epoch in range(1, network.epochs + 1):
        network.train(current_epoch)
        network.test()
