#!/usr/bin/env python

import genomenet as GenomeNet

if __name__ == '__main__':
    seed = 42
    EPOCHS = 2
    network = GenomeNet(seed, EPOCHS)
    network.load("../../data")
    for current_epoch in range(1, network.epochs + 1):
        network.train(current_epoch)
        network.test()
