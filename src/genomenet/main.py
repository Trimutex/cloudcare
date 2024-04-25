#!/usr/bin/env python

import genomenet

if __name__ == '__main__':
    seed = 42
    EPOCHS = 2
    network = genomenet.GenomeNet(seed, EPOCHS)
    network.info()
    network.load("../../data")
    for current_epoch in range(1, network.epochs + 1):
        network.train(current_epoch)
        network.test()
