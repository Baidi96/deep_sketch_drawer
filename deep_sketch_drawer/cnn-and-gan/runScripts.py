import random
import argparse
import convEngine
import gan
import produceHED
import testConv
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="produceHED", help="produceHED|hed|conv|gan|testCONV")
    parser.add_argument("--cv_lr", type=float, default=0.001, help="learning rate for convolutional networks")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--discrim_lr", type=float, default=0.001, help="learning rate for discriminator")
    parser.add_argument("--gan_lr", type=float, default=0.001, help="learning rate for gan")
    parser.add_argument("--epochs", type =int, default = 1000, help= "epochs for running models")
    parser.add_argument("--testFile", type = str, default = "hed.hdf5", help = "saved model file for conv")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    types = args.type
    cv_lr = args.cv_lr
    batch_size = args.batch_size
    discrim_lr = args.discrim_lr
    gan_lr = args.gan_lr
    epochs = args.epochs
    testFile = args.testFile
    print(types)
    if types == "produceHED":
        #run HED producer code
        produceHED.produceHEDmain("colorFolder", "outputFolder")
    elif types == "hed":
        #train using HED model
        convEngine.convMain(types, cv_lr, batch_size, epochs)
    elif types == "conv":
        #train using pretrained HED model and extra layers on top
        convEngine.convMain(types, cv_lr, batch_size, epochs)
    elif types == "gan":
        gan.ganMain(5000, batch_size, discrim_lr, gan_lr)
    elif types == "testCONV":
        print("tested")
        testConv.test("colorFolder", "outputFolder", testFile)

main()
