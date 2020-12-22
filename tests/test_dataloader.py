import matplotlib.pyplot as plt
from clutils.datasets import CLMNIST

dataset = CLMNIST('/data/cossu', download=False, pixel_in_input=1, perc_test=0.25,
train_batch_size=32, test_batch_size=0, sequential=True,
normalization=255.0, max_label_value=10, return_sequences=True)

train, test = dataset.get_task_loaders([3,4])
for x,y in train:
    print(x.size())
    print(y.size())
    plt.imshow(x[0].view(28,28), cmap='gray')
    #plt.savefig('ris.png')
    break
