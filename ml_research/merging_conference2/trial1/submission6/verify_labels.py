import torchvision.datasets as datasets
import numpy as np

# SVHN labels
svhn_test = datasets.SVHN(root='./data', split='test', download=False)
labels = np.unique(svhn_test.labels)
print("SVHN unique labels:", labels)

# DTD classes
dtd_test = datasets.DTD(root='./data', split='test', download=False)
print("DTD classes:", dtd_test.classes[:10], "... total:", len(dtd_test.classes))
