# 读取IMDB数据集
import torchtext
from torchtext import datasets
# train_iter = torchtext.datasets.IMDB(root='./data', split=('train', 'test'))
# train_iter = iter(train_iter)

# next(train_iter)


train_iter, test_iter = datasets.IMDB(root='data', split=('train', 'test'))  # pyright: ignore[reportArgumentType]
train_iter = iter(train_iter)
data = next(train_iter)
print(data)