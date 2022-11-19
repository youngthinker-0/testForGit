import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


if __name__ == "__main__":
    train_loss_path = 'train_loss.txt'

    y_train_loss = data_read(train_loss_path)
    x_train_loss = range(len(y_train_loss))

    plt.figure()


ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')
plt.ylabel('loss')

plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('loss curve')
plt.show()
