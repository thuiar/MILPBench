import matplotlib.pyplot as plt
import os

def plot_log(log):
    with open(log, 'r', encoding='utf-8')as f:
        lines = f.readlines()
    train_losses = []
    valid_losses = []
    epoches = []
    epoch = 1
    for i in range(len(lines)):
        if f'EPOCH {epoch}' in lines[i]:
            try:
                train_loss = float(lines[i+1].split(':')[3].strip().split(' ')[0])
                train_losses.append(train_loss)
                valid_loss = float(lines[i+2].split(':')[3].strip().split(' ')[0])
                valid_losses.append(valid_loss)
                epoches.append(epoch)
            except:
                print('error')
            epoch += 1
    print('epoches: ',epoches)
    print('train loss: ',train_losses)
    print('valid loss: ',valid_losses)

    plt.plot(epoches, train_losses, label = 'train loss')
    plt.plot(epoches, valid_losses, label = 'valid loss')
    plt.xlabel = 'epoch'
    plt.ylabel = 'loss'
    plt.title = 'learning curve'
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{os.path.split(log)[0]}/learning curve.png')
    print('lr graph done')

plot_log('model/1_item_placement/1/train_log.txt')