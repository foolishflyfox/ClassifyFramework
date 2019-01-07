import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_history(train_loss, train_acc, val_acc, save_path, suptitle="Training history"):
    train_color = (147.0/255, 82.0/255, 249.0/255)
    val_color = (31.0/255, 223.0/255, 31.0/255)
    title_fontsize = 16
    sub_title_fontsize = 16
    y_label_fontsize = 14
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(suptitle, fontsize=title_fontsize, color='r')
    plt.subplot(211)
    plt.title('Training loss', fontsize=sub_title_fontsize, loc='left', color='#1f1fff')
    plt.plot(range(1,1+len(train_loss)), train_loss, 'o-', c=train_color)
    plt.xlabel('Epochs')
    plt.ylabel('Loss', fontsize=y_label_fontsize)
    plt.subplot(212)
    plt.title('training&validation accuracy', fontsize=sub_title_fontsize,
             loc='left', color='#1f1fff')
    plt.plot(range(1, 1+len(train_acc)), train_acc, 'o-', c=train_color,
            label='train acc')
    plt.plot(range(1, 1+len(val_acc)), val_acc, 'o-', c=val_color,
            label='val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc', fontsize=y_label_fontsize)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    suffix = os.path.splitext(save_path)[1]
    valid_suffix = ['png', 'pdf', 'ps', 'eps', 'svg']
    if len(suffix) > 2 and suffix[1:] in valid_suffix:
        plt.savefig(save_path, format=suffix[1:])
    else:
        print(f'Save history image error: save file suffix should in {valid_suffix}')

