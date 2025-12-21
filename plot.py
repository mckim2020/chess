import matplotlib.pyplot as plt
import re

def quick_plot(filename='training.log'):
    episodes = []
    losses = []
    
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(r'Episode (\d+) \| Loss: ([\d.e+\-]+)', line)
            if match:
                episodes.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, linewidth=0.8, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Episodes')
    plt.grid(True, alpha=0.3)
    plt.savefig('loss.png', dpi=300)
    plt.show()

quick_plot('train.out')
