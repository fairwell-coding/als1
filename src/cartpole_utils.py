import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_results(train_metrics, test_metrics, window=100):
    fig, ax_list = plt.subplots(2, 2)
    ax_list[0,0].set_title('Training')
    ax_list[0,0].plot(train_metrics['reward'], lw=2, color='green')
    ax_list[0,0].set_ylabel('Acc. Reward')
    # ax_list[1,0].plot(time_list[:N_trial], lw=2, color='blue')
    # ax_list[1,0].set_ylabel('Trial duration')
    ax_list[1,0].plot(train_metrics['loss'], lw=2, color='red')
    # ax_list[2,0].set_ylabel('Bellman error')
    ax_list[1,0].set_xlabel('Episode')
    ax_list[1,0].set_ylabel('Loss')
    ax_list[1,0].set_yscale('log')

    ax_list[0,1].set_title('Testing')
    ax_list[0,1].plot(test_metrics['reward'], lw=2, color='green')
    ax_list[0,1].set_ylabel('Acc. Reward')
    ax_list[0,1].axhline(np.mean(test_metrics['reward']),color='black',linestyle='dashed',lw=4)
    # ax_list[0,1].set_ylim(ax_list[0,0].get_ylim())

    ax_list[1,1].plot(test_metrics['loss'], lw=2, color='red')
    ax_list[1,1].axhline(np.mean(test_metrics['loss']), lw=4, color='black', linestyle='dashed')
    # ax_list[2,0].set_ylabel('Bellman error')
    ax_list[1,1].set_xlabel('Episode')
    ax_list[1,1].set_yscale('log')
    ax_list[1,1].set_ylabel('Loss')



    # ax_list[1,1].plot(time_list[N_trial:], lw=2, color='blue')
    # ax_list[1,1].set_ylabel('Trial duration')
    # ax_list[1,1].axhline(np.mean(time_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    # ax_list[1,1].set_ylim(ax_list[1,0].get_ylim())

    # ax_list[2,1].plot(err_list[N_trial:], lw=2, color='red')
    # ax_list[2,1].set_ylabel('Bellman error')
    # ax_list[2,1].axhline(np.mean(err_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    # ax_list[2,1].set_ylim(ax_list[2,0].get_ylim())
    # ax_list[2,1].set_yscale('log')

    # ax_list[2,1].set_xlabel('Trial number')
    plt.show()
