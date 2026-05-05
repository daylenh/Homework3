import matplotlib.pyplot as plt

def plot_accuracy(models, accuracy):
    plt.figure()
    plt.bar(models, accuracy)
    plt.title("Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.savefig("report/accuracy.png")

def plot_speed(models, train_times):
    plt.figure()
    plt.bar(models, train_times)
    plt.title("Training Time Comparison")
    plt.xticks(rotation=45)
    plt.savefig("report/speed.png")

def plot_test_speed(models, test_times):
    plt.figure()
    plt.bar(models, test_times)
    plt.title("Inference Time Comparison")
    plt.xticks(rotation=45)
    plt.savefig("report/test_speed.png")

def plot_tradeoff(train_times, accuracy, models):
    plt.figure()
    plt.scatter(train_times, accuracy)
    for i, m in enumerate(models):
        plt.annotate(m, (train_times[i], accuracy[i]))
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Speed vs Accuracy Tradeoff")
    plt.savefig("report/tradeoff.png")