import numpy as np
import pandas as pd


def read_cates():
    file = "../audioset/idx_mid_class.csv"
    df = pd.read_csv(file, header=0, sep=',')
    mid = df["mid"]
    class_name = df["display_name"]
    mid2class = dict()
    for m, c in zip(mid, class_name):
        mid2class[m] = c
    return mid, mid2class


def filter_videos(file, mids, mid2class, wfile):
    mids = set(mids)
    df = pd.read_csv(file, header=2, sep=", ", engine='python')
    videos = df["# YTID"].values
    start_seconds = df["start_seconds"].values
    end_seconds = df["end_seconds"].values
    positive_labels = df["positive_labels"].values

    def delete_extra_zero(n):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        return n

    remained_videos = list()
    for i, v in enumerate(videos):
        labels = set(positive_labels[i][1:-1].split(","))
        if len(mids & labels) > 0:
            labels = list(mids & labels)
            classes = list()
            for l in labels:
                classes.append(mid2class[l])
            start = delete_extra_zero(start_seconds[i])
            end = delete_extra_zero(end_seconds[i])
            name = v + "_" + str(start) + "_" + str(end)
            remained_videos.append("\t".join([name, ",".join(classes)]))

    with open(wfile, 'w', encoding='utf-8') as f:
        f.write("filename\tevent_labels\n")
        f.write("\n".join(remained_videos))


def filter_repeated_videos(avvp_file, file):
    avvp_filename = pd.read_csv(avvp_file, header=0, sep="\t")["filename"].values
    filename = pd.read_csv(file, header=0, sep="\t")["filename"].values
    af, fi = set(), set()
    for f in avvp_filename:
        af.add(f[:11])
    for f in filename:
        fi.add(f[:11])
    print(len(af & fi))


def check_inconsistence(avvp_file, total_file):
    avvp_df = pd.read_csv(avvp_file, header=0, sep="\t")
    avvp_filename = avvp_df["filename"].values
    avvp_labels = avvp_df["event_labels"].values
    avvp_dict = dict()
    for f, l in zip(avvp_filename, avvp_labels):
        avvp_dict[f] = l

    total_df = pd.read_csv(total_file, header=0, sep="\t")
    total_filename = total_df["filename"].values
    total_labels = total_df["event_labels"].values
    total_dict = dict()
    for f, l in zip(total_filename, total_labels):
        total_dict[f] = l

    filenames = list(avvp_dict.keys() & total_dict.keys())
    count = 0
    for key in filenames:
        avvp_l = avvp_dict[key]
        total_l = total_dict[key]
        if set(avvp_l.split(",")) != set(total_l.split(",")):
            count += 1
    print(count)


def check_speech(avvp_file):
    df = pd.read_csv(avvp_file, header=0, sep="\t")
    labels = df["event_labels"].values
    count = 0
    for i, l in enumerate(labels):
        if l == 'Speech':
            count += 1
    print(count)


def merge_eval_train(eval_file, balance_file, unbalance_file, file):
    evaldf = pd.read_csv(eval_file, header=0, sep="\t")
    balancedf = pd.read_csv(balance_file, header=0, sep="\t")
    unbalancedf = pd.read_csv(unbalance_file, header=0, sep="\t")

    eval = [(f, l) for f, l in zip(evaldf["filename"].values, evaldf["event_labels"].values)]
    balance = [(f, l) for f, l in zip(balancedf["filename"].values, balancedf["event_labels"].values)]
    unbalance = [(f, l) for f, l in zip(unbalancedf["filename"].values, unbalancedf["event_labels"].values)]

    item = eval + balance + unbalance
    item = sorted(item, key=lambda x: x[0])
    item = ["\t".join([it[0], it[1]]) for it in item]

    with open(file, 'w', encoding='utf-8') as f:
        f.write("filename\tevent_labels\n")
        f.write("\n".join(item))


def filter_AVVP(avvp_file, total_file, wfile):
    total = pd.read_csv(total_file, header=0, sep="\t")
    total_filename = total["filename"].values
    total_labels = total["event_labels"].values
    total_f2l = dict()
    for f, l in zip(total_filename, total_labels):
        total_f2l[f] = l

    avvp = pd.read_csv(avvp_file, header=0, sep="\t")
    avvp_filename = avvp["filename"].values
    # avvp_labels = avvp["event_labels"].values

    remain_files = sorted(list(set(total_filename) - (set(total_filename) & set(avvp_filename))))
    print(len(total_filename))
    print(len(avvp_filename))
    print(len(remain_files))

    remained = list()
    for rf in remain_files:
        remained.append("\t".join([rf, total_f2l[rf]]))

    with open(wfile, 'w', encoding='utf-8') as f:
        f.write("filename\tevent_labels\n")
        f.write("\n".join(remained))


def filter_specific_category(total_file, wfile, cate):
    df = pd.read_csv(total_file, header=0, sep="\t")
    filename = df["filename"].values
    labels = df["event_labels"].values

    remained = list()
    filter = 0
    for f, l in zip(filename, labels):
        if l not in cate:
            remained.append("\t".join([f, l]))
        else:
            filter += 1
    print(filter)
    print(len(remained))

    with open(wfile, 'w', encoding='utf-8') as f:
        f.write("filename\tevent_labels\n")
        f.write("\n".join(remained))


if __name__ == '__main__':
    # mid, mid2class = read_cates()
    # filter_videos("../audioset/unbalanced_train_segments.csv", mid, mid2class,
    #               "../audioset/AVVP_addi_unbalanced_train.csv")
    # check_inconsistence("../data/AVVP_dataset_full.csv", "../audioset/AVVP_addi_unbalanced_train.csv")
    # merge_eval_train("../audioset/AVVP_addi_eval.csv",
    #                  "../audioset/AVVP_addi_balanced_train.csv",
    #                  "../audioset/AVVP_addi_unbalanced_train.csv",
    #                  "../audioset/AVVP_addi_full.csv")
    # filter_speech("../audioset/AVVP_addi_full.csv", "../audioset/AVVP_remained_full.csv")
    # filter_AVVP("../data/AVVP_dataset_full.csv",
    #             "../audioset/AVVP_addi_full.csv",
    #             "../audioset/AVVP_addi_full_filter.csv")
    filter_specific_category("../data/audioset_full_filter_AVVP.csv",
                             "../data/audioset_full_filter_AVVP_speech.csv",
                             ["Speech"])
