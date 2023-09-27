from torchdata.datapipes.iter import FileLister, FileOpener


def main():
    datapipe1 = FileLister("tfrecord_data", "*.tfrecord")
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()
    for example in tfrecord_loader_dp:
        print(example)


if __name__ == "__main__":
    main()
