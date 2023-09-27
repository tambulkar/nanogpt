import apache_beam as beam


def main():
    with open("input.txt") as f:
        data = f.read()

    num_splits = 10
    for i in range(num_splits):
        with open(f"txt_data/input-{i}.txt", "w") as f:
            if i == num_splits-1:
                chunk = data[i*(len(data)//num_splits):]
            else:
                chunk = data[i*(len(data)//num_splits): (i+1)*(len(data)//num_splits)]
            f.write(chunk)


if __name__ == "__main__":
    main()
