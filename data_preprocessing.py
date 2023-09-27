import apache_beam as beam
import tensorflow as tf

class SerializeTextExample(beam.DoFn):
    def __init__(self, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def process(self, element, *args, **kwargs):
        for i in range(len(element)-self.block_size-1):
            feature = {
                "data": tf.train.Feature(int64_list=tf.train.Int64List(value=element[i:i+self.block_size])),
                "target": tf.train.Feature(int64_list=tf.train.Int64List(value=element[i+1:i+self.block_size+1])),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            yield example_proto.SerializeToString()

class GetDataAndTargets(beam.DoFn):
    def __init__(self, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def process(self, element, *args, **kwargs):
        for i in range(len(element) - self.block_size - 1):
            yield {"data": element[i:i+self.block_size], "target": element[i+1:i+self.block_size+1]}


class EncodeStrings(beam.DoFn):
  """Parse each line of input text into words."""
  def process(self, element, *args, **kwargs):
    """Returns an iterator over the words of this element.

    The element is a line of text.  If the line is blank, note that, too.

    Args:
      element: the element being processed

    Returns:
      The processed element.
    """
    stoi, lines = element
    encoded_str_as_list = [stoi[char] for line in lines for char in line]
    yield encoded_str_as_list


class ExtractDistinctChars(beam.DoFn):
    def process(self, element):
        chars = set(element)
        yield chars


def merge_sets(sets):
    return sorted(list(set().union(*sets)))

def charset_to_encoding_map(charset):
    stoi = {ch: i for i, ch in enumerate(charset)}
    return [stoi]


def group_lines_and_charset(lines, charset):
    return [(charset, lines)]


def main():
    with beam.Pipeline() as p:
        input_text = p | beam.io.textio.ReadFromText("txt_data/input-*.txt", strip_trailing_newlines=False)

        distinct_chars = (
                input_text
                | "Extract distinct chars" >> beam.ParDo(ExtractDistinctChars())
                | "Combine distinct chars" >> beam.CombineGlobally(merge_sets)
                | "Get Encoding Map" >> beam.FlatMap(charset_to_encoding_map)
        )

        lines = input_text | "Batch lines" >> beam.BatchElements(min_batch_size=10, max_batch_size=10)

        result = (
                lines
                | "Group Results" >> beam.FlatMap(group_lines_and_charset, beam.pvalue.AsSingleton(distinct_chars))
                | "Process results" >> beam.ParDo(EncodeStrings())
                | "SerializeTextExample" >> beam.ParDo(SerializeTextExample(block_size=8))
                | "WriteToTFRecord" >> beam.io.WriteToTFRecord("tfrecord_data/input", file_name_suffix=".tfrecord")
        )


if __name__ == "__main__":
    main()
