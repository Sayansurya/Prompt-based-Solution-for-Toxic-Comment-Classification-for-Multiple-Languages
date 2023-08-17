import os
import logging
from transformers import DataProcessor, InputExample
logger = logging.getLogger(__name__)

class ToxicAProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            print(line)
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

# Add your task to the following mappings

processors_mapping = {
    "toxic":ToxicAProcessor()
}

num_labels_mapping = {
    "toxic":2
}

output_modes_mapping = {
    "toxic":"classification"
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "toxic": text_classification_metrics,
}

# For regression task only: median
median_mapping = {
    "sts-b": 2.5
}

bound_mapping = {
    "sts-b": (0, 5)
}