# AI4TALK

This is the repo for the AI4Talk track of the AI Journey contest ([competition page](https://dsworks.ru/champ/745de8cb-e023-4b64-9be4-b95b9519f8d3#overview)). 

It contains training data, sample testing data, and the baseline solution.

To receive the data simply clone it with the following command `git clone https://github.com/AIRI-Institute/AI4TALK` or download as an archive.

---

Training data is structured as follows:
```
training_data
├── 1.mp3
├── 2.mp3
├── ...
├── asr.csv
└── translation.csv
```
It contains `csv` files with the annotated data for both tasks, and `mp3` files for the asr task, referenced in the `asr.csv`. See the files structure description in the [Data](https://dsworks.ru) section of the competition page, the tasks are introduced at the competition page as well.
