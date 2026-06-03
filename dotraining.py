from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
    set_seed,
    AutoConfig
)

from datasets import Dataset

import evaluate
import sys
import time
import numpy as np

# The first two fields of the CSV datasets (filename and field name) are ignored
# and are used only for reference when debugging.
ignoreLineCount = 2

# This script is used to train the address autofill model for Firefox.
# The configuration below allows two modes, 'all' and 'supported'. All mode
# handles all of the field types in the fieldTypesDict list whereas supported
# mode only handles the field types that are supported by Firefox autofill.
# Generally, we have been using supported mode.
#
# It is expected that there is a file 'training-supported.txt',
# 'validation-supported.txt' and 'testing-supported.txt' in the same directory
# as this script which contains the training, validation and test data in CSV
# format. In 'all' mode, the '-supported' should be removed from the filenames,
# allowing both datasets to coexist.
#
# This training, validation and test data is generated separately from
# sample forms from various regions.
#
# To train:
#   python dotraining.py train
# To test:
#   python dotraining.py test
#
# There is also a random forest classifier that can be tried out with:
#   python dotraining.py forest
# It requires sklearn.ensemble.RandomForestClassifier.
#
# A special case 'together' is also supported to handle a file named
# 'together-supported.txt' intended to be a concatenation of all three of
# the input files.
#   python dotraining.py together
#
#
# If a different string is passed to this script as an argument, then it is
# treated as a single token list to test with.
#
# Trained models are saved in the output-models directory.
#
# The CSV data has four fields: source filename, expected fieldname,
# expected fieldname index (from fieldTypesDict), and the set of tokens.

# ---- Configuration Section ----

# This section allow configuration of the training and testing.

# Source model to use for training
modelName = "huawei-noah/TinyBERT_General_4L_312D"

# There are two modes: 'all' and 'supported'. All mode handles all of the
# field types in the fieldTypesDict list whereas supported mode only handles
# the field types that are supported by Firefox autofill. This python script
# will use different datasets in each mode. Select the desired one by setting
# the value of dataVariant to "" for all mode and "-supported" for supported mode.
#dataVariant = ""
dataVariant = "-supported"

# Number of epochs to train. This is the number of passes through the training
# data that are performed.
numEpochs = 15

# ---- End Configuration Section ----

if dataVariant == "-supported":
  modelExtra = "supported"
else:
  modelExtra = "all"

# Append an extra string to the filename to test variations.
#modelExtra = modelExtra + "-updated"
#dataVariant = dataVariant + "-updated"

saveModelName = "autofill-tiny-" + modelExtra
saveModelDir = "output-models/" + saveModelName

# Other models that were experimented with.
#modelName = "nhull/random-forest-model"
#modelName = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#modelName = "distilbert/distilbert-base-uncased"
#modelName = "Mozilla/tinybert-uncased-autofill"
#modelName = "microsoft/MiniLM-L12-H384-uncased"
#modelName = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
#modelName = "Xenova/bert-base-uncased"

#transformers.set_seed(189)

# The weights could be used, but are not for now.
weights = [
 1,1,1,1,3,
 2,2,3,2,2,
 2,4,3,4,3,
 3,1,1,1,1,
 1,1,1,2,1,
 2,2,2,1,1,
 1,1,1,2,2,
 2,2,2,1,1,
 1,1,1,1,1,
 1,2,2,2,2,
 1,1,1,1,1,
 1,1,1,1,1,
 1,1,1,1,1,
 1
]

# List of fields with their ids
fieldTypesDict = {
  'other': 1,
  'given-name': 2,
  'family-name': 3,
  'name': 4,
  'additional-name': 5,
  'phonetic-given-name': 6,
  'phonetic-family-name': 7,
  'phonetic-name': 8,
  'honorific-prefix': 9,
  'honorific-suffix': 10,
  'nickname': 11,
  'street-address': 12,
  'address-lookup': 13,
  'address-line1': 14,
  'address-line2': 15,
  'address-line3': 16,
  'address-level1': 17,
  'address-level2': 18,
  'address-level3': 19,
  'address-level4': 20,
  'street': 21,
  'address-streetname': 22,
  'address-housenumber': 23,
  'address-extra-housesuffix': 24,
  'postal-code': 25,
  'postal-code-lookup': 26,
  'postal-code-and-city': 27,
  'postal-code-or-suburb': 28,
  'country': 29,
  'country-name': 30,
  'tel': 31,
  'tel-country-code': 32,
  'tel-national': 33,
  'tel-area-code': 34,
  'tel-local': 35,
  'tel-local-prefix': 36,
  'tel-local-suffix': 37,
  'tel-extension': 38,
  'organization': 39,
  'organization-title': 40,
  'bday': 41,
  'bday-day': 42,
  'bday-month': 43,
  'bday-year': 44,
  'email': 45,
  'apartment': 46,
  'floor': 47,
  'stair': 48,
  'building': 49,
  'block': 50,
  'address-extra': 51,
  'cc-name': 52,
  'cc-given-name': 53,
  'cc-additional-name': 54,
  'cc-family-name': 55,
  'cc-number': 56,
  'cc-exp': 57,
  'cc-exp-month': 58,
  'cc-exp-year': 59,
  'cc-csc': 60,
  'cc-type': 61,
  'sex': 62,
  'id-number': 63,
  'vat-number': 64,
  'reference-point': 65,
  'loginname': 66,
}
fieldTypesReversedDict = {v: k for k,v in fieldTypesDict.items()}

fieldNamesCloseDict = {
  "street-address": ["address-line1", "street"],
  "address-line1": ["street-address", "street"],
  "address-line2": ["apartment"],
  "street": ["street-address", "address-line1"],
  "postal-code-and-city": ["postal-code"],
  "postal-code-and-suburb": ["postal-code"],
  "tel": ["tel-national"],
  "tel-national": ["tel"],
  "apartment": ["address-line2"],
  "given-name": ["cc-given-name"],
  "additional-name": ["cc-additonal-name"],
  "family-name": ["cc-family-name"],
  "name": ["cc-name"],
  "cc-given-name": ["given-name"],
  "cc-additional-name": ["additonal-name"],
  "cc-family-name": ["family-name"],
  "cc-name": ["name"],
  "loginname": ["email"],
  "email": ["loginname"],
  "country": ["country-name"],
  "country-name": ["country"],
}

def readFile(filetype):
  list = []

  file = open(filetype + dataVariant + ".txt", encoding="utf-8")
  lines = file.readlines()
  
  for line in lines:
    line = line.strip()
    lineData = line.split(",", ignoreLineCount + 1)
    print(lineData)
    try:
      list.append({"label": int(lineData[ignoreLineCount]), "text": lineData[ignoreLineCount + 1]})
    except Exception:
      print(filetype + ".txt : " + line)
      raise
  dataset = Dataset.from_list(list)
  return dataset

def doTraining():
  tokenizer = AutoTokenizer.from_pretrained(modelName)
  
  def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True, max_length=512)
  
  ds = readFile("training")
  train_ds = ds.map(preprocess_function, batched=True)
  
  ds = readFile("validation")
  validate_ds = ds.map(preprocess_function, batched=True)
  
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  accuracy = evaluate.load("accuracy")
  
  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      return accuracy.compute(predictions=predictions, references=labels)
  
  model = AutoModelForSequenceClassification.from_pretrained(
      modelName, num_labels=len(fieldTypesDict), ignore_mismatched_sizes=True,
      id2label=fieldTypesReversedDict, label2id=fieldTypesDict
  )
  
  training_args = TrainingArguments(
      seed=189,
      data_seed=189,
      output_dir=saveModelName,
#      learning_rate=1e-5,
#      per_device_train_batch_size=16,
#      per_device_eval_batch_size=16,
      num_train_epochs=numEpochs,
#      weight_decay=0.01,
      eval_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=validate_ds,
      processing_class=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )
  
  trainer.train()
  trainer.save_model(saveModelDir)

  test("testing")

def test(filename):
  classifier = pipeline("text-classification", model=saveModelDir, truncation=True, max_length=512)

  list = []
  expectedList = []
  autocompleteList = []
  detailsList = []

  file = open(filename + dataVariant + ".txt", encoding="utf-8")
  lines = file.readlines()

  count = 0
  for line in lines:
    line = line.strip()
    lineData = line.split(",", ignoreLineCount + 1)
    list.append(lineData[ignoreLineCount + 1])

    if lineData[ignoreLineCount + 1].startswith("a-c-"):
      actype = lineData[ignoreLineCount + 1].split(" ", maxsplit=1)[0][4:]
      if actype in fieldTypesDict:
        autocompleteList.append(actype)
      else:
        autocompleteList.append(None)
    else:
      autocompleteList.append(None)

    print (lineData)
    expectedList.append(fieldTypesReversedDict[int(lineData[ignoreLineCount])])
    if ignoreLineCount == 2:
      detailsList.append([lineData[0], lineData[1]]);
    else:
      detailsList.append(["", ""]);

    count = count + 1

  for l in list:
    print(l)

  results = classifier(list, truncation=True)

  correct = 0
  close = 0
  blank = 0
  fieldCorrect = {}

  for result in zip(results, expectedList, autocompleteList, detailsList):
    actualresult = None
    suffix = ""

#    if result[2] is not None:
#      actualresult = result[2]
#    else:
    actualresult = result[0]["label"]

    match = 0
    if actualresult == result[1]:
      correct += 1
      close += 1
      match = 1
    elif actualresult in fieldNamesCloseDict and result[1] in fieldNamesCloseDict[actualresult]:
      suffix = " -"
      close += 1
    else:
      suffix = " X"

    if result[2] is not None:
      suffix += " AC: " + result[2]

    if result[1] in fieldCorrect:
      fieldCorrect[result[1]] = (fieldCorrect[result[1]][0] + match, fieldCorrect[result[1]][1] + 1)
    else:
      fieldCorrect[result[1]] = (match, 1)

    if result[1] == "other" and actualresult != "other":
      blank += 1

    probability = result[0]["score"]
    if ignoreLineCount == 2:
      print(result[3][0] + "," + result[3][1] + "  ", end="")
    print(result[1].ljust(26, " ") + "  " + result[0]["label"].ljust(26, " ") + " " + f"{probability:.4f}" + suffix)

  print(f"Total Accuracy: {(correct / len(results)):.4f} {correct}/{len(results)}")
  print(f"Close Accuracy: {(close / len(results)):.4f} {close}/{len(results)}")
  print(f"Expect Blank: {(blank / len(results)):.4f} {blank}/{len(results)}")

  print("Field Accuracy:")
  for field in sorted(fieldCorrect.keys()):
    print("  " + field + " : " + str(fieldCorrect[field][0] / fieldCorrect[field][1]))

  for result in zip(results, expectedList, autocompleteList):
    if result[1] == "other" and result[0] == "other" and result[2] is None:
      print("SPECIAL: " + result[0] + " " + result.probability + "\n");

def infer(text):
  classifier = pipeline("text-classification", model=saveModelDir)

  results = classifier([text])
  for result in results:
    print(result)

def forest():
  import pandas as pd
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn import metrics
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  trainingDS = readFile("training")
  trainingDS = pd.DataFrame(trainingDS)
  trainingDS_X = trainingDS["text"]
  trainingDS_Y = trainingDS["label"]

  testingDS = readFile("testing")
  testingDS = pd.DataFrame(testingDS)
  testingDS_X = testingDS["text"]
  testingDS_Y = testingDS["label"]

  vectorizer = CountVectorizer();
  trainingXCount = vectorizer.fit_transform(trainingDS_X);
  testingXCount = vectorizer.transform(testingDS_X);

  random_forest_model = RandomForestClassifier()
  random_forest_model.fit(trainingXCount, trainingDS_Y)

  before = time.perf_counter()

  yprediction = random_forest_model.predict(testingXCount)

  duration = time.perf_counter() - before

  accuracy = metrics.accuracy_score(testingDS_Y, yprediction)

  print(f"  Random Forest: {accuracy:.2f}% Time: {duration:.2f}")

  correct = 0
  close = 0
  fieldCorrect = {}

  for result in zip(yprediction, testingDS_Y.values, testingDS_X.values):
    suffix = ""

    found_result = result[0]
#    if result[2].startswith("ac-"):
#      fieldtype = result[2][3:result[2].find(" ")]
#      if fieldtype in fieldTypesDict:
#        found_result = fieldTypesDict[fieldtype]

    found_field_type = fieldTypesReversedDict[found_result]

    match = 0
    if found_result == result[1]:
      correct += 1
      close += 1
      match = 1
    elif found_field_type in fieldNamesCloseDict and fieldTypesReversedDict[result[1]] in fieldNamesCloseDict[found_field_type]:
      suffix = " -"
      close += 1
    else:
      suffix = " X"

    fieldtype = fieldTypesReversedDict[result[1]]
    if fieldtype in fieldCorrect:
      fieldCorrect[fieldtype] = (fieldCorrect[fieldtype][0] + match, fieldCorrect[fieldtype][1] + 1)
    else:
      fieldCorrect[fieldtype] = (match, 1)

#    print(found_result + "  " + result[1] + suffix)
    print(fieldTypesReversedDict[found_result] + "  " + fieldTypesReversedDict[result[1]] + suffix)

  print("Total Accuracy: " + str(correct / len(yprediction)))
  print("Close Accuracy: " + str(close / len(yprediction)))

  print("Field Accuracy:")
  for field in sorted(fieldCorrect.keys()):
    print("  " + field + " : " + str(float(fieldCorrect[field][0]) / float(fieldCorrect[field][1])))

if len(sys.argv) == 2 and sys.argv[1] == "train":
  doTraining()
elif len(sys.argv) == 2 and sys.argv[1] == "test":
  test("testing")
elif len(sys.argv) == 2 and sys.argv[1] == "together":
  test("together")
elif len(sys.argv) == 2 and sys.argv[1] == "forest":
  forest()
else:
  infer(" ".join(sys.argv[1:]))
