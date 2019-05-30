### Pipeline

- Proprocess.py: function fill_median and fill_mean: Imputation is happening on the entire data set . Needs to be only on train set
- City,county, district shouldn’t be dropped，or at least decribe the reason to drop them

### Changes
- In the main.py, I aggragate all the cleaning, imputation, features generation into one function called transform().It's called by the main loop (run_time_validation) that only do the transform on every trainning set.
- I add one function in the pipeline/features_generator.py called dummize_top_k(). It can deal with the situation like city, county, district that have many values. It will get the top k categories, and make the other values as 'other', and null as 'unknown'.
- In my main.py's transform function, I called dummize_top_k on city, county, district to generate the dummy columns


### Mislabel the outcome
- Run the code on the donors choose problem

### Changes

- Instead of labelling getting funded as 1, this time, I change the label as not getting funded in 60 days as 1 else as 0.

### Write up the results

- More description to required in the notebook: in 5.compare all the model, how and why?
- Graphs 3 and 4 end at different places - they must be wrong


### Changes
This time, I use one notebook to analyze the results named [results_analysis.ipnb](./results_analysis.ipnb)

### Code quality:
- Suggestion in naming a file: processor.py -> data_process.py or preprocess.py
- Hard coded/Not reusable Repeated code: e.g.: line 78-106 in main.py ev.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,1.0)
- More comments required: For example, what is 'xs_lst' and what does it use for?  also, line 117-129: run_time_validation, what are the input arguments? Same for functions in time_validate.py

### Changes:

- I renamed process.py as preprocess.py in the pipeline.
- Instead of hard coding every k percentage, I used a loop in the main.py
- I have updated all the docstrings of every functions.
