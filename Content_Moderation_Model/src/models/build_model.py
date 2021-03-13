import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import feature_column
import pickle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

def train_model(train_path, test_path):
    """Import data, train model and save to predictions/predictions.csv
      Input: Path to training data, path to test data
      Output: CSV of predictions saved to predictions/predictions.csv"""

    # Import and create test/validation set
    train_val_df = pd.read_csv(train_path)
    train_val_df = train_val_df[~train_val_df['Violation'].isin([5, 7])]

    # Set target and feature columns (target: 'ViolationBool')
    data_columns = ['ViolationBool',
                 'Violation', 
                 'LanguageCode', 
                 'TitleText', 
                 'title_len',
                 'text_len', 
                 'titletext_len', 
                 'title_word_no', 
                 'text_word_no',
                 'titletext_word_no', 
                 'title_avg_word_length', 
                 'text_avg_word_length',
                 'titletext_avg_word_length', 
                 'year', 
                 'month', 
                 'hour', 
                 'YYMM']
    train_val_df = train_val_df[data_columns]

    # 70:30 train/validation split, resetting indices
    train_df, val_df = train_test_split(train_val_df, test_size=0.3, random_state=42) 
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)

    # Import test set, resetting index
    test_df = pd.read_csv(test_path)
    test_df = test_df[~test_df['Violation'].isin([5, 7])]
    test_df = test_df[data_columns]
    test_df.reset_index(inplace=True, drop=True)

    # Convert training/validation and test sets to TensorFlow format
    # Note that 'Violation' is dropped here, as it is not used in training, but is needed when exporting the predictions
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      train_df.drop('Violation', axis=1), train_df["ViolationBool"], num_epochs=None, shuffle=False)

    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      test_df.drop('Violation', axis=1), test_df["ViolationBool"], shuffle=False) 

    # Define text embedded feature column 'TitleText'
    embedded_text_feature_column = hub.text_embedding_column(
      key="TitleText",  # the `key` is the column that contains the text
      module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    tf.get_logger().setLevel('WARNING')

    # Create numeric an categorical feature columns
    feature_columns = [embedded_text_feature_column]

    cat_features = ['LanguageCode', 'month', 'hour']
    for col_name in cat_features:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, train_df[col_name].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)

    num_features = ['titletext_word_no', 'text_len', 'title_len']

    for header in num_features:
        feature_columns.append(feature_column.numeric_column(header))

    my_config = tf.estimator.RunConfig(tf_random_seed=42)

    # Define TensorFlow estimator
    estimator = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[700, 210],
      config=my_config,
      optimizer=lambda: tf.keras.optimizers.Adam(
          learning_rate=tf.compat.v1.train.exponential_decay(
              learning_rate=0.001,
              global_step=tf.compat.v1.train.get_global_step(),
              decay_steps=8000,
              decay_rate=0.96)))

    # Training for 10,000 steps
    estimator.train(input_fn=train_input_fn, steps=10000);   

    # Save trained model
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))
    estimator.export_saved_model("../models/binary", serving_input_fn)

    # Evaluate and print accuracy scores
    #train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    #print("Training set accuracy: {accuracy}".format(**train_eval_result))

    # Evaluate and print accuracy scores
    #eval_result = estimator.evaluate(input_fn=val_input_fn)
    #print("Evaluation set accuracy: {accuracy}".format(**eval_result))

    test_eval_result = estimator.evaluate(input_fn=test_input_fn)
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    # Get predictions from trained model
    def get_predictions(model, input_fn):
      class_id = [];  probability = []
      for x in estimator.predict(input_fn = input_fn):
          class_id.append(x['class_ids'][0]);
          probability.append(x['logistic'][0])
      return pd.DataFrame(data = {'Prediction' : class_id, 'Probability' : probability})

    # Predictions df
    preds = get_predictions("", test_input_fn)

    # Add predictions to test data set
    test_df_preds = test_df.copy() 
    test_df_preds['Prediction'] = preds['Prediction']
    test_df_preds['Probability'] = preds['Probability']

    # Calibrate model to the optimal threshold based on the precision-recall curve
    pre, rec, thresh = precision_recall_curve(test_df_preds['ViolationBool'], test_df_preds['Probability'])

    f1_scores = 2*rec*pre/(rec+pre)
    best_thresh = thresh[np.argmax(f1_scores)]

    test_df_preds['Prediction'] = np.where(test_df_preds['Probability']>=best_thresh, 1, 0)

    # Export prediction data
    test_df_preds.to_csv('predictions/binary_predictions.csv')
    pickle.dump(test_df_preds, open('predictions/binary_predictions.pkl', 'wb'))

