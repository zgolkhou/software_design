import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import pickle


def train_multiclass_binary_model(train_path, test_path):
  """Import data, train model and save to predictions/predictions.csv
    Inputs: Path to binary model output for training data, path to test data
    Output: CSV of predictions saved to predictions/predictions.csv"""

  # Import and create test/validation set
  train_val_df = pd.read_csv(train_path)

  # Set target and feature columns (target: 'Violation')
  data_columns = ['Violation', 
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

  # Removing label 0, 5, 6 ,7, 8, 12 from the training & validation sets.    
  my_df_cat = train_val_df[train_val_df['Violation'] != 0].copy()

  my_df_cat = my_df_cat[~my_df_cat['Violation'].isin([5, 6, 7, 8, 12])]

  my_df_cat['Violation'] = np.where(my_df_cat['Violation']>=5, my_df_cat['Violation']-4, my_df_cat['Violation'])
  my_df_cat = my_df_cat.reset_index()

  my_df_cat['Violation'] = my_df_cat['Violation'].astype('int')
  my_df_cat['Violation'] = my_df_cat['Violation'] - 1

  train_df, val_df = train_test_split(my_df_cat, test_size=0.1, random_state=42) 
  train_df.reset_index(inplace=True, drop=True)
  val_df.reset_index(inplace=True, drop=True)

  # Import test set, resetting index
  test_df = pd.read_csv(test_path)
  test_df = test_df[data_columns]

  # Removing label 0, 5, 6 ,7, 8, 12 from the test set.
  test_df = test_df[test_df['Violation'] != 0].copy()

  test_df = test_df[~test_df['Violation'].isin([5, 6, 7, 8, 12])]

  test_df['Violation'] = np.where(test_df['Violation']>=5, test_df['Violation']-4, test_df['Violation'])
  #test_df = test_df.reset_index();
  test_df.reset_index(inplace=True, drop=True)

  test_df['Violation'] = test_df['Violation'].astype('int')
  test_df['Violation'] = test_df['Violation'] - 1

  # Convert training/validation and test sets to TensorFlow format
  train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["Violation"], num_epochs=None, shuffle=False)

  val_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    val_df, val_df["Violation"], shuffle=False)

  test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    test_df, test_df["Violation"], shuffle=False) 

  predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["Violation"], shuffle=False)

  # Define text embedded feature column 'TitleText'
  embedded_text_feature_column = hub.text_embedding_column(
    key="TitleText",  
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

  config = tf.estimator.RunConfig(tf_random_seed=1)

  my_config = tf.estimator.RunConfig(tf_random_seed=42)

  # Define TensorFlow estimator
  estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=7,
    hidden_units=[400, 110],
    config=my_config,
    optimizer=lambda: tf.keras.optimizers.Adam(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=0.001,
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=5000,
            decay_rate=0.96)))

  # Training for 10,000 steps
  estimator.train(input_fn=train_input_fn, steps=10000)

  # # Save trained model
  # serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
  # tf.feature_column.make_parse_example_spec(feature_columns))
  # estimator.export_saved_model("../models/multiclass_binary", serving_input_fn)  

  # Evaluate and print accuracy scores
  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  print("Training set accuracy: {accuracy}".format(**train_eval_result))

  # Evaluate and print accuracy scores
  #eval_result = estimator.evaluate(input_fn=val_input_fn)
  #print("Training set accuracy: {accuracy}".format(**eval_result))

  test_eval_result = estimator.evaluate(input_fn=test_input_fn)
  print("Test set accuracy: {accuracy}".format(**test_eval_result))

  def get_predictions_category(model, input_fn):
    class_id = []
    probabilities = []
    for x in estimator.predict(input_fn = input_fn):
      class_id.append(x['class_ids'][0])
      probabilities.append(x['probabilities'])
    return pd.DataFrame(data = {'Prediction_Category' : class_id, 'Probability': probabilities})

  preds_cat = get_predictions_category("", test_input_fn)
  test_df['Prediction_Category'] = preds_cat['Prediction_Category']
  test_df['Probability'] = preds_cat['Probability']

  violation_codes = {
          #0:'No Violation',
          0:'Rewards Code',
          1:'Profanity',
          2:'Poor Quality/Spam',
          3:'Directs Business Away',
          #5:'Inappropriate Media Content',
          #6:'Vendor Request',
          #7:'Emojis',
          #8:'Other',
          4:'Medical Advice',
          5:'Shipping and Customs',
          6:'Customer Care'
          #12:'Pricing Complaints'
            }
  performance_df = pd.DataFrame({
      'Violation_Category': test_df['Violation'],
      'Prediction_Category': test_df['Prediction_Category']
        }).replace(violation_codes)
  
  # Add performance_df category strings as revisions to test_df
  test_df['Violation_Category'] = performance_df['Violation_Category']
  test_df['Prediction_Category'] = performance_df['Prediction_Category']

  # Combine binary non-violation prediction data with multiclass prediction data
  binary_df = pd.read_csv('predictions/binary_predictions.csv').drop('Unnamed: 0', axis=1)  

  # Create df of only non-violations, creating custom index starting where test_df ends
  binary_neg_df = binary_df.loc[binary_df['Prediction'] == 0]

  # Add required columns to test_df for merging with binary_neg_df
  test_df['ViolationBool'] = 1
  test_df['Prediction']  = 1

  # Add Violation_Category and Prediction_Category to binary_neg_df
  violation_dict = {0.0: 'Clean',
                  1.0: 'Rewards Code',
                  2.0: 'Profanity',
                  3.0: 'Poor Quality/Spam',
                  4.0: 'Directs Business Away', 
                  6.0: 'Vendor Request',
                  8.0: 'Other',
                  9.0: 'Medical Advice',
                  10.0: 'Shipping and Customs',
                  11.0: 'Customer Care',
                  12.0: 'Pricing Complaints'}
  binary_neg_df['Violation_Category'] = binary_neg_df.loc[:, 'Violation'].replace(violation_dict)
  binary_neg_df['Prediction_Category'] = 'Clean'

  # Make column orders consistent
  columns = ['ViolationBool', 'Violation', 'Violation_Category', 
            'Prediction', 'Prediction_Category', 'Probability', 
            'LanguageCode', 'TitleText', 'title_len', 'text_len',
            'titletext_len', 'title_word_no', 'text_word_no', 
            'titletext_word_no','title_avg_word_length', 'text_avg_word_length',
            'titletext_avg_word_length', 'year', 'month', 'hour', 'YYMM']
  binary_neg_df_reorder = binary_neg_df[columns]
  test_df_reorder = test_df[columns]

  # Concatenate multiclass data results and binary data results
  all_preds_df = pd.concat([test_df_reorder, binary_neg_df_reorder], ignore_index=True, sort=False)

  # Export multiclass prediction data    
  test_df.to_csv('predictions/multiclass_predictions.csv')

  # Export multiclass prediction data    
  test_df.to_csv('predictions/multiclass_predictions.csv')
  pickle.dump(test_df, open('predictions/multiclass_predictions.pkl', 'wb'))

  # Export combined prediction data
  all_preds_df.to_csv('predictions/all_predictions.csv')
  pickle.dump(all_preds_df, open('predictions/all_predictions.pkl', 'wb'))




        
    
