from data.create_data_sets import create_data
from models.build_model import train_model
from models.build_multiclass_model import train_multiclass_model
from models.build_multiclass_model_from_binary import train_multiclass_binary_model
from results.show_model_results import show_classification_report, show_multiclassification_report
from results.show_model_results import print_confusion_matrix, plot_roc_curve, print_auc_score


# # Create balanced_test, unbalanced_train, unbalanced_valid and oot_validation_202008_202009 data sets and save to '../data/processed' folder
# create_data()

# Run script to train binary model
train_model('../data/processed/balanced_test.csv', '../data/processed/live_test_set.csv')

# # Run script to train multiclass model
# train_multiclass_model('../data/processed/balanced_test.csv', '../data/processed/live_test_set.csv')

# # Run script to train multiclass model on binary model results
# train_multiclass_binary_model('predictions/binary_predictions.csv', '../data/processed/unbalanced_valid.csv')

# # Show binary classification report
# show_classification_report('predictions/binary_predictions.csv')

# # Show multiclassification report
# show_multiclassification_report('predictions/multiclass_predictions.csv')

# # Show binary confusion matrix
# print_confusion_matrix('predictions/binary_predictions.csv')

# # Show binary ROC curve 
# plot_roc_curve('predictions/binary_predictions.csv')

# # Show binary AUC score
# print_auc_score('predictions/binary_predictions.csv')
