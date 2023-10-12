# %% [markdown]
# # Bankruptcy Prediction Using XGBoost
# 
# The aim is to accurately forecast whether companies will face bankruptcy in the future.

# %%
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay, average_precision_score
from xgboost import XGBClassifier, plot_importance
from category_encoders.woe import WOEEncoder
from tensorflow_addons.activations import sparsemax
from scipy.special import softmax
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import optuna
from optuna import Trial, visualization
import os
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

random_state = 123456
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    
global_seed = random_state
seed_all(global_seed)

# %%
# # Limit GPU Memory in TensorFlow
# physical_devices = tf.config.list_physical_devices('GPU')

# if len(physical_devices) > 0:
#     for device in physical_devices:
#         tf.config.experimental.set_virtual_device_configuration(
#             device,
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)])
#         print('{} memory limit set'.format(device))
# else:
#     print("Not enough GPU hardware devices available")


# %% [markdown]
# ## The Data
# 

# %%
data = pd.read_csv("data.csv")
data = data.drop(data.columns[0], axis=1)
data['status_label'] = data['status_label'].replace({'failed': 1, 'alive': 0}) # 1: failed, 0: alive
data_without_year = data.drop("year", axis=1)

# %%
print("Number of features: ", len(data.columns) - 2)
print("Number of rows: ", len(data))

# %%
# Check if there are some duplicates rows
print("Number of duplicated rows: ", data.duplicated().sum())

# Check if there are some missing values or null values
print(data.isnull().sum())

# %% [markdown]
# ### The distribution of values for each feature

# %%
print(data_without_year.describe())

# %% [markdown]
# ### The correlation of the input features with the target feature

# %%
corr = data_without_year.corr().sort_values(by='status_label', ascending=False)
print(corr['status_label'])

# %% [markdown]
# ### The shape of the data

# %%
labelsCount = data['status_label'].value_counts()
print("Percentage label 0: ", "{:.2f}".format((labelsCount[0] * 100) /
(labelsCount[0] + labelsCount[1])), "%")
print("Percentage label 1: ", "{:.2f}".format((labelsCount[1] * 100) /
(labelsCount[0] + labelsCount[1])), "%")
print(labelsCount)

# %% [markdown]
# The dataset is highly unbalanced.

# %% [markdown]
# ### Pre-processing
# 
# Divide the dataset into two dataframes, one for the input features X and one for the target feature y.
# 
# Divide into training set and test set.

# %%
train_data = data[data['year'] < 2015]
test_data = data[data['year'] >= 2015]

train_data = train_data.drop('year', axis=1)
test_data = test_data.drop('year', axis=1)

# Split the data into training and test sets
X_train = train_data.drop('status_label', axis=1)
y_train = train_data['status_label']

X_test = test_data.drop('status_label', axis=1)
y_test = test_data['status_label']

# # Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_train = pd.DataFrame(X_train, columns=X_test.columns)
# X_test = scaler.transform(X_test)
# X_test = pd.DataFrame(X_test, columns=X_train.columns)

# print("Input feature after feature scaling: \n")
print(pd.DataFrame(X_train).describe())

# %% [markdown]
# ## Modelling - XGBoost
# 
# To systematically compare the performance of different machine learning models, we implemented a function `evaluate_models` that conducts k-fold cross-validation for each model on the dataset and calculates their recall scores. We compared two models in this experiment:
# 
# 1. Random Forest
# 2. XGBoost
# 
# **Methodology**
# - **Data Splitting**: We used k-fold cross-validation with 10 folds to assess the performance of each model. The dataset was split into training and validation sets for each fold.
# - **Metric of Evaluation**: The main metric used for evaluation is the Recall score, which is particularly important for imbalanced classes.
# - **Model Fitting and Evaluation**: For each fold, each model was trained on the training set and evaluated on the validation set. The recall score for each model was computed for every fold.
# 
# **Function Details**
# 1. **`evaluate_models`**: Conducts k-fold cross-validation and computes the recall scores for each model. The average recall score over all folds is used for model comparison.
# 
# 2. **`plot_recall_scores`**: Plots the average recall scores to provide a visual comparison between models.

# %%
MODEL_NAMES = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

def evaluate_models(X, y, num_splits=10):
    """
    Evaluate and compare models based on recall score.
    
    Parameters:
    - X: Features dataset
    - y: Target dataset
    - num_splits: Number of splits for k-fold validation
    
    Returns:
    - None
    """
    print("=" * 10, " Evaluating Models ", "=" * 10)
    
    # Initialize recall scores dictionary
    recall_scores = defaultdict(float)
    
    # Perform k-fold cross-validation
    kfold = KFold(n_splits=num_splits, shuffle=False)
    
    for idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for model_name, model in MODEL_NAMES.items():
            model.fit(X_train, y_train.values.ravel())
            score = recall_score(y_val, model.predict(X_val))
            recall_scores[model_name] += score
            
            print(f"Iteration {idx+1} - Recall for {model_name}: {score:.2f}")
            
    print("-" * 50)
    
    # Calculate average recall scores
    avg_recall = {k: v / num_splits for k, v in recall_scores.items()}
    
    # Identify and print the best model
    best_model = max(avg_recall, key=avg_recall.get)
    print(f"The best model is {best_model} with an average recall of {avg_recall[best_model]:.2f}")
    
    # Plot average recall scores
    plot_recall_scores(avg_recall)

def plot_recall_scores(recall_scores):
    """
    Plot average recall scores for each model.
    
    Parameters:
    - recall_scores: Dictionary containing average recall scores
    
    Returns:
    - None
    """
    colors = ['green' if model == max(recall_scores, key=recall_scores.get) else 'blue' for model in recall_scores]
    plt.bar(recall_scores.keys(), recall_scores.values(), color=colors)
    plt.title("Average Recall Scores")
    plt.xlabel("Models")
    plt.ylabel("Recall")
    plt.show()

# %%
evaluate_models(X_train, y_train)

# %% [markdown]
# The k-fold cross-validation results show that XGBoost is the more appropriate model for our specific use case, based on the Recall metric. However, further tuning may be required to improve the model's performance.

# %% [markdown]
# ### Hyperparameter Tuning
# 
# **Hyperparameter Optimization Strategy**
# To optimize the performance of our XGBoost model, we employed a two-stage strategy for hyperparameter tuning:
# 
# 1. Random Search: We initialized an XGBoost classifier and set up a Random Search with the above hyperparameter space. The search was executed and the best set of hyperparameters.
# 2. Grid Search: Post Random Search, we conducted a Grid Search specifically focusing on the `reg_alpha` parameter to fine-tune its optimal value.
# 
# **Hyperparameter Space**
# The initial hyperparameter space considered for Random Search was as follows:
# 
# - **n_estimators**: Range from 100 to 500 with a step of 100
# - **max_depth**: Range from 3 to 10
# - **min_child_weight**: Range from 1 to 6 with a step of 2
# - **gamma**: Values from 0.0 to 0.4 in steps of 0.1
# - **subsample**: Values from 0.6 to 0.9 in steps of 0.1
# - **colsample_bytree**: Values from 0.6 to 0.9 in steps of 0.1
# - **scale_pos_weight**: Fixed at 30 to account for class imbalance
# - **learning_rate**: Fixed at 0.01
# 
# **Methodology**
# We used 10-fold cross-validation for both Random Search and Grid Search. The scoring metric used was Recall, as we are particularly interested in the True Positive Rate.

# %%
space = {
    'n_estimators': range(100, 500, 100),
    'max_depth': range(3, 10),
    'min_child_weight': range(1, 6, 2),
    'gamma': [i / 10.0 for i in range(0, 5)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'scale_pos_weight': [4],  # because of high class imbalance
    'learning_rate': [0.01]
}

def epoch(i, reg):
    """
    Used to return string with epoch and reg_alpha.
    :param i: is the epoch
    :param reg: is the value of reg_alpha
    :return: string
    """

    return str(i) + " (n=" + str(reg) + ")"

def random_search_all_param(X_train, y_train):
    """
    It deals with choosing the best parameters for the model.
    Use the Random search.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :return: dictionary of the best parameters of the model
    """

    print("=" * 10, " Random Search ", "=" * 10)

    model = XGBClassifier(tree_method = 'gpu_hist', gpu_id=0)
    num_split = 10
    cv = KFold(n_splits=num_split, random_state=None, shuffle=False)
    # RandomSearch
    search = RandomizedSearchCV(model, space, random_state=0, scoring='recall', n_jobs=-1, cv=cv, return_train_score=True)
    result = search.fit(X_train, y_train)

    print("Best Score: ", result.best_score_)
    print("Best Hyperparameters: ", result.best_params_)

    print(" ------------------------------------------------------- ")

    cv_result = pd.DataFrame(result.cv_results_)
    print(cv_result)

    # Plot accuracy during training and validation
    epochs = [i for i in range(0, len(cv_result['mean_train_score']))]
    plt.plot(epochs, cv_result['mean_train_score'], 'g', label='Training score')
    plt.plot(epochs, cv_result['mean_test_score'], 'b', label='Validation score')
    plt.title('Training and Validation score')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print(" ------------------------------------------------------- ")

    return result.best_params_


def number_estimators(X_train, y_train, best_param):
    """
    It deals with choosing the best number of estimators around a range of estimators for the model.
    Use the Grind search.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :param best_param: dictionary of the best parameters of the model selected by RandomSearch
    :return: dictionary of the best parameters of the model
    """

    print("=" * 10, " Grind Search ", "=" * 10)

    model = XGBClassifier(tree_method = 'gpu_hist', gpu_id=0)
    model.set_params(**best_param)
    number_split = 10
    cv = KFold(n_splits=number_split, random_state=None, shuffle=False)

    space = dict()
    space['reg_alpha'] = [1e-5, 1e-2, 0.1, 1, 100, 200]
    # GrindSearch
    search = GridSearchCV(model, space, n_jobs=-1, cv=cv, scoring='recall', return_train_score=True)
    result = search.fit(X_train, y_train)

    print("Best Score: ", result.best_score_)
    print("Best Hyperparameters: ", result.best_params_)

    print(" ------------------------------------------------------- ")

    cv_result = pd.DataFrame(result.cv_results_)
    print(cv_result)

    # Plot accuracy during training and validation
    epochs = [epoch(i,  cv_result['param_reg_alpha'][i]) for i in range(0, len(cv_result['mean_train_score']))]
    plt.plot(epochs, cv_result['mean_train_score'], 'g', label='Training accuracy')
    plt.plot(epochs, cv_result['mean_test_score'], 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return result.best_params_['reg_alpha']


def hyperparameter_optimization(X_train, y_train):
    """
    The function is concerned with finding the best parameters for the model.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :return: dictionary of the best parameters of the model
    """

    best_param = random_search_all_param(X_train, y_train)
    best_param['reg_alpha'] = number_estimators(X_train, y_train, best_param)

    return best_param

# %%
def get_best_parameters_csv():
    """
    This function takes care of obtaining the best parameters used by the "best_param.csv" file.
    :return: dictionary of parameters
    """
    best = pd.read_csv('best_params_xgboost.csv', index_col=0)
    dict_best_param = {}

    for i in best.to_dict('records'):
        dict_best_param = dict(i)

    d = dict_best_param.copy()
    for key in dict_best_param:
        if str(dict_best_param[key]) == 'nan' or key not in space:
            d.pop(key)

    return d

# %%
# True: research the best model parameters.
# False: load the best parameters found during past searches from the csv file.
search_best_parameter = True

if search_best_parameter:
    best_params = hyperparameter_optimization(X_train, y_train)
else:
    best_params = get_best_parameters_csv()

# %% [markdown]
# ### Testing

# %%
def write_best_param(param, recall):
    """
    This function takes care of writing the best parameters found in the "best_params_xgboost.csv" file
    if the recall is better than the past recall.
    :param param: param of model
    :param recall: recall score of model
    """
    param['Recall'] = recall

    try:
        csv = pd.read_csv('best_params_xgboost.csv')
        csv_param = {}

        # convert into records [{...}]
        for i in csv.to_dict('records'):
            csv_param = dict(i)

        if recall > csv_param['Recall']:
            pd.DataFrame(param, index=[0]).to_csv('best_params_xgboost.csv')
            print("\n Write into best_param.csv")
    except:
        pd.DataFrame(param, index=[0]).to_csv('best_params_xgboost.csv')
        print("\n Write into best_param.csv")

def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params):
    """
    Train and evaluate an XGBoost model. Optionally use SMOTE for handling class imbalance.
    
    Parameters:
    - X_train: DataFrame, training feature set
    - X_test: DataFrame, test feature set
    - y_train: Series, training labels
    - y_test: Series, test labels
    - best_params: dict, best parameters for the model
    
    Returns:
    - None, prints evaluation metrics and plots relevant graphs
    """
    
    print("=" * 20, " XGBoost ", "=" * 20)
    
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive outcome
    recall = recall_score(y_test, y_pred)
    
    print(f"\nAccuracy: {'{:.2f}'.format(accuracy_score(y_test, y_pred) * 100)}%")
    print(f"Recall score: {'{:.2f}'.format(recall_score(y_test, y_pred) * 100)}%")
    print(f"Train score: {model.score(X_train, y_train)}")
    print(f"Test score: {model.score(X_test, y_test)}")
    print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['class 0', 'class 1']).plot()
    plt.show()
    
    plot_importance(model)
    plt.show()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    print(f"AUC: {'{:.2f}'.format(auc * 100)}%")
    print(f"AUC PR: {'{:.2f}'.format(auc_pr * 100)}%")
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='XGBoost Model').plot()
    plt.show()

    write_best_param(model.get_params(), recall)

# %%
train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params)

# %% [markdown]
# ## Modelling - TabNet
# 
# ### Train/Validation Split

# %%
train_data = data[data['year'] < 2012]
val_data = data[(data['year'] >= 2012) & (data['year'] < 2015)]
test_data = data[data['year'] >= 2015]

train_data = train_data.drop('year', axis=1)
val_data = val_data.drop('year', axis=1)
test_data = test_data.drop('year', axis=1)

X_train = train_data.drop('status_label', axis=1)
y_train = train_data['status_label']

X_val = val_data.drop('status_label', axis=1)
y_val = val_data['status_label']

X_test = test_data.drop('status_label', axis=1)
y_test = test_data['status_label']

# %% [markdown]
# ### Tenserflow Data

# %%
def prepare_tf_dataset(
    X,
    batch_size,
    y = None,
    shuffle = False,
    drop_remainder = False,
):
    size_of_dataset = len(X)
    if y is not None:
        y = tf.one_hot(y.astype(int), 2)
        ds = tf.data.Dataset.from_tensor_slices((np.array(X.astype(np.float32)), y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(np.array(X.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=size_of_dataset)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    autotune = tf.data.experimental.AUTOTUNE
    ds = ds.prefetch(autotune)
    return ds

train_ds = prepare_tf_dataset(X_train, 16384, y_train)
val_ds = prepare_tf_dataset(X_val, 16384, y_val)
test_ds = prepare_tf_dataset(X_test, 16384, y_test)

# %% [markdown]
# ### TabNet

# %%
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])

# %%
class FeatureBlock(tf.keras.Model):
    """
    Implementation of a FL->BN->GLU block
    """
    def __init__(
        self,
        feature_dim,
        apply_glu = True,
        bn_momentum = 0.9,
        fc = None,
        epsilon = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim 

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc # shared layers can get re-used
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=epsilon)

    def call(self, x, training = None):
        x = self.fc(x) 
        x = self.bn(x, training=training) 
        if self.apply_gpu: 
            return glu(x, self.feature_dim) # GLU activation applied to BN output
        return x

    
class FeatureTransformer(tf.keras.Model):
    def __init__(
        self,
        feature_dim,
        fcs = [],
        n_total = 4,
        n_shared = 2,
        bn_momentum = 0.9,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kwrgs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
        }

        # build blocks
        self.blocks = []
        for n in range(n_total):
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kwrgs, fc=fcs[n])) 
            else:
                self.blocks.append(FeatureBlock(**kwrgs)) 

    def call(self, x, training = None):
        # input passes through the first block
        x = self.blocks[0](x, training=training) 
        # for the remaining blocks
        for n in range(1, self.n_total):
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training) 
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]
    
class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return sparsemax(x * prior_scales)
    
class TabNet(tf.keras.Model):
    def __init__(
        self,
        num_features,
        feature_dim,
        output_dim,
        n_step = 2,
        n_total = 4,
        n_shared = 2,
        relaxation_factor = 1.5,
        bn_epsilon = 1e-5,
        bn_momentum = 0.7,
        sparsity_coefficient = 1e-5
    ):
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum
        }

        self.feature_transforms = [FeatureTransformer(**kargs)]
        self.attentive_transforms = []
            
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features)
            )
        
        # Final output layer
        self.head = tf.keras.layers.Dense(2, activation="softmax", use_bias=False)

    def call(self, features, training = None):

        bs = tf.shape(features)[0] 
        out_agg = tf.zeros((bs, self.output_dim)) 
        prior_scales = tf.ones((bs, self.num_features)) 
        importance = tf.zeros([bs, self.num_features]) # importances
        masks = []

        features = self.bn(features, training=training) 
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](
                masked_features, training=training
            )
            
            if step_i > 0:
                # first half of the FT output goes towards the decision 
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out
                scale_agg = tf.reduce_sum(out, axis=1, keepdims=True) / (self.n_step - 1)
                importance += mask_values * scale_agg
                

            if step_i < self.n_step:
                # second half of the FT output goes as input to the AT
                x_for_mask = x[:, self.output_dim :]
                
                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )

                prior_scales *= self.relaxation_factor - mask_values
                
                # multiply the second half of the FT output by the attention mask to enforce sparsity
                masked_features = tf.multiply(mask_values, features)

                # penalize the amount of sparsity
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(-mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )
                
                # append mask
                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))
                   
        self.selection_masks = masks
        
        final_output = self.head(out)
        
        # Add sparsity loss
        loss = total_entropy / (self.n_step-1)
        self.add_loss(self.sparsity_coefficient * loss)
        
        return final_output, importance

# %% [markdown]
# ### Hyperparameter Tuning

# %%
def Objective(trial):
    feature_dim = trial.suggest_categorical("feature_dim", [32, 64, 128, 256, 512])
    n_step = trial.suggest_int("n_step", 2, 9, step=1)
    n_shared = trial.suggest_int("n_shared", 0, 4, step=1)
    relaxation_factor = trial.suggest_float("relaxation_factor", 1., 3., step=0.1)
    sparsity_coefficient = trial.suggest_float("sparsity_coefficient", 0.00000001, 0.1, log=True)
    bn_momentum = trial.suggest_float("bn_momentum", 0.9, 0.9999)
    tabnet_params = dict(num_features=X_train.shape[1],
                         output_dim=feature_dim,
                         feature_dim=feature_dim,
                         n_step=n_step, 
                         relaxation_factor=relaxation_factor,
                         sparsity_coefficient=sparsity_coefficient,
                         n_shared = n_shared,
                         bn_momentum = bn_momentum
                     )
    class_weight = trial.suggest_float("class_weight", 1, 10.5, step=0.5)
    
    cbs = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )]
    
    tn = TabNet(**tabnet_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,clipnorm=10)
    loss = [tf.keras.losses.CategoricalCrossentropy(from_logits=False),None]
    
    tn.compile(
            optimizer,
            loss=loss)

    tn.fit(train_ds, 
          epochs=100, 
          validation_data=val_ds,
          callbacks=cbs,
          verbose=1)
    
    
    val_preds, _ =  tn.predict(val_ds)
    pr_auc = average_precision_score(y_val, val_preds[:,1])
    
    return pr_auc

# %%
def read_best_params_from_csv(filename='best_params_tabnet.csv'):
    """
    Read best parameters from a CSV file.
    
    :param filename: Name of the CSV file to read from.
    :return: Dictionary containing the best parameters if file exists, else empty dictionary.
    """
    try:
        csv_params = pd.read_csv(filename).to_dict('records')[0]
        return csv_params
    except FileNotFoundError:
        return {}

def write_best_params_to_csv(params, filename='best_params_tabnet.csv'):
    """
    Write best parameters to a CSV file.
    
    :param params: Dictionary containing the best parameters.
    :param filename: Name of the CSV file to write to.
    """
    pd.DataFrame([params]).to_csv(filename, index=False)


# %%
FINE_TUNE = True

if FINE_TUNE:
    study = optuna.create_study(direction="maximize", study_name='TabNet optimization')
    study.optimize(Objective, n_jobs=1, n_trials=100, gc_after_trial=True, show_progress_bar=True)
    best_params = study.best_params
    write_best_params_to_csv(best_params)
else:
    best_params = read_best_params_from_csv()


# %% [markdown]
# ### Training

# %%
tabnet = TabNet(num_features = X_train.shape[1],
                output_dim = 256,
                feature_dim = best_params['feature_dim'],
                n_step = best_params['n_step'], 
                relaxation_factor = best_params['relaxation_factor'],
                sparsity_coefficient = best_params['sparsity_coefficient'],
                n_shared = best_params['n_shared'],
                bn_momentum = best_params['bn_momentum'])


# Early stopping based on validation loss    
cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=10)

loss = [tf.keras.losses.CategoricalCrossentropy(from_logits=False), None]

tabnet.compile(optimizer,
               loss=loss)

tabnet.fit(train_ds, 
           epochs=1000, 
           validation_data=val_ds,
           callbacks=cbs,
           verbose=1,
           class_weight={0: 1, 1: best_params['class_weight']})

# %%
test_preds, test_imps = tabnet.predict(test_ds)

nan_mask = np.isnan(test_preds).any(axis=1)

test_preds_clean = test_preds[~nan_mask, :]
y_test_clean = y_test[~nan_mask]

test_preds_class_clean = np.argmax(test_preds_clean, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_test_clean, test_preds_class_clean))

test_preds_positive_class_clean = test_preds_clean[:, 1]

roc_auc = roc_auc_score(y_test_clean, test_preds_positive_class_clean)

print(f"ROC-AUC: {roc_auc}")

# %%
test_imps_df = pd.DataFrame(test_imps, columns=X_train.columns)

mean_importance = test_imps_df.mean().sort_values(ascending=False)

mean_importance.nlargest(20).sort_values().plot(kind='barh', figsize=(20, 10))
plt.xlabel('Importance')
plt.title('Top 20 Features')
plt.show()

mean_importance


