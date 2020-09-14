import pandas
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, validation_curve



def extractData(data_file):
       data = pd.read_csv(data_file)
       X = data.iloc[:, :-1]
       Y = data.iloc[:, -1].values
       return X, Y


def plotPerformance(actual, predicted, label, title):
     conf_matrix = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
     print(conf_matrix)
     print(conf_matrix.to_latex())

     y_max = max(actual)
     plt.figure(figsize=(6, 4))
     plt.scatter(actual, predicted)
     plt.plot([0, y_max], [0, y_max], '--k')
     plt.axis('tight')
     plt.xlabel('True {}'.format(label))
     plt.ylabel('Predicted {}'.format(label))
     plt.tight_layout()
     plt.title(title)
     plt.show()

def plotLearningCurve(estimator, data,  model,   X, Y):
    train_sizes, train_scores, validation_scores, fit_times, _ = learning_curve(model, X, Y, train_sizes=np.linspace(0.1,1,10),return_times=True)
    train_scores_mean = np.mean(train_scores,1)
    validation_scores_mean = np.mean(validation_scores, 1)
    fit_times_mean = np.mean(fit_times,1)

    _, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].plot(train_sizes, train_scores_mean, label='Training score')
    axes[0].plot(train_sizes, validation_scores_mean, label='Validation score')
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('Training set size')
    title = 'Learning curve for a ' + str(estimator).split('(')[0] + ' model for ' + data
    axes[0].set_title(title, fontsize=18, y=1.03)
    axes[0].legend()

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    plt.show()

def plotValidationCurve(estimator, data,  grid_search,   X, Y, param_grid):
    df = pd.DataFrame(grid_search.cv_results_)
    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score',
               'std_train_score']
    fig, ax = plt.subplots(1, len(param_grid),
                             figsize=(5 * len(param_grid), 7),
                             sharey='row')
    axes = []
    if len(param_grid) == 1:
        axes.append(ax)
    else:
        axes = ax

    axes[0].set_ylabel("Score", fontsize=25)
    for idx, (param_name, param_range) in enumerate(param_grid.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results] \
            .agg({'mean_train_score': 'mean',
                  'mean_test_score': 'mean',
                  'std_train_score': 'std',
                  'std_test_score': 'std'})

        previous_group = df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=20)
        axes[idx].set_ylim(0.0, 1.1)
        lw = 2
        axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                       color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                       color="navy", lw=lw)


    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=30)
    fig.legend(handles, labels, loc='upper right', ncol=1, fontsize=20)
    plt.show()



    # grouped_df = df_gs.groupby(f'param_{param_name}')[results] \
    #     .agg({'mean_train_score': 'mean',
    #           'mean_test_score': 'mean',
    #           'std_train_score': 'std',
    #           'std_test_score': 'std'})
    #
    # plt.plot(param_values, grouped_df['mean_train_score'], label='Training score')
    # plt.plot(param_values, grouped_df['mean_test_score'], label='Cross-Validation score')
    #
    # plt.ylabel('Score', fontsize=14)
    # plt.xlabel(param_name, fontsize=14)
    # title = 'Validation curve with ' + str(estimator).split('(')[0]
    # plt.title(title, fontsize=18, y=1.03)
    # plt.legend()
    # plt.show()

def getBestModel(classifyAlgorithm, parameter_grid, train_x, train_y):
       grid_search = GridSearchCV(classifyAlgorithm, param_grid=parameter_grid, cv=5, return_train_score=True, n_jobs=-1,  verbose=5)
       grid_search.fit(train_x, train_y)
       print('Best params : {}'.format(grid_search.best_params_))
       classifier = grid_search.best_estimator_
       return classifier, grid_search