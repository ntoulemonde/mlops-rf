# %%
# Import modules
# Inspired from
import mlflow
import plotnine as p9
import polars as pl
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %%
# Set up mlflow

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(experiment_name="Titanic")


# %%
# Import data
# Inspired from Kaggle https://www.kaggle.com/c/titanic/data
titanic_df = pl.from_pandas(sns.load_dataset("titanic"))
# Add an id
titanic_df = titanic_df.with_columns(id=titanic_df.hash_rows(seed=202602) + "")

titanic_df.describe()

# %%
# Plot number of persons who survived
(p9.ggplot(titanic_df) + p9.aes("survived") + p9.geom_bar())
# %%
# Survived given Sex
(p9.ggplot(titanic_df) + p9.aes("survived") + p9.geom_bar() + p9.facet_wrap("sex"))

# %%
print(
    f"% of women that survived is : {(titanic_df.filter(pl.col('sex') == 'female').select('survived').sum() / titanic_df.filter(pl.col('sex') == 'female').select('survived').count())[0, 0]:.3f}"
)
print(
    f"% of men that survived is : {(titanic_df.filter(pl.col('sex') == 'male').select('survived').sum() / titanic_df.filter(pl.col('sex') == 'male').select('survived').count())[0, 0]:.3f}"
)

# Women have survived at 74% when men at 19%
# %%
# Looking at missing values
titanic_df.null_count()

# %%
# Need to fill age. Cabin also empty but not useful
titanic_df = titanic_df.with_columns(
    pl.col("age").fill_null(titanic_df.select("age").median()[0, 0])
)
titanic_df.describe()
# Estimate a model that says that all women survived and men not
# %%
# Select features and target variable
features = ["sex"]
X = titanic_df.select(features)
y = titanic_df.with_columns(
    pl.when(pl.col("sex") == "female").then(1).otherwise(0).alias("survived_mod")
).select(["survived_mod"])

# %%
columns_to_encode = ["sex"]
label_encoder = LabelEncoder()
for this_column_to_encode in columns_to_encode:
    X = X.with_columns(
        pl.col(this_column_to_encode).map_batches(label_encoder.fit_transform)
    )

# %%
model = LogisticRegression()
model.fit(X, y)

# %%
# Test with the real data
y_pred = model.predict(X)

# Calculate the accuracy of the model by comparing with real data
accuracy = accuracy_score(titanic_df.select("survived"), y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# %%
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "Logistic Regression based on only women survived")
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_input(
        mlflow.data.from_polars(
            pl.concat(
                [
                    X,
                    y,
                    titanic_df.select("id"),
                    pl.DataFrame({"survived_pred": y_pred}),
                ],
                how="horizontal",
            ),
            targets="survived_mod",
            predictions="survived_pred",
        ),
        context="Modified dataset with women only who survived",
    )


# %%
(p9.ggplot(titanic_df) + p9.aes("survived") + p9.geom_bar() + p9.facet_wrap("pclass"))
# %%
(p9.ggplot(titanic_df) + p9.aes("survived") + p9.geom_bar() + p9.facet_wrap("embarked"))

# %%
(
    p9.ggplot(
        titanic_df.with_columns(
            fare=pl.when(pl.col("fare") < 1).then(1).otherwise("fare")
        )
    )
    + p9.aes("fare", group="survived", fill="survived")
    + p9.geom_histogram()
    + p9.scale_x_log10()
)
# %%
(
    p9.ggplot(titanic_df)
    + p9.aes("survived", "fare", colour="survived")
    + p9.geom_point(position=p9.position_jitter(width=0.2, height=0.2))
)
# %%
(
    p9.ggplot(titanic_df)
    + p9.aes("age", group="survived", fill="survived")
    + p9.geom_histogram()
)

# %%
# Select features and target variable
features = ["pclass", "sex", "age", "sibsp", "fare", "id"]
X = titanic_df.select(features)
y = titanic_df.select(["survived"])

# %%
columns_to_encode = ["pclass", "sex", "sibsp", "pclass"]
label_encoder = LabelEncoder()
for this_column_to_encode in columns_to_encode:
    X = X.with_columns(
        pl.col(this_column_to_encode).map_batches(label_encoder.fit_transform)
    )

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=202602
)
# %%
# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train.drop("id").to_pandas().values, y_train.to_pandas().values)

# %%
from matplotlib import pyplot as plt

tree.plot_tree(clf, proportion=True)
plt.show()
# %%
y_pred = clf.predict(X_test.drop("id").to_pandas().values)
accuracy = accuracy_score(y_test.to_pandas(), y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# %%
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "Decision Tree Classifier - true data")
    mlflow.sklearn.log_model(clf, "Decision Tree Classifier")
    mlflow.log_params(clf.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_input(
        mlflow.data.from_polars(
            pl.concat(
                [
                    X,
                    y,
                ],
                how="horizontal",
            ),
            targets="survived",
        ),
        context="Original dataset",
    )

# %%
