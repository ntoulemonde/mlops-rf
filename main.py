# %%
# Import modules
# Inspired from
import mlflow
import plotnine as p9
import polars as pl
import seaborn as sns
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
X = titanic_df_sex.select(features)
y = titanic_df.with_columns(
    pl.when(pl.col("sex") == "female").then(1).otherwise(0).alias("survived")
).select(["survived"])

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
            pl.concat([X, y, titanic_df.select("id")], how="horizontal"),
            source="train_data.csv",
        ),
        context="training dataset",
    )

# %%
# Select features and target variable
features = [
    "pclass",
    "sex",
    "age",
    "sibSp",
    "parch",
    "fare",
    "embarked",
]
X_train = df_train_sex.select(features)
y_train = df_train_sex.select(["Survived"])

# %%
columns_to_encode = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
label_encoder = LabelEncoder()
for this_column_to_encode in columns_to_encode:
    X_train = X_train.with_columns(
        pl.col(this_column_to_encode).map_batches(label_encoder.fit_transform)
    )

# %%
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
