# %%
# Import modules
# Inspired from
import plotnine as p9
import polars as pl

# %%
# Import data
# From Kaggle https://www.kaggle.com/c/titanic/data
df_train = pl.read_csv("data/train.csv")
df_test = pl.read_csv("data/test.csv")

df_train.describe()
df_test.describe()
# %%
df_train.describe()

# %%
# Plot number of persons who survived
(p9.ggplot(df_train) + p9.aes("Survived") + p9.geom_bar())
# %%
# Survived given Sex
(
    p9.ggplot(df_train)
    + p9.aes("Survived", group="Sex")
    + p9.geom_bar()
    + p9.facet_wrap("Sex")
)

# %%
print(
    f"% of women that survived is : {(df_train.filter(pl.col('Sex') == 'female').select('Survived').sum() / df_train.filter(pl.col('Sex') == 'female').select('Survived').count())[0, 0]:.3f}"
)
print(
    f"% of men that survived is : {(df_train.filter(pl.col('Sex') == 'male').select('Survived').sum() / df_train.filter(pl.col('Sex') == 'male').select('Survived').count())[0, 0]:.3f}"
)

# Women have survived at 74% when men at 19%

# %%
