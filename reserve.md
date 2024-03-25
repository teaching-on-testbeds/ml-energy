::: {.cell .markdown}

# Prepare resources for the ML energy experiment

:::

::: {.cell .code}
```python
import chi, os

PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
```
:::

