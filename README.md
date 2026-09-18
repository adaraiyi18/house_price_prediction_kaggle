# House Price Prediction Model
Predicting house prices for an Iowa-based real estate investment firm, with the goal of spotting homes that are underpriced relative to their features (bedrooms, lot size, year built, etc.), so the firm can buy below market and resell at a profit. This model predicts what a house should sell for based on its features, so a big gap between predicted price and listed price can flag a deal worth investigating. This was a project from the Possible Tech Career Accelerator.

## Datasets
- Source: [Kaggle Home Data for ML Course](https://www.kaggle.com/competitions/home-data-for-ml-course)
- Location: Ames, Iowa
- ``train.csv``: 1,460 home sales, 81 columns (80 features + ``SalePrice``)
- ``test.csv``:1,459 home sales, 80 columns with no ``SalePrice``

Working assumptions: past sale prices are representative of future prices, and this Ames-specific data generalizes to the Ames housing market. It may not transfer well to other cities.
  
## Model Architecture
- Algorithm: Decision Tree Regressor ````sklearn.tree.DecisionTreeRegressor````
- Target: ``SalePrice``
- Features used:
  - ``GrLivArea`` - above-grade living area (sq ft)
  - ``YearBuilt`` - original construction year
  - ``OverallQual`` - overall material/finish quality
  - ``BedroomAbvGr`` - bedrooms above grade
  - ``FullBath`` - full bathrooms above grade
  - ``LotArea`` - lot size (sq ft)
- ``random_state=1`` is set so results are reproducible

## Training and testing models
Two rounds were run:
- Naive check: ``housing_model.fit(X, y)`` trained and tested on the same training data. This is not a real test — the model is graded on houses it already memorized, which is why the error below looks unrealistically small.
- Real test: ``train_test_split(X, y, random_state=0)`` held back a validation set the model never saw during training. The model was trained on ``train_X``/``train_y`` and evaluated on the unseen ``val_X``/``val_y``. This is the number that actually reflects how the model performs on new houses.

## Evaluation Metrics
The metrics used is Mean Absolute Error (MAE) on average, how many dollars off each prediction was from the actual sale price.

| Version | Features | MAE | Notes |
|---|---|---|---|
| Train-on-train (not a real test) | 5 | $73.31 | Tested on data the model already saw — unrealistically low |
| Validation split | 5 | $32,240.31 | First honest test, on unseen houses |
| Validation split | 6 (added OverallQual) | $27,017.21 | Overall build/finish quality improved accuracy |

## Exploratory Findings
- Sale prices are right-skewed — most homes cluster in the low-to-mid range, with a long tail of expensive outliers.
- Year built skews recent, with a large jump in homes built around the 2000s versus earlier decades.
- Neighborhood with the most home sales in the dataset: North Ames.
- A home's finish quality drives how much was spent getting it market-ready, which should track with sale price — and the data backed that up, cutting error by ~$5,200
- Given prices topped out under $1M in the data, the model shouldn't be trusted to price homes above that — it has never seen an example that high.
- The model shows real improvement (MAE dropped from ~$32K to ~$27K) but the remaining error is too large to trust on its own for investment decisions. This model should be used as an initial screening layer to surface candidate houses, followed by manual due diligence before any purchase decision.

## Next Steps
- Try more features, or a different model (e.g., Random Forest), to push MAE down further
- Handle non-numeric features (e.g., Neighborhood) via one-hot encoding
- Re-evaluate before trusting the model on homes priced above $1M, outside the range it was trained on
