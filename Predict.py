
def predict(
    site_id: str,
    issue_date: str,
    assets,
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:
    date = issue_date[-5:]
    input_df = assets['features']
    train_input = input_df[(input_df['WY'] == issue_date) & (input_df['site_id'] == site_id)]
    pred_model_10 = assets['models'][date][0]
    pred_model_50 = assets['models'][date][1]
    pred_model_90 = assets['models'][date][2]

    pred10 = pred_model_10.predict(train_input)
    pred50 = pred_model_50.predict(train_input)
    pred90 = pred_model_90.predict(train_input)

    return pred10, pred50, pred90

