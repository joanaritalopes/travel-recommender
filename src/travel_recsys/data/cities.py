import pandas as pd
import csv

from travel_recsys.config import DATA_INTERIM, DATA_RAW

GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

KEEP_COLUMNS = [
    "geonameid", "name", "asciiname", "country_code",
    "latitude", "longitude", "population",
]



def load_data(path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=GEONAMES_COLUMNS,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )


def select_candidates(df, min_population = 30000, max_per_country=30) -> pd.DataFrame:
    selected = (df['population'] >= min_population) & (df['feature_code'] != 'PPLX')
    return (
        df.loc[selected]
        .sort_values('population', ascending=False)
        .groupby('country_code')
        .head(max_per_country)
        .loc[:, KEEP_COLUMNS]
        .reset_index(drop=True)
    )


def main():
    df = load_data(DATA_RAW / 'cities15000.txt')
    selected = select_candidates(df)

    out_path = DATA_INTERIM / 'selected_cities.csv'
    selected.to_csv(out_path, index=False)

    print(f"{len(selected)} cities across {selected['country_code'].nunique()} countries")
    print(f"saved to {out_path}")



if __name__ == "__main__":
    main()
