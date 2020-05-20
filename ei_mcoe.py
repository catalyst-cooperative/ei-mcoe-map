"""Functions for compiling EI MCOE dataframes."""

# ----------------------------------------------------------
# ---------------------- Package Imports -------------------
# ----------------------------------------------------------

import os
import pudl
import numpy as np
import pandas as pd
from scipy import stats
import sqlalchemy as sa
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask.distributed import Client
#import ei_constants as c

import logging
logger = logging.getLogger(__name__)

# Below, for TESTING PURPOSES

#pudl_settings = pudl.workspace.setup.get_defaults()
#pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
#pudl_out = pudl.output.pudltabl.PudlTabl(pudl_engine, freq='AS',rolling=True)


# ----------------------------------------------------------
# -------------------- Global Variables --------------------
# ----------------------------------------------------------

input_dict = {
    'plant_index_cols': ['plant_id_pudl',
                         'report_year'],
    'plant-fuel_index_cols': ['plant_id_pudl',
                              'fuel_type_code_pudl',
                              'report_year'],
    'unit-fuel_index_cols': ['plant_id_pudl',
                             'plant_id_eia',
                             'unit_id_pudl',
                             'fuel_type_code_pudl',
                             'report_year'],
    'merge_cols_qual': ['plant_name_eia',
                        'state',
                        'city',
                        'county',
                        'latitude',
                        'longitude'],
    'merge_cols_simple': ['fuel_type_code_pudl'],
    'eia_sum_cols': ['total_fuel_cost',
                     'net_generation_mwh',
                     'capacity_mw',
                     'total_mmbtu'],
    'eia_pct_cols': ['net_generation_mwh',
                     'capacity_mw'],
    'ferc_sum_cols': ['capex_total',
                      'opex_fuel',
                      'opex_production_total']
}

eia_wa_col_dict = {
    'generator_age_years': 'capacity_mw',
    'heat_rate_mmbtu_mwh': 'net_generation_mwh'
}

fuel_types = ['coal', 'gas', 'oil', 'waste']

greet_tech_list = ['Boiler',
                   'IGCC',
                   'Combined_Cycle',
                   'Gas_Turbine',
                   'ICE',
                   'Not_Specified']

tech_rename_greet = {'Conventional Steam Coal': 'Boiler',
                     'Coal Integrated Gasification Combined Cycle': 'IGCC',
                     'Natural Gas Fired Combined Cycle': 'Combined_Cycle',
                     'Natural Gas Steam Turbine': 'Gas_Turbine',
                     'Petroleum Coke': 'Boiler',
                     'Petroleum Liquids': 'ICE',
                     'Landfill Gas': 'Boiler',  # Might change to ICE/Turbine
                     'Wood/Wood Waste Biomass': 'Boiler'}

nerc_regions = ['US', 'ASCC', 'FRCC', 'HICC', 'MRO', 'NPCC',
                'RFC', 'SERC', 'SPP', 'TRE', 'WECC']

# PM2.5 values for certain generator types according to GREET
# UNITS: g/kWh
pm_tech_dict = {
    'coal_Boiler': [0.043, 3.504, 0.131, 0.118, 0.110, 0.185, 0.257, 0.180,
                    0.161, 0.110, 0.247],
    'coal_IGCC': [0.008, 0.008, 0.568, 0.008, 0.008, 0.008, 0.008, 0.008,
                  0.008, 0.008, 0.008],
    'coal_Not_Specified': 0.026,
    'oil_Gas Turbine': [0.070, 0.039, 0.079, 0.081, 0.122, 0.083, 0.089, 0.087,
                        0.155, 0.083, 0.067],
    'oil_ICE': [0.012, 0.039, 0.079, 0.081, 0.122, 0.083, 0.089, 0.087, 0.155,
                0.083, 0.067],
    'oil_Boiler': [0.132, 0.039, 0.079, 0.081, 0.122, 0.083, 0.089, 0.087,
                   0.155, 0.083, 0.067],
    'oil_Not_Specified': 0.071,
    'gas_Combined_Cycle': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001],
    'gas_Gas_Turbine': [0.036, 0.038, 0.036, 0.026, 0.040, 0.036, 0.049, 0.036,
                        0.036, 0.043, 0.037],
    'gas_ICE': [0.455, 0.455, 0.483, 0.455, 0.528, 0.524, 0.510, 0.492, 0.357,
                0.397, 0.414],
    'gas_Boiler': [0.041, 0.041, 0.023, 0.041, 0.041, 0.038, 0.040, 0.053,
                   0.041, 0.040, 0.045],
    'gas_Not_Specified': 0.133,
    'waste_Boiler': [0.610, 1.235, 2.495, 2.021, 1.804, 1.836, 1.904, 2.014,
                     1.271, 0.610, 2.053],
    'waste_Not_Specified': 0.610
}

# EPA's determination of monetary damages per tons of polllutant emitted.
# UNITS: 2006 U.S. dollars / ton of pollutant emitted
# https://www.epa.gov/benmap/response-surface-model-rsm-based-benefit-ton-estimates
benmap_value = {'sox': 100000,
                'nox': 19000,
                'pm2.5': 570000}

# EPA's value of a life
# https://www.epa.gov/environmental-economics/mortality-risk-valuation#whatvalue
epa_life_value = 7400000

# How to rename columns for the final output
name_clean_dict = {
    'plant_id_eia': 'EIA Plant Code',
    'plant_id_pudl': 'PUDL Plant Code',
    'plant_name_eia': 'Plant Name',
    'unit_id_pudl': 'PUDL Unit ID',
    'report_year': 'Report Year',
    'generator_id': 'EIA Generator ID',
    'eia_unit_count': 'Number of EIA units under this PUDL Code',
    'ferc1_unit_count': 'Number of FERC Form 1 Entries under this PUDL Code',
    'capacity_mw': 'MW Nameplate Capacity',
    'weighted_ave_generator_age_years': 'Age',
    'retirement_date': 'Retirement Date',
    'fuel_type_code_pudl': 'Broad Fuel Classification',
    'energy_source_code_1': 'Technology',
    'total_mmbtu': 'Annual Total Fuel Consumption MMBtu',
    'net_generation_mwh': 'Annual Electricity Net Generation MWh',
    'weighted_ave_heat_rate_mmbtu_mwh': 'Heat Rate MMBtu/MWh',
    'sig_hr': 'Significant Heat Rate Discrepancy?',
    'max_min_hr_diff': 'Difference Between Maximum and Minimum Unit Heat Rates',
    'total_fuel_cost': 'Fuel Cost',
    'fix_var_om_mwh': 'Total O&M MWh',
    # 'variable_om_mwh': 'Variable O&M',
    # 'fixed_om_mwh': 'Fixed O&M',
    # 'variable_om_mwh_ferc1': 'Variable O&M (FERC1)',
    # 'fixed_om_mwh_ferc1': 'Fixed O&M (FERC1)',
    # 'variable_om_mw_nems': 'Variable O&M (NEMS)',
    # 'fixed_om_mwh_nems': 'Fixed O&M (NEMS)',
    # 'variable_om_is_NEMS': 'Variable O&M used NEMS?',
    # 'fixed_om_is_NEMS': 'Fixed O&M used NEMS?',
    'mcoe': 'Marginal Cost of Energy',
    'co2_mass_tons': 'Annual CO2 Emissions tons',
    'nox_mass_tons': 'Annual NOx Emissions tons',
    'so2_mass_tons': 'Annual SO2 Emissions tons',
    'pm_mass_tons': 'Annual PM2.5 Emissions tons',
    'total_damages': 'Annual Public Health Damages',
    'premature_deaths': 'Annual Premature Dealths',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'state': 'State',
    'county': 'County',
    'city': 'City'}

# Field: [Source, Description] for optional inclusion in multi-index
data_source_dict = {
    'EIA Plant Code': ['EIA860', 'table 2 or 3.1'],
    'PUDL Plant Code': ['PUDL', ''],
    'Plant Name': ['EIA860', 'table 2'],
    'PUDL Unit ID': ['PUDL', 'maps onto EIA plant code'],
    'Report Year': ['EIA860', 'table X; adapted from report_date for FERC Form 1 compatability'],
    'Latitude': ['EIA860', 'table 2'],
    'Longitude': ['EIA860', 'table 2'],
    'State': ['EIA860', 'table 2'],
    'County': ['EIA860, table 2'],
    'City': ['EIA860', 'table 2'],
    'EIA Generator ID': ['EIA860', 'table 3.1'],
    'MW Nameplate Capacity': ['EIA860', 'table 3.1; aggregated by level'],
    'Age': ['EIA860', 'table 3.1; report_date - operating_date; aggregated by weighted average'],
    'Retirement Date': ['EIA860', 'table 3.1; based on most recent retirement date if all aggregated units have a listed retirement date'],
    'Technology': ['EIA860', 'table 3.1; Energy Source 1; aggregated by unique value'],
    'Broad Fuel Classification': ['EIA860', 'reference table 28'],
    'Annual Total Fuel Consumption MMBtu': ['EIA923', 'aggregated by sum'],
    'Annual Electricity Net Generation MWh': ['EIA923', 'aggregated by sum'],
    'Heat Rate MMBtu/MWh': ['EIA923', 'aggregated by weighted average'],
    'Significant Heat Rate Discrepancy?': ['', 'True if more than one standard deviation away from the mean of a given fuel type'],
    'Difference Between Maximum and Minimum Unit Heat Rates': ['', 'calculated by subtracting the minimum unit heat rate from the minimum unit heat rate'],
    'Fuel Cost': ['EIA923', ''],
    'Total O&M MWh': ['FERC Form 1', 'FERC costs disaggregated based on EIA fuel percent of net generation. Comes from FERC value opex_nofuel'],
    # 'Variable O&M': ['FERC Form 1 & NEMS', 'FERC costs disaggregated based on EIA fuel percent of net generation; NEMS added for 2018 values (though likely more recent)'],
    # 'Fixed O&M': ['FERC Form 1 & NEMS', 'FERC costs disaggregated based on EIA fuel percent of net generation; NEMS added for 2018 values (though likely more recent)'],
    # 'Variable O&M used NEMS?': ['NEMS', 'pltf860.txt column no.63; aggregated by sum; True indicates use of NEMS in Variable O&M column'],
    # 'Fixed O&M used NEMS?': ['NEMS', 'pltf860.txt column no.64; aggregated by sum; True indicates use of NEMS in Fixed O&M column'],
    # 'Variable O&M (FERC1)': ['FERC Form 1', 'table X; disaggregated based on EIA fuel percent of net generation'],
    # 'Fixed O&M (FERC1)': ['FERC Form 1', 'table X; disaggregated based on EIA fuel percent of capacity'],
    # 'Variable O&M (NEMS)': ['NEMS', 'pltf860.txt column no.63; aggregated by sum'],
    # 'Fixed O&M (NEMS)': ['NEMS', 'pltf860.txt column no.64; aggregated by sum'],
    'Marginal Cost of Energy': ['', 'Calculated with EIA923 fuel cost, net generation, and FERC Form 1 O&M costs'],
    'Annual CO2 Emissions tons': ['CEMS', 'table X; reported in tons/kwh'],
    'Annual NOx Emissions tons': ['CEMS', 'table X; reported in lbs/kwh'],
    'Annual SO2 Emissions tons': ['CEMS', 'table X; reported in tons/kwh'],
    'Annual PM2.5 Emissions tons': ['GREET', 'Electricity Generation table 2.1; paired PM2.5 g/kWh values with EIA plants/units based on NERC region and generation technology type; converted to tons annually'],
    'Annual Public Health Damages': ['EPA benMAP', 'calculated damages by summing national values at the 2020 threshold for pm, sox, and nox in 06$/ton; see http://www2.epa.gov/benmap/response-surface-model-rsm-based-benefit-ton-estimates'],
    'Annual Premature Dealths': ['', 'divide annual damages by EPA statistical value of a life ($7.4 million); see https://www.epa.gov/environmental-economics/mortality-risk-valuation#whatvalue'],
    'Number of EIA units under this PUDL code': ['', ''],
    'Number of FERC Form 1 entires under this PUDL code': ['', '']
}

# ----------------------------------------------------------
# -------------------- Fluid Functions ---------------------
# ----------------------------------------------------------


def date_to_year(mcoe_out_df):
    """Convert report_date to report_year for EIA FERC integration.

    This function synchonizes FERC Form 1 and EIA data. FERC
    Form 1 Data only has a column for report year while EIA has the full date;
    having a date column in common is essential when merging the data from both
    sources, therefore we select the less specific version, year, and
    standardize it for both EIA and FERC.

    Args:
        mcoe_out_df (pandas.DataFrame): A DataFrame containing raw EIA data.
        The output of running the pudl_out.mcoe() function.
    Returns:
        pandas.DataFrame: A DataFrame with a column for the report year
            rather than the full date.
    """
    logger.info('Converting date to year')
    year_df = (
        mcoe_out_df.assign(report_year=lambda x: x.report_date.dt.year)
        .drop('report_date', axis=1))
    return year_df


def add_generator_age(with_year_raw_eia_df):
    """Calculate and add a column for generator age.

    This function calculates generator age by subtracting the year the
    generator went into operation from the report year. (EIA data must be run
    through date_to_year() first).

    This is calculated here as opposed to in the eia860.py file because of how
    the 'operating_date' is merged with the rest of the data. It seemed cleaner
    to calculate the generator age here.

    Args:
        with_year_rwa_eia_df (pandas.DataFrame): A DataFrame with raw EIA data
            run through the date_to_year() function.
    Returns:
        pandas.DataFrame: A DataFrame with a new column for generator age.
    """
    logger.info('Calculating generator age')
    gen_df = with_year_raw_eia_df.astype({'operating_date': 'datetime64[ns]'})
    gen_df = gen_df.assign(generator_age_years=(gen_df.report_year -
                                                gen_df.operating_date.dt.year))
    return gen_df


def eliminate_retired_plants(raw_eia_all_plants_df):
    """Eliminate plants with retirement dates.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
    Returns:
        pandas.DataFrame: A DataFrame containing raw EIA data without plants
            that have been retired.

    """
    logger.info('Eliminating retired plants')
    # Read recent EIA file containing retirement info
    retired_df = (
        pd.read_excel('february_generator2020.xlsx', 'Retired', header=1,
                      dtype={'Generator ID': 'str',
                             'Plant ID': 'Int64',
                             'Retirement Year': 'Int64'})
        .rename(columns={'Plant ID': 'plant_id_eia',
                         'Generator ID': 'generator_id',
                         'Retirement Year': 'retirement_year'}))
    # Keep relevant columns
    retired_df = retired_df[['plant_id_eia', 'generator_id',
                             'retirement_year']]
    # Merge raw EIA data with retired generators data and delete those with
    # retirement dates
    raw_with_retired_df = (
        pd.merge(raw_eia_all_plants_df, retired_df,
                 on=['plant_id_eia', 'generator_id'], how='left'))
    raw_no_retired_df = (
        raw_with_retired_df.loc[raw_with_retired_df['retirement_year'].isna()]
        .drop('retirement_year', axis=1))
    return raw_no_retired_df


def prep_raw_eia(pudl_out):
    """Add generator age and report year column to raw eia data.

    This acts as the singular function needed to prep the raw EIA data. Any
    functions needing this "cleaned" or "prepped" raw data will refer to it as
    raw_eia_df.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the tables for EIA and FERC Form 1 analysis.
    Returns:
        pandas.DataFrame: A DataFrame containing raw EIA data with a column for
            generator age and a column for report year.
    """
    logger.info('Prepping raw EIA data')
    raw_eia = pudl_out.mcoe()
    year_df = date_to_year(raw_eia)
    year_age_df = add_generator_age(year_df)
    year_age_no_retired_plants_df = eliminate_retired_plants(year_age_df)
    return year_age_no_retired_plants_df


def test_segment(df):
    """Grab a DataFrame chunk pertaining to a single plant id.

    Args:
        df (pandas.DataFrame): Any DataFrame with the column names 'plant_id_
            pudl' and 'report_year'. Can also be multindex.
    Returns:
        pandas.DataFrame: A part of the input DataFrame where plant_id_pudl is
            32.
    """
    if type(df.columns) == pd.core.indexes.base.Index:
        if 'plant_id_pudl' in df.columns.tolist():
            df = (
                df.loc[df['plant_id_pudl'] == 32]
                .sort_values('report_year', ascending=False))
        else:
            df = (
                df.loc[df['PUDL Plant Code'] == 32]
                .sort_values('Report Year', ascending=False))
    elif type(df.columns) == pd.core.indexes.multi.MultiIndex:
        df = (
            df.loc[df[('PUDL', '', 'PUDL Plant Code')] == 32]
            .sort_values(('EIA860', 'table X; adapted from report_date for \
                           FERC Form 1 compatability', 'Report Year'),
                         ascending=False))
    return df


def year_selector(df, years):
    """Define the range of dates represented in final dataframe."""
    logger.info('Selecting specific years')
    df_years = df.loc[df['Report Year'].isin(years)]
    return df_years


def weighted_average(df, wa_col_dict, idx_cols):
    """Generate a weighted average for multiple columns at once.

    When aggregating the data by plant and fuel type, many of the values can
    be summed. Heat rates and generator age, however, are claculated with a
    weighted average. This function exists because there is no python or numpy
    function to calculate weighted average like there is for .sum() or .mean().

    In this case, the heat rate calculation is based the 'weight' or share
    of generator net generation (net_generation_mwh) and the generator age is
    based on that of the generators' capacity (capacity_mw). As seen in the
    global eia_wa_col_dict dictionary.

    Args:
        df (pandas.DataFrame): A DataFrame containing, at minimum, the columns
            specified in the other parameters wa_col_dict and by_cols.
        wa_col_dict (dict): A dictionary containing keys and values that
            represent the column names for the 'data' and 'weight' values.
        idx_cols (list): A list of the columns to group by when calcuating
            the weighted average value.
    Returns:
        pandas.DataFrame: A DataFrame containing weigted average values for
            specified 'data' columns based on specified 'weight' columns.
            Grouped by an indicated set of columns.
    """
    merge_df = df[idx_cols]
    for data, weight in wa_col_dict.items():
        logger.info(' - Calculating weighted average for ' + data)
        df['_data_times_weight'] = df[data] * df[weight]
        df['_weight_where_notnull'] = df[weight] * pd.notnull(df[data])
        g = df.groupby(idx_cols)
        result = g[
            '_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        result = result.to_frame(name='weighted_ave_'+data).reset_index()
        merge_df = pd.merge(merge_df, result, on=idx_cols, how='outer')
    return merge_df


def calc_inflation(index, year, value):
    """Calculate new cost with inflation depending on index specified.

    This function calculates inflation using two different indexes. First,
    the one used in the NEMS model to calculate nominal fixed and variable
    costs of NEMS data (reported in 87$ - equivalent to 1). Second, the BLS
    CPI-U annual percent change rate for use in calculating the public health
    damages (reported in 06$). The calculation for public health damages is a
    bit more involved because it is not in comparison to a base value, rather
    each percent change value is in relation to the prior year.

    Args:
        index (str): The desired index to use for the inflation calculation
            (either nems or bls). NEMS index coded to only read 2018 values
        year (int): The year for which you'd like to calculate inflation
        value (pd.Series): The original values you'd like to calculate
            inflation for.
    Returns:
        pd.Series: The new, inflation adjusted values for a given year.
    """
    # For use with NEMS fixed and variable cost data. Reported in '87' dollars.
    if index == 'nems':
        nems_idx = pd.read_excel('NEMS_GDP_infl.xlsx', header=3,
                                 names=['Year', 'Rate'])
        nems_2018 = float(nems_idx.loc[nems_idx['Year'] == 2018].Rate)
        new_cost = value * nems_2018
    # For use with public health damages calculation
    elif index == 'bls':
        bls_idx = pd.read_excel('cpipress5.xlsx', header=5)
        bls_idx = bls_idx.iloc[:, 1:6]
        bls_idx.columns = ['Date', 'C-CPI-U_month', 'CPI-U_month',
                           'C-CPI-U_year', 'CPI-U_year']
        bls_idx['CPI-U_year'] = bls_idx['CPI-U_year']/100
        # run inflation calculations for a given year. Reported in '06 dollars.
        # Done by aggregating the yearly percent change values like so:
        # pct_cng = (pct1 + pct2) + (pct1 * pct2) -- INCREASE
        # pct_cng = (pct1 - pct2) + (pct1 * pct2) -- DECREASE
        curr_yr_idx = (
            bls_idx.loc[bls_idx['Date'] == f'December {year}'].index.values[0])
        if year > 2006:
            # Made a dataframe with only the relevant years
            relevant_yrs_df = bls_idx[7:(curr_yr_idx+1)]
        # You cannot simply work backwards from a percent increase, you must
        # apply the forumula x/(1+x/100) -- I add the *-1 because it becomes
        # a decrease.
        if year < 2006:
            relevant_yrs_df = bls_idx[curr_yr_idx:7]
            relevant_yrs_df.loc[:, ('CPI-U_year')] = (
                relevant_yrs_df['CPI-U_year'].apply(lambda x: x/(1+x/100)*-1))
        pc_list = relevant_yrs_df['CPI-U_year'].to_list()
        pct_cng = np.sum(pc_list) + np.prod(pc_list)
        new_cost = value * pct_cng + value
    else:
        print('Not a valid index. Select between nems and bls')
    return new_cost


def add_source_cols(df):
    """Create column multindex to show data sources.

    Args:
        df (pandas.DataFrame): A finished dataframe in need of adding a
            multiindex for sources.
    Returns:
        pandas.DataFrame: A DataFrame with a multindex that references the
            source of each of the columns. Only column headers that appear in
            the name_clean_dict AND data_source_dict will be included in the
            output.
    """
    logger.info('Adding source cols')
    source_dict = {}
    # Make a new dictionary with only sources of columns relevant to this table
    for key, value in data_source_dict.items():
        if key in df.columns.to_list():
            source_dict.update({key: value})
    # Swap columns and rows to map on sources
    df = df.T.reset_index()
    df['Source'] = df['index'].map(lambda x: data_source_dict[x][0])
    df['Description'] = df['index'].map(lambda x: data_source_dict[x][1])
    # Switch columns back to rows
    df = df.set_index(['Source', 'Description', 'index']).T
    return df


def output_aesthetic(df, add_sources=False):
    """Change column names; add sources as multindex if specified."""
    logger.info('Making it look pretty...')
    # Make a list of columns that are in the level df (with or without mcoe)
    new_cols = [x for x in list(name_clean_dict.keys())
                if x in df.columns.tolist()]
    # Rename columns
    df_clean = (
        df[new_cols].rename(columns=name_clean_dict))
    # Potential to add sources as multindex
    if add_sources is True:
        df_clean = add_source_cols(df_clean)
    return df_clean


# ----------------------------------------------------------
# --------------------- * P A R T  1 * ---------------------
# ----------------------------------------------------------

"""
Part 1 Functions are mostly neutral and applicable accross eia and ferc
datasets therefore they're primarily located in the 'fluid functions' section.
The build_part1_output() function pulls all of the information together and is
the only function that needs to pull together the full output.
"""


def check_agg_for_diff(raw_eia_df, qual_cols, level):
    """Check EIA qualitative data agg to see if there are outliers.

    Sometimes the pudl id does not line up with the exact EIA reporting and
    there are plants with the same pudl id and different latitudes, longitudes,
    plant names, etc. This is usually due to a difference in utility ownership.
    This function checks to see when/where that happens and makes note of it.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        qual_cols (list): A list of the columns you'd like to check to see
            whether or not they are all the same for a given pudl id.
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    """
    logger.info(' - Checking EIA aggregation for plants with different \
                qualitative data')
    for col in qual_cols:
        is_unique_df = (
            raw_eia_df.groupby(input_dict[level+'_index_cols'], as_index=False)
            .agg({col: lambda x: len(x.unique()) > 1}))
        pudl_id_deviant_list = (
            # Error note: this does not work if '==' replaced with 'is'
            is_unique_df.loc[is_unique_df[col] == True]['plant_id_pudl']
            .drop_duplicates()
            .tolist())
        if pudl_id_deviant_list:
            print(f'{pudl_id_deviant_list} have internal differences in {col}')


def calc_agg_retire_date(retire_col, level):
    """Calculate a retirement age for the aggregated data.

    This function is used within the groupby.agg() method to calcualte an
    approporate retirement date for the given aggregation. It will only
    show a retirement date if all of the entities within the aggregation unit
    also have retirement dates (there is no None) in the list, seeing as an
    entity cannot be considered "retired" until all of its components are. As
    a result, when all entities do have a retirement date, the recorded date is
    that which is most recent.

    Args:
        retire_col(pd.Series): The retirement_date column as a series.
    Returns:
        datetime.Datetime: The date of the most recent retirement (or None) if
            one or more entities within the aggregation unit are have a
            retirement date of None.
    """
    # Create appropirate sub-unit name
    if level == 'plant-fuel':
        sub_lev = 'unit/s'
    elif level == 'unit-fuel':
        sub_lev = 'generator/s'
    # Make list out of retirement dates in groupby
    retire_list = retire_col.tolist()
    # Record a retirement data only if all entities in group have a retirement
    # date. Then, select the most recent date.
    if None not in retire_list:
        date = max(retire_list)
    else:
        none_val = 0
        for x in retire_list:
            if x is None:
                none_val += 1
        date = f'{len(retire_list)-none_val} of {len(retire_list)} {sub_lev} \
               have a retirement date'
    return date


def build_part1_output(raw_eia_df, level):
    """Create final data output for Part 1.

    This function compiles information at the plant-fuel or unit-fuel level
    and either sums or finds the weighted average of the aggregated columns.
    Sum applies to 'total_fuel_cost', 'net_generation_mwh', 'capacity_mw', and
    'total_mmbtu'. Weighted average applies to generator age and heat rate.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pandas.DataFrame: A DataFrame that either reflects the plant level
            or unit level EIA data separated by fuel type.
    """
    logger.info('Building Part 1 output')
    # Data validation and heads up
    check_agg_for_diff(raw_eia_df, input_dict['merge_cols_qual'], level)
    # Regroup raw EIA data to desired level, calculating sum and weighted
    # average of the necessary columns.
    agg_df = (
        raw_eia_df.assign(count='place_holder')
        .groupby(input_dict[level+'_index_cols'], as_index=False)
        # Must use .join because x.unique() arrays are not hashable
        .agg({'generator_id': lambda x: '; '.join(x.unique()),
              'energy_source_code_1': lambda x: '; '.join(x.unique()),
              'count': lambda x: x.count(),
              'total_fuel_cost': lambda x: x.sum(),
              'net_generation_mwh': lambda x: x.sum(),
              'capacity_mw': lambda x: x.sum(),
              'total_mmbtu': lambda x: x.sum(),
              'plant_name_eia': lambda x: x.iloc[0],
              'retirement_date': lambda x: calc_agg_retire_date(x, level),
              'state': lambda x: x.iloc[0],
              'city': lambda x: x.iloc[0],
              'county': lambda x: x.iloc[0],
              'latitude': lambda x: x.iloc[0],
              'longitude': lambda x: x.iloc[0]}))
    wa_df = (
        weighted_average(raw_eia_df, eia_wa_col_dict,
                         input_dict[level+'_index_cols']))
    agg_wa_df = (
        pd.merge(agg_df, wa_df, on=input_dict[level+'_index_cols'],
                 how='left')
        .drop_duplicates()
        .reset_index())
    logger.info('Finished compiling Part 1 '+level+' level data')
    return agg_wa_df


def clean_part1_output(part1_df, drop_calcs=False):
    """Clean output for Part 1: round, drop calcs if desired.

    Args:
        part1_df (pandas.DataFrame): The result of running build_part1_output()
        drop_calcs: A boolean to indicate whether you'd like all of the
            calculations or just the index data.
    Returns:
        pandas.DataFrame: A DataFrame that either reflects the plant level
            or unit level EIA data separated by fuel type.
    """
    logger.info('Cleaning Part 1 Output')
    out_df = part1_df.round(2)
    # Just keep the generic plant data if you want
    if 'unit_id_pudl' in out_df.columns.tolist():
        level = 'unit-fuel'
    else:
        level = 'plant-fuel'
    if drop_calcs is True:
        out_df = out_df[input_dict[level+'_index_cols'] +
                        input_dict['merge_cols_qual']]
    logger.info('Finished cleaning Part 1 data')
    return out_df


# ----------------------------------------------------------
# --------------------- * P A R T  2 * ---------------------
# ----------------------------------------------------------

"""
Part 2 Functions are primarily used to make EIA923 and FERC Form 1 data
compatible with one another at the plant-fuel level.

We use EIA936 data broken down by plant and fuel type to inform the FERC Form 1
data disaggregation in the same manner. In other words, we calculate the
percent that each fuel type contributes to a given plant-level statistic (in
this case capacity, net generation, or cost) for the EIA data and use those
fuel percentages to disaggregate FERC Form 1 fixed and operating cost data by
plant and fuel type.

We use the combined information from EIA923 and FERC Form 1 to calculate an
mcoe value for each fuel type within each plant for any given report year.
"""


def fuel_type_to_col_eia(eia_plant_fuel_df, col):
    """Make fuel type row values into columns.

    Reorient dataframe by plant so that fuel type percentages are columns
    rather than row values. Used for merging with FERC Form 1 data. The
    output DataFrame is ready to merge with another DataFrame from this output
    with a different 'col' specified. It is also ready to merge with FERC Form
    1 plant level data to perform fuel level disagregation.

    Args:
        eia_plant_fuel_df (pandas.DataFrame): A DataFrame with eia fuel type
            percent values as rows.
        col (str): The name of the percent value column (such as
            'capacity_mw' or 'net_generation_mwh') that will be reoriented
            to suit the calculation of fuel type breakdown of FERC Form 1
            data.
    Returns:
        pandas.DataFrame: A DataFrame with the fuel type percent values
            as columns rather than rows.
    """
    logger.info('Turning eia fuel percent values for ' + col + ' into columns')
    pcts_as_cols_df = (
        eia_plant_fuel_df.pivot_table(
            'pct_'+col, input_dict['plant_index_cols'], 'fuel_type_code_pudl')
        .reset_index()
        .rename(columns={'coal': 'pct_'+col+'_coal',
                         'gas': 'pct_'+col+'_gas',
                         'oil': 'pct_'+col+'_oil',
                         'waste': 'pct_'+col+'_waste'}))
    return pcts_as_cols_df


def calc_fuel_pcts_eia(eia_plant_fuel_df, pct_col1, pct_col2):
    """Calculate the fuel type breakdown for input columns.

    For use specifically in the eia_fuel_pcts() function where a DataFrame
    containing eia data aggragated by fuel type and plant totals is created.

    Args:
        eia_plant_fuel_df (pandas.DataFrame): A DataFrame containing eia data
            with values aggregated by plant fuel type AND plant total.
        pct_col1 (str): A single column name to be broken down by fuel
            type.
        pct_col2 (str): A single column name to be broken down by fuel
            type.
    Returns:
        pandas.DataFrame: A DataFrame containing the percent breakdown by
            fuel type of pct_col1. Merged with that of pct_col2.
    """
    logger.info(' -- Calculating eia fuel type percentages')
    # Calculate percent that each fuel contributes to input cols
    # (capcity and net gen in this case)
    eia_plant_fuel_df['pct_'+pct_col1] = (
        eia_plant_fuel_df[pct_col1+'_fuel_level'] /
        eia_plant_fuel_df[pct_col1+'_plant_level'])
    eia_plant_fuel_df['pct_'+pct_col2] = (
        eia_plant_fuel_df[pct_col2+'_fuel_level'] /
        eia_plant_fuel_df[pct_col2+'_plant_level'])
    # Reorient table so that fuel type percents become columns
    # (makes it easier to run calculations on FERC1 data)
    pct_df1 = fuel_type_to_col_eia(eia_plant_fuel_df, pct_col1)
    pct_df2 = fuel_type_to_col_eia(eia_plant_fuel_df, pct_col2)
    # Merge percent dfs so that they are both included.
    # pd.merge will not take a df list.
    eia_pct_df = (
        pd.merge(pct_df1, pct_df2, on=input_dict['plant_index_cols'],
                 how='outer'))
    return eia_pct_df


def prep_plant_fuel_data_eia(raw_eia_df):
    """Group EIA data by specified level to ready for FERC Form 1 merge.

    This function aggregates EIA data at the plant-fuel level to ready it for
    integration with FERC Form 1 cost data once it's been disaggregated with
    EIA fuel breakdown percents.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through date_to_year()
            and add_generator_age() functions.
    Returns:
        pandas.DataFrame: A DataFrame with EIA data disaggregated
            by plant and fuel ready to merge with FERC Form 1 data once it's
            been similarly disaggregated with EIA percent data.
    """
    logger.info(' - Building eia table broken down by plant and fuel type')
    eia_plant_fuel_df = (
        raw_eia_df.assign(eia_unit_count='place holder')
        .groupby(input_dict['plant-fuel_index_cols'], as_index=False)
        .agg({'total_fuel_cost': lambda x: x.sum(),
              'net_generation_mwh': lambda x: x.sum(),
              'capacity_mw': lambda x: x.sum(),
              'total_mmbtu': lambda x: x.sum(),
              'eia_unit_count': lambda x: x.count()}))
    return eia_plant_fuel_df


def compile_fuel_pcts_eia(raw_eia_df):
    """Extract EIA's fuel type percents to map onto FERC Form 1 data.

    This function pulls together the outputs of functions calc_fuel_pcts_eia()
    (which itself calls eia_pct_df_maker()) to create an output DataFrame with
    EIA data aggregated by level with columns indicating the percent that each
    fuel type contribues to the total capacity and net generation.

    This data is now staged for a merge with FERC Form 1 data so that its cost
    values can be multiplied by the EIA fuel percentages and so disaggreated by
    the specified level.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
    Returns:
        pandas.DataFrame: A DataFrame with columns for capacity
            and net generation for each fuel type at the plant or unit level.
    """
    logger.info(' - Readying EIA fuel pct data to merge with FERC')
    # Create df with the plant level totals (combines fuel types) for the
    # aggregated mcoe data.
    eia_plant_df = (
        raw_eia_df.groupby(input_dict['plant_index_cols'], as_index=False)
        [input_dict['eia_sum_cols']]
        .sum())
    # Create df with the plant-fuel level totals
    eia_plant_fuel_df = prep_plant_fuel_data_eia(raw_eia_df)
    # Merge with fuel type totals for df with plant totals and fuel totals
    merge_df = (
        pd.merge(eia_plant_df, eia_plant_fuel_df,
                 on=input_dict['plant_index_cols'], how='outer',
                 suffixes=['_plant_level', '_fuel_level'])
        .drop_duplicates())
    # Calculate the percentage that each fuel type (coal, oil, gas, waste)
    # accounts for for the specified columns (net gen & capacity)
    # **NOTE** cannot feed this function a list of col names beacuse merge
    # function does not take a list.
    eia_pct_df = (
        calc_fuel_pcts_eia(
            merge_df, 'net_generation_mwh', 'capacity_mw'))
    # Return table needed for ferc fuel type delineation and final FERC1 merge.
    return eia_pct_df


def prep_plant_data_ferc1(raw_ferc1_df):
    """Ready FERC Form 1 data for merging with EIA-932 fuel pct breakdown.

    The output DataFrame for this function will be ready to merge with the EIA
    percent breakdowns by plant and fuel type.

    Args:
        raw_ferc1_df (pandas.DataFrame): A DataFrame with raw FERC Form 1 Data.
    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 data aggregated by
            plant.
    """
    logger.info(' - Building FERC table broken down by plant')
    ferc1_plant_df = (
        raw_ferc1_df.assign(ferc1_unit_count='place_holder')
        .groupby(input_dict['plant_index_cols'], as_index=False)
        .agg({'capex_total': lambda x: x.sum(),
              'opex_fuel': lambda x: x.sum(),
              'opex_production_total': lambda x: x.sum(),
              'ferc1_unit_count': lambda x: x.count()})
        .assign(opex_nofuel_ferc1=lambda x: (x.opex_production_total -
                                             x.opex_fuel))
        .rename(
            columns={'capex_total': 'capex_total_ferc1',
                     'opex_fuel': 'opex_fuel_ferc1',
                     'opex_production_total': 'opex_production_total_ferc1'}))
    return ferc1_plant_df


def calc_cap_op_by_fuel_ferc1(ferc1_pct_prep_df):
    """Calculate FERC Form 1 cost breakdowns from EIA-923 fuel percentages.

    A helper function for specific use within the merge_ferc_with_eia_pcts()
    function. Once the the outputs of eia_fuel_pcts() and prep_plant_data_
    ferc1() are merged, this function multiplies the EIA fuel pct
    columns by the FERC From 1 values, so calculating the equivalent FERC cost
    breakdown by fuel. Capex is based on capacity and opex is based on net
    generation.

    Args:
        ferc1_pct_df (pandas.DataFrame): A DataFrame with raw ferc plant
            values merged with eia percents.
    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 Data disaggregated
            by plant and fuel type based on EIA percent values.

    """
    logger.info(' -- Building FERC table broken down by plant and fuel type')
    # Did not use .assign here because need to integrate local variables.
    for fuel in fuel_types:
        ferc1_pct_prep_df['capex_'+fuel] = (
            ferc1_pct_prep_df['capex_total_ferc1'] *
            ferc1_pct_prep_df['pct_capacity_mw_'+fuel])
        ferc1_pct_prep_df['opex_nofuel_'+fuel] = (
            ferc1_pct_prep_df['opex_nofuel_ferc1'] *
            ferc1_pct_prep_df['pct_net_generation_mwh_'+fuel])
    ferc1_plant_fuel_df = ferc1_pct_prep_df
    return ferc1_plant_fuel_df


def cost_subtable_maker(ferc1_pct_df, cost_type):
    """Rearange cost breakdown data.

    A helper function for specific use within the merge_ferc_with_eia_pcts()
    function. It that takes the result of running calc_cap_op_by_fuel_ferc1()
    on newly merged eia_fuel_pcts() output and prep_plant_data_ferc1() (which
    outputs a table with ferc cost data broken down by fuel type) and melts it
    so that the fuel type columns become row values again rather than columns.

    This function must be executed once for every FERC value that has been
    disaggregated. (i.e. once for capex and once for opex). The two melted
    tables are later merged in the meta function merge_ferc_with_eia_pcts().

    Args:
        ferc_pct_df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            disaggregated by plant and fuel type.
        cost_type (str): A string with the name of the cost column to subdivide
            by.
    Returns:
        pandas.DataFrame: A DataFrame with disaggregated FERC Form 1 data
            melted so that columns become row values.
    """
    logger.info(' -- Melting FERC pct data back to row values')
    # apply EIA fuel percents to specified FERC cost data.
    cost_df = ferc1_pct_df.rename(columns={'count_ferc1': 'ferc1_unit_count'})
    cost_df = (
        cost_df[(input_dict['plant_index_cols'] + ['ferc1_unit_count'] +
                [cost_type+'_coal', cost_type+'_gas', cost_type+'_oil',
                cost_type+'_waste'])]
        .rename(
            columns={cost_type+'_coal': 'coal',
                     cost_type+'_gas': 'gas',
                     cost_type+'_oil': 'oil',
                     cost_type+'_waste': 'waste'}))
    col_to_row_df = (
        pd.melt(cost_df, input_dict['plant_index_cols']+['ferc1_unit_count'])
        .rename(columns={'value': cost_type,
                         'variable': 'fuel_type_code_pudl'})
        .dropna(subset=[cost_type]))
    return col_to_row_df


def disaggregate_ferc1(eia_pct_df, ferc1_plant_df):
    """Disaggregate FERC Form 1 Data based on EIA fuel breakdown percents.

    This is a meta function that brings together the outputs of several other
    functions to create a cohesive output: FERC Form 1 data broken down
    by plant and fuel type. It's ready to merge with EIA data from prep_eia_
    plant_fuel_data() and then for calculation of mcoe values.

    Args:
        eia_pct_df (pandas.DataFrame): A DataFrame with EIA aggregated by
            plant and fuel type.
        ferc1_plant_df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            aggregated by plant.
    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 data disaggregated
            by the fuel percentage breakdowns depicted in the EIA data.
    """
    logger.info(' - Merging FERC Form 1 data with EIA percent data')
    # Merge prepped EIA923 percent data with FERC1 cost data
    ferc1_pct_prep_df = (
        pd.merge(eia_pct_df, ferc1_plant_df, on=input_dict['plant_index_cols'],
                 how='outer'))
    ferc1_pct_df = calc_cap_op_by_fuel_ferc1(ferc1_pct_prep_df)
    # capex_melt_df = cost_subtable_maker(ferc1_pct_df, 'capex')
    opex_melt_df = cost_subtable_maker(ferc1_pct_df, 'opex_nofuel')
    # Merge disaggregated capex and opex FERC1 tables
    # ferc_cap_op_df = (
    #     pd.merge(capex_melt_df, opex_melt_df,
    #              on=input_dict['plant-fuel_index_cols'] + ['ferc1_unit_count'],
    #              how='outer'))
    return opex_melt_df


def add_nems(mcoe_df):
    """Incorporate NEMS aeo2020 data to account for missing FERC O&M costs.

    Args:
        mcoe_df (pandas.DataFrame): A DataFrame containing mcoe factors from
            FERC Form 1 and EIA. Pre-mcoe calculation.
    Returns:
        pandas.DataFrame: A DataFrame with NEMS values added to account for
            missing FERC Form 1 O&M costs.
    """
    logger.info(' - Adding NEMS cost data')
    path = os.getcwd()
    nems_df = pd.read_csv(path+'/pltf860.csv',
                          dtype={'plant_id': 'int32',
                                 'fixed_om_mwh_87': 'float32',
                                 'variable_om_mwh_87': 'float32'})
    # Grab NEMS costs and group at plant level. Add year column as 2018.
    nems_cost_df = (
        nems_df.groupby('plant_id')[['fixed_om_mwh_87', 'variable_om_mwh_87']]
        .sum()
        .reset_index()
        .assign(report_year=2018,
                plant_id_pudl=lambda x: x.plant_id)
        .drop('plant_id', axis=1))
    # Merge with mcoe df and adjust cost for inflation.
    # NEMS fixed (capex) cost in $/kw and variable (opex) cost in $/mwh
    # Adjust to mw and mwh for mcoe calculation.
    nems_mcoe_merge_df = (
        pd.merge(mcoe_df, nems_cost_df,
                 on=['plant_id_pudl', 'report_year'], how='left')
        .assign(fixed_om_mwh_nems=lambda x: (
                    calc_inflation('nems', 2018, x.fixed_om_mwh_87)),
                variable_om_mwh_nems=lambda x: (
                    calc_inflation('nems', 2018, x.variable_om_mwh_87)),
                capex_nems=lambda x: (x.fixed_om_mwh_nems * 1000 *
                                      x.capacity_mw),
                opex_nofuel_nems=lambda x: (x.variable_om_mwh_nems *
                                            x.net_generation_mwh))
        .drop(['fixed_om_mwh_87', 'variable_om_mwh_87'], axis=1))
    return nems_mcoe_merge_df


def merge_ferc1_eia_mcoe_factors(eia_fuel_df, ferc_fuel_df):
    """Produce final EIA and FERC Form 1 merge and calculate MCOE value.

    This function combines the FERC Form 1 data, now disaggregated by plant and
    fuel type, with EIA data aggregated in the same manner. It then runs
    calculations to add columns for mcoe and rearanges/renames columns for
    the desired output.

    Args:
        eia_fuel_df (pandas.DataFrame): A DataFrame with EIA data broken
            down by plant and fuel type.
        ferc_fuel_df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            broken down by plant and fuel type.
    Returns:
        pandas.DataFrame: A DataFrame with EIA and FERC Form 1 data broken
            down by plant and fuel type. MCOE values calculated.
    """
    logger.info(' - Merging FERC and EIA mcoe data on plant and fuel type')
    # Merge FERC1 and EIA923 on plant, fuel, and year using prep_plant_fuel_
    # data_eia() output associated with key 'plant_fuel_ag'
    eia_ferc_merge_df = (
        pd.merge(eia_fuel_df, ferc_fuel_df,
                 on=input_dict['plant-fuel_index_cols'], how='outer')
        .assign(
            fuel_cost_mwh_eia923=lambda x: (x.total_fuel_cost /
                                            x.net_generation_mwh),
            fix_var_om_mwh=lambda x: (x.opex_nofuel / x.net_generation_mwh)))
            # variable_om_mwh_ferc1=lambda x: (x.opex_nofuel /
            #                                 x.net_generation_mwh),
            # fixed_om_mwh_ferc1=lambda x: x.capex / x.net_generation_mwh))
    return eia_ferc_merge_df


def calc_mcoe(df):
    """Calculate mcoe values with NEMS subsitutes.

    Args:
        df (pandas.DataFrame): The DataFrame for which you'd like to calculate
            an mcoe value.
    Returns: pd.Series: A series of calculated mcoe values
    """
    mcoe = float('nan')
    # if np.isnan(df.opex_nofuel):
    #    var_cost = df.opex_nofuel_nems
    if df.net_generation_mwh > 0:
        mcoe = ((df.total_fuel_cost + df.opex_nofuel) / df.net_generation_mwh)
    return mcoe


def compare_heatrate(raw_eia_df, merge_df, sd_mean_cols=False):
    """Compare heatrates within plants to find outliers.

    Outputs a pandas DataFrame containing information about whether unit level
    heat rates differ significantly from plant level heatrates. To determine
    significant deviation, we calculate the mean and standard deviation of
    heatrate values for plants of a certain fuel type during a certain year.
    Units that are more than one standard deviation from the mean value are
    considered significantly different and are marked as True.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
                pudl_out.mcoe() function that has been run through the
                prep_raw_eia() function.
        merge_df (pandas.DataFrame): The DataFrame you'd like to merge the
            significant heat rate column with.
    Returns:
        pandas.DataFrame: A Dataframe with a boolean column to show whether
            the heat rate of a given unit is significantly different from
            the others within its aggregation group.
    """
    logger.info(' - Comparing heat rates internally')
    # Get plant and unit level wahr then merge for comparison.
    unit_level_df = build_part1_output(raw_eia_df, 'unit-fuel')
    # Find and report difference between min and max heat rate values
    unit_groups = (
        unit_level_df.groupby(input_dict['plant-fuel_index_cols']))
    max_min_df = (
        unit_groups['weighted_ave_heat_rate_mmbtu_mwh']
        .agg(max_heat_rate=('max'),
             min_heat_rate=('min'))
        .assign(
            max_min_hr_diff=lambda x: x.max_heat_rate - x.min_heat_rate)
        .round(2)
        .reset_index())
    max_min_df = (max_min_df[input_dict['plant-fuel_index_cols']
                  + ['max_min_hr_diff']])
    # Remove crazy outliers for mean / std calculation
    unit_level_df_low = (
        unit_level_df.loc[
            unit_level_df['weighted_ave_heat_rate_mmbtu_mwh'] < 20])
    # Find average heat rate for fuel type and year and standard dev.
    mean_yr_hr_df = (
        unit_level_df_low
        .groupby(['report_year', 'fuel_type_code_pudl'])
        .agg(
            mean_hr_mwh=('weighted_ave_heat_rate_mmbtu_mwh', 'mean'),
            std_hr=('weighted_ave_heat_rate_mmbtu_mwh', 'std'))
        .reset_index())
    # Merge year and fuel type hr average & std with unit_level df
    unit_level_df_mean = (
        pd.merge(unit_level_df, mean_yr_hr_df,
                 on=['report_year', 'fuel_type_code_pudl'], how='left')
        .assign(
            hr_dist_from_mean=lambda x: (
                abs(x.weighted_ave_heat_rate_mmbtu_mwh - x.mean_hr_mwh)),
            hr_z_score=lambda x: (x.hr_dist_from_mean / x.std_hr),
            sig_hr=lambda x: x.hr_z_score > 1))
    # Account for mean and standard deviation of heat rates for each fuel type.
    # mean_dict = {}
    # std_dict = {}
    # for fuel in fuel_types:
    #     mean = (
    #         unit_level_df_low.query(f"fuel_type_code_pudl=='{fuel}'")
    #         ['weighted_ave_heat_rate_mmbtu_mwh'].mean()).round(2)
    #     std = (
    #         unit_level_df_low.query(f"fuel_type_code_pudl=='{fuel}'")
    #         ['weighted_ave_heat_rate_mmbtu_mwh'].std()).round(2)
    #     mean_dict[fuel] = mean
    #     std_dict[fuel] = std
    # print('excluding heat rates over 34:')
    # print(f'average heat rates for fuel types: {mean_dict}')
    # print(f'standard deviation for fuel types: {std_dict}')
    # # Create columns for distance from mean and bool for whether over one std
    # unit_level_df['hr_dist_from_mean'] = (
    #     unit_level_df.apply(
    #         lambda x: abs(x.weighted_ave_heat_rate_mmbtu_mwh -
    #                       mean_dict[x.fuel_type_code_pudl]), axis=1))
    # unit_level_df['sig_hr'] = (
    #     unit_level_df.apply(
    #         lambda x: (x.hr_dist_from_mean > std_dict[x.fuel_type_code_pudl])
    #         axis=1))
    # Return columns with mean and SD calculations OR merge to regular output
    if sd_mean_cols is True:
        final_hr_df = unit_level_df_mean
    else:
        # Aggregate to plant level
        plant_level_df = (
            unit_level_df_mean
            .groupby(input_dict['plant-fuel_index_cols'])['sig_hr']
            .any()
            .reset_index())
        # Merge with max min difference calculations
        with_min_max_df = (
            pd.merge(plant_level_df, max_min_df,
                     on=input_dict['plant-fuel_index_cols'], how='left'))
        # Merge with input df (merge_df)
        logger.info('preparing merge: checking df length compatability')
        if len(merge_df) != len(with_min_max_df):
            print('df passed not the same length as plant-level aggreation df; \
                   check for duplicates')
        final_hr_df = (
            pd.merge(merge_df, with_min_max_df,
                     on=input_dict['plant-fuel_index_cols']))
        # If only one eia unit, change heat rate diff to say "only one unit"
        final_hr_df.loc[
            (final_hr_df['eia_unit_count'] == 1)
            & (final_hr_df['max_min_hr_diff'] == 0),
            'max_min_hr_diff'] = 'only one unit'
    return final_hr_df


def build_part2_output(raw_eia_df, raw_ferc1_df):
    """Compile final data output for Part 2.

    A function that compiles the neccesary data for Part 2 and outputs a
    DataFrame with mcoe variables. It also adds a data validation check for
    heat rate, implementing a boolean to indicate whether there are units
    within the plant/fuel aggregation that are outliers. Finally, it adds a
    field for plant_id_eia.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through date_to_year()
            and add_generator_age() functions.
        raw_ferc1_df (pandas.DataFrame): A DataFrame containing FERC Form 1
            data.
    Returns:
        pandas.DataFrame: A DataFrame with the MCOE variables and calculations.
    """
    logger.info('Building Part 2 output')
    # Prep data for EIA-FERC integration
    eia_plant_fuel_df = prep_plant_fuel_data_eia(raw_eia_df)
    ferc1_plant_df = prep_plant_data_ferc1(raw_ferc1_df)
    # Find EIA fuel breakdown percents and apply them to FERC Form 1 data.
    eia_pct_df = compile_fuel_pcts_eia(raw_eia_df)
    ferc1_plant_fuel_df = disaggregate_ferc1(eia_pct_df, ferc1_plant_df)
    # Merge FERC, EIA, and NEMS; calculate mcoe
    eia_ferc1_merge_df = merge_ferc1_eia_mcoe_factors(eia_plant_fuel_df,
                                                      ferc1_plant_fuel_df)
    # Add NEMS data for most recent year O&M cost
    #with_nems_df = add_nems(eia_ferc1_merge_df)
    # Validate NEMS data_eia
    # validate_nems(with_nems_df)

    # Calculate MCOE value
    eia_ferc1_merge_df['mcoe'] = (
        eia_ferc1_merge_df.apply(lambda x: calc_mcoe(x), axis=1))
    # Add significant heat rate identifier (pudl id 125 = Comanche for test)
    with_sig_hr_df = compare_heatrate(raw_eia_df, eia_ferc1_merge_df)
    logger.info('Finished compiling Part 2 data compilation')
    return with_sig_hr_df


def clean_part2_output(part2_df, nems_cols=True):
    """Clean output for Part 2: select columns, choose NEMS column type, round.

    Args:
        part2_df (pandas.DataFrame): The output from running bulid_part2_
            output().
        nems_cols (boolean): An indication of whether to include the NEMS cost
            columns alongside the FERC Form 1 cost columns for fixed and
            variable cost or keep the NEMS columns as boolean flag columns with
            True is NEMS is used and merged into a common fixed or variable
            column.
    Returns:
        pandas.DataFrame: A DataFrame with all of the components of Part 2,
            cleaned and ready for output.
    """
    logger.info('Cleaning Part 2 output')
    # Select output columns
    out_df = (
        part2_df[[
            'plant_id_pudl',
            'fuel_type_code_pudl',
            'report_year',
            'fuel_cost_mwh_eia923',
            'fix_var_om_mwh',
            #'variable_om_mwh_ferc1',
            #'fixed_om_mwh_ferc1',
            #'variable_om_mwh_nems',
            #'fixed_om_mwh_nems',
            'mcoe',
            'eia_unit_count',
            'ferc1_unit_count',
            'sig_hr',
            'max_min_hr_diff']])
    # Combine NEMS and FERC1 cost columns; create bool column to show NEMS use.
    if nems_cols is False:
        out_df = (
            out_df.assign(
                variable_om_mwh=lambda x: (
                    x.variable_om_mwh_ferc1.fillna(x.variable_om_mwh_nems)),
                fixed_om_mwh=lambda x: (
                    x.fixed_om_mwh_ferc1.fillna(x.fixed_om_mwh_nems)),
                variable_om_is_NEMS=lambda x: (
                    x.variable_om_mwh_ferc1.isna() &
                    x.variable_om_mwh_nems.notna()),
                fixed_om_is_NEMS=lambda x: (
                    x.fixed_om_mwh_ferc1.isna() & x.fixed_om_mwh_nems.notna()))
            .drop(['variable_om_mwh_ferc1', 'variable_om_mwh_nems',
                   'fixed_om_mwh_ferc1', 'fixed_om_mwh_nems'], axis=1))
    out_df = out_df.round(2)
    out_df['ferc1_unit_count'] = out_df['ferc1_unit_count'].astype('Int64')
    logger.info('Finished cleaning Part 2 data')
    return out_df

# ----------------------------------------------------------
# --------------------- * P A R T  3 * ---------------------
# ----------------------------------------------------------

"""
Part 3 functions add information about emissions and public health
impacts. The pull information from CEMS (sox, nox, co2), GREET (pm2.5), and
EPA (public health damages, premature deaths).
"""


def get_cems():
    """Retrieve CEMS data for a given year for all states.

    This function requires running pudl and having epacems files stored in
    parquets. Reads the parquet files and provides data from all years and
    states on emissions. Takes a minute or so to run. Renames some fields for
    eventual merge with EIA data.

    Returns:
        pandas.DataFrame: A DataFrame with emissions data for all states in
            a given year.
    """
    logger.info('Getting CEMS data....this may take a sec.')
    client = Client()
    cols = ['plant_id_eia', 'unitid',
            'so2_mass_lbs', 'nox_mass_lbs', 'co2_mass_tons']
    out_df = pd.DataFrame()
    for yr in range(1995, 2019):
        # TOP = jupyter; BOTTOM = atom (here)
        epacems_path = (os.path.dirname(os.getcwd())
                        + f'/PUDL_DIR/parquet/epacems/year={yr}')
        # epacems_path = os.getcwd() + f'/PUDL_DIR/parquet/epacems/year={yr}'
        cems_dd = (
            dd.read_parquet(epacems_path, columns=cols)
            .groupby(['plant_id_eia', 'unitid'])[
                     ['so2_mass_lbs', 'nox_mass_lbs', 'co2_mass_tons']]
            .sum())
        cems_df = (
            client.compute(cems_dd)
            .result()
            .assign(year=yr))
        out_df = pd.concat([out_df, cems_df])
    out_df = (
        out_df.reset_index()
        .rename(columns={'unitid': 'boiler_id', 'year': 'report_year'})
        .astype({'so2_mass_lbs': 'float64',
                 'nox_mass_lbs': 'float64',
                 'co2_mass_tons': 'float64'}))
    return out_df


def add_cems_to_eia(part1_df, bga_df, cems_df, raw_eia_df, level):
    """Merge EIA plant or unit level data with CEMS emissions data.

    Args:
        part1_df (pandas.DataFrame): The output from running either
            built_part1_output(pudl_out, 'plant'), or build_part1_output(
            pudl_out, 'unit').
        bga_df (pandas.DataFrame): The output from running pudl_out.bga()
        cems_df (pandas.DataFrame): The output of running get_cems(). Added as
            a parameter so only have to run get_cems() once because it takes a
            while.
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pandas.DataFrame: A DataFrame containing an merge of the part1 output
            and the CEMS emissions data.
    """
    logger.info(' - Adding cems to EIA data')
    id_cols = ['plant_id_eia', 'unit_id_pudl', 'generator_id',
               'report_year']
    cems_cols = ['so2_mass_lbs', 'nox_mass_lbs', 'co2_mass_tons']
    # Add boiler id to EIA data. Boilder id matches (almost) with CEMS unitid.
    eia_with_boiler_id = (
        pd.merge(raw_eia_df[id_cols+['plant_id_pudl', 'fuel_type_code_pudl']],
                 bga_df[id_cols+['boiler_id']], on=id_cols, how='left'))
    eia_cems_merge = (
        pd.merge(eia_with_boiler_id, cems_df, on=['plant_id_eia', 'boiler_id',
                 'report_year'], how='left')
        .groupby(['plant_id_pudl', 'plant_id_eia', 'unit_id_pudl',
                  'fuel_type_code_pudl', 'report_year'])[cems_cols].sum()
        .reset_index())
    eia_cems_agg = (
        eia_cems_merge.groupby(input_dict[level+'_index_cols'])[cems_cols]
        .sum()
        .assign(so2_mass_tons=lambda x: x.so2_mass_lbs / 2000,
                nox_mass_tons=lambda x: x.nox_mass_lbs / 2000)
        .drop(['so2_mass_lbs', 'nox_mass_lbs'], axis=1)
        .reset_index())
    return eia_cems_agg


def calc_tech_pct(raw_eia_df, level):
    """Show technology percent at a given aggregation level.

    This function takes in the raw EIA data file and calculates the generation
    technology percent make up of a given level of aggregation based on the
    stated capacity of that generator. For example, if fed "plant", the
    function will calculate what percent of the plant (at the fuel level) is
    made up of boilers vs combined cycle vs. the other technology options.

    Args:
        raw_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pandas.DataFrame: A DataFrame showing the percent breakdown of each
            generation technology type per 'level' (plant or unit broken down
            by fuel). Rows for level, columns for technology type.
    """
    # Map GREET names onto tech description and find totals for level of
    # aggregation specified.
    logger.info(' -- Calculating tech percents')
    greet_df = (
        raw_eia_df.assign(greet_tech=lambda x: x.technology_description
                          .map(tech_rename_greet)
                          .fillna('Not_Specified')))
    # Calculate level totals
    total_df = (
        raw_eia_df.groupby(input_dict[level+'_index_cols'])
        [input_dict['eia_pct_cols']]
        .sum()
        .reset_index()
        .rename(columns={'capacity_mw': 'cap_total_mw',
                         'net_generation_mwh': 'net_gen_total_mwh'}))
    # Merge back to raw data to get sidebyside capacity values for pct calc.
    merge_df1 = (
        pd.merge(greet_df, total_df, on=input_dict[level+'_index_cols'],
                 how='left'))
    # Separate because need to put variable into assign statement. Pivots
    # table so that each row is the level specified and there are columns for
    # each technology type percent. Return with net_gen total to tranform
    # GREET numbers from mwh to annual values.
    merge_df = (
        merge_df1.assign(pct=(merge_df1['net_generation_mwh'] /
                              merge_df1['net_gen_total_mwh']).round(2))
        .pivot_table('pct', (input_dict[level+'_index_cols'] +
                     ['net_gen_total_mwh', 'nerc_region']),
                     'greet_tech', np.sum)
        .fillna(0)
        .reset_index())
    return merge_df


def calc_pm_value(tech_df, level):
    """Calculate pm2.5 value for the given level of aggregation.

    This function calculates the amount of pm2.5 associated with each unit or
    plant by fuel type by figuring out what kinds of generator technologies are
    used in that aggregation unit and then multiplying their percent make up (
    based on net generation) by net generation (mwh/year) and the values for
    kwh/mwh (1000/1) and ton/g (0.000001/1) to convert to pm2.5 tons/year.

    Args:
        tech_df (pandas.DataFrame): The output of calc_tech_pct() - a DataFrame
            with the percents for each technology (based on capacity) shown as
            columns with rows for each level of aggregation (unit or plant).
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pandas.DataFrame: A DataFrame with the value for pm2.5 in lb/year
            calculated based on tech type values in g/kwh from the GREET model.
    """
    logger.info(' -- Calculating pm2.5 values')
    # Make technology columns into rows; make column for fuel + tech type.
    pm_df = (
        pd.melt(tech_df, (input_dict[level+'_index_cols'] +
                ['net_gen_total_mwh', 'nerc_region']))
        .assign(pm_id=lambda x: x.fuel_type_code_pudl+'_'+x.greet_tech))
    # Must do this part separatly because does not work in assign...
    # Calculate pm2.5 value based on region and technology
    pm_df['pm_g_kwh'] = (
        pm_df.apply(lambda x: (
            cnct_rgn_tech_to_pm_value(x.nerc_region, x.pm_id)), axis=1))
    # Must do separately becasue its a series...(don't really understand)
    # Go from g/kwh to tons/year
    pm_df = (
        pm_df.assign(
            pm_mass_tons=lambda x: (x.pm_g_kwh * x.net_gen_total_mwh
                                    * 1000 * 0.000001))
        .groupby(input_dict[level+'_index_cols'])['pm_mass_tons']
        .sum()
        .reset_index())
    return pm_df


def cnct_rgn_tech_to_pm_value(nerc, tech):
    """Connect NERC region and tech type to PM2.5 value.

    This function is for specific use in calc_pm_value(). It finds the right
    PM2.5 value based on GREET designation by NERC region and technology type.
    Important to note that these values are still in g/kwh.

    Args:
        nerc (pd.Series): The column 'nerc_region' in the pm_df.
        tech (pd.Series): The column 'pm_id' created in the pm_df.

    Returns:
        float: the PM2.5 value associated with a given NERC region and
            technology type.
    """
    if tech not in pm_tech_dict.keys():
        pm_value = 0
    elif 'Not_Specified' in tech:
        pm_value = pm_tech_dict[tech]
    else:
        idx = nerc_regions.index(nerc)
        pm_value = pm_tech_dict[tech][idx]
    return pm_value


def add_pm_values(raw_eia_df, level):
    """Compile a DataFrame with level  index cols and adjusted pm2.5 values.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        level (str): An string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pd.DataFrame: A DataFrame with an additional column for PM2.5 values
            calculated from GREET data and sorted by generation technology
            type and region.
    """
    logger.info(' - Adding pm values to the dataframe')
    tech_df = calc_tech_pct(raw_eia_df, level)
    pm_df = calc_pm_value(tech_df, level)
    return pm_df


def calc_public_health_damages(pm_cems_df):
    """Create columns for monetary damages for each pollutant.

    Args:
        pm_cems_df (pandas.DataFrame): A DataFrame with all pollutants in lbs
            per year accounted for.
    Returns:
        pandas.DataFrame: A DataFrame with a column containing the monetary
            damages of all pollutants emitted by a given plant or unit in a
            given year.
    """
    logger.info(' - Calculating public health damages from emissions')
    damages_df = pd.DataFrame()
    # Must do a for loop here because x.report_year does not feed into calc_
    # inflation function because it needs to use i
    for year in pm_cems_df['report_year'].unique().tolist():
        year_df = pm_cems_df.loc[pm_cems_df['report_year'] == year]
        try:
            damage_yr_df = (
                year_df
                .assign(so2_damages=lambda x: (
                    calc_inflation('bls', year,
                                   tons_to_dollars(x.so2_mass_tons, 'sox'))),
                        nox_damages=lambda x: (
                    calc_inflation('bls', year,
                                   tons_to_dollars(x.nox_mass_tons, 'nox'))),
                        pm_damages=lambda x: (
                    calc_inflation('bls', year,
                                   tons_to_dollars(x.pm_mass_tons, 'pm2.5'))),
                        total_damages=lambda x: (x.so2_damages + x.nox_damages
                                                 + x.pm_damages)))
        except IndexError as error:
            print(f'{year} not accounted for in in inflation index: {error}')
        damages_df = damages_df.append(damage_yr_df)
    return damages_df


def tons_to_dollars(emis_col_tons, pollutant):
    """Convert tons of pollutant to dollars in damages based on EPA benMAP.

    Args:
        emis_col_tons (pd.Series): A column with the quantitiy of emissions in
            tons.
        pollutant (str): A string representing the name of the pollutant. Can
            be 'sox', 'nox', or 'pm2.5'
    Returns:
        pd.Series: A new column with the monetary damanges equated with the
            given quantitiy of pollutaion.
    """
    dollars = emis_col_tons * benmap_value[pollutant]
    return dollars


def calc_value_of_life(emis_df):
    """Calculate the amount of premature deaths caused by pollutants.

    Args:
        emis_df (pandas.DataFrame): A DataFrame with the monetary damages per
            pollutant calculated.
    Returns:
        pandas.DataFrame: A DataFrame with a new column for the amount of
            premature deaths caused by the given quantity of pollution.
    """
    logger.info(' - Calculating the amount of premature deaths')
    emis_df['premature_deaths'] = (
        emis_df['total_damages'] / epa_life_value).round(1)
    return emis_df


def build_part3_output(raw_eia_df, bga_df, cems_df, level):
    """Merge EIA, CEMS, and GREET data based on specified agg level.

    Args:
        raw_eia_df (pandas.DataFrame): A DataFrame containing EIA data from the
            pudl_out.mcoe() function that has been run through the prep_raw_
            eia() function.
        bga_df (pandas.DataFrame): The output from running pudl_out.bga()
        cems_df (pandas.DataFrame): The output of running get_cems(). Added as
            a parameter so only have to run get_cems() once because it takes a
            while.
        level (str): A string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    """
    logger.info('Building Part 3 output')
    part1_df = build_part1_output(raw_eia_df, level)
    eia_cems_df = add_cems_to_eia(part1_df, bga_df, cems_df, raw_eia_df, level)
    greet_df = add_pm_values(raw_eia_df, level)
    eia_cems_greet_df = (
        pd.merge(eia_cems_df, greet_df, on=input_dict[level+'_index_cols'],
                 how='outer'))
    with_damages_df = calc_public_health_damages(eia_cems_greet_df)
    with_prem_death_df = calc_value_of_life(with_damages_df)
    logger.info(f'Finished building Part 3 {level} level data')
    return with_prem_death_df


def clean_part3_output(part3_df):
    """Clean output for Part 3: round values.

    Args:
        part3_df (pandas.DataFrame): The result of running build_part3_output()
    Returns:
        pandas.DataFrame: A curated version of the build_part3_output() ready
            for merge with the other parts in main().
    """
    logger.info('Cleaning Part 3 output')
    out_df = (
        part3_df.drop(['so2_damages', 'nox_damages', 'pm_damages'], axis=1)
        .round(2))
    logger.info(f'Finished cleaning Part 3 output')
    return out_df


# ----------------------------------------------------------
# ---------------- * F I N A L - C O M P * -----------------
# ----------------------------------------------------------


def generate_source_df():
    """Create a DataFrame with column headers and their respective sources."""
    logger.info('Generating separate source dataframe')
    source_df = (
        pd.DataFrame(data_source_dict).T
        .reset_index()
        .rename(columns={'index': 'Columns', 0: 'Source', 1: 'Description'}))
    return source_df


def main(pudl_out, cems_df, level, add_sources=False):
    """Create and compile tables from all three parts; final output.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the tables used for EIA and FERC analysis.
        cems_df (pandas.DataFrame): The output of running get_cems(). Added as
            a parameter so only have to run get_cems() once because it takes a
            while.
        level (str): A string indicating the level of desired aggregation.
            Either 'plant-fuel' or 'unit-fuel' are acceptable.
    Returns:
        pandas.DataFrame: A DataFrame with outputs from part 1, 2, & 3 for the
            speciied level of aggregation.
    """
    # Compile raw files to pass to methods.
    raw_eia_df = prep_raw_eia(pudl_out)
    raw_ferc1_df = pudl_out.plants_steam_ferc1()
    bga_df = date_to_year(pudl_out.bga())
    # Calculate first and third parts.
    p1 = clean_part1_output(build_part1_output(raw_eia_df, level))
    p3 = clean_part3_output(build_part3_output(raw_eia_df, bga_df, cems_df,
                                               level))
    # Calculate second part based on specified aggregation level.
    if level == 'plant-fuel':
        p2 = clean_part2_output(build_part2_output(raw_eia_df, raw_ferc1_df))
    elif level == 'unit-fuel':
        p2 = pd.DataFrame(columns=input_dict[level+'_index_cols'])  # Empty
    else:
        print("Not a valid level. Input either 'plant-fuel' or 'unit-fuel'.")
    # Merge data from three parts together.
    p1p3 = (
        pd.merge(p1, p3, on=input_dict[level+'_index_cols'], how='outer'))
    full_df = (
        pd.merge(p1p3, p2, on=input_dict[level+'_index_cols'], how='outer'))
    # Account for eia/pudl id descrepancies by grouping by pudl id.
    clean_df = (
        full_df.groupby(input_dict[level+'_index_cols'], as_index=False)
        .first())
    # Add plant_id_eia column to plant-level data; combine like technology col.
    if level == 'plant-fuel':
        eia_id_df = (
            pd.merge(full_df, raw_eia_df[['plant_id_pudl', 'plant_id_eia']],
                     on=['plant_id_pudl'], how='outer')
            .groupby(input_dict[level+'_index_cols'], as_index=False)
            .agg({'plant_id_eia': lambda x: '; '.join(
                list(map(str, list(x.unique()))))}))
        clean_df = (
            pd.merge(clean_df, eia_id_df, on=input_dict[level+'_index_cols'],
                     how='outer'))
    if add_sources is False:
        final_df = output_aesthetic(clean_df)
    else:
        final_df = output_aesthetic(clean_df, add_sources=True)
    logger.info('Finished compiling all parts!')
    return final_df


# ----------------------------------------------------------
# ------------------- Data Validation ----------------------
# ----------------------------------------------------------

"""
Data validation functions
are used to confirm the FERC Form 1 EIA overlap.
They consist of numeric and graphic manipulations.
"""


def plot_heat_rate(pudl_out):
    """Plot the difference in heat rate for different fuel types."""
    raw_eia_df = prep_raw_eia(pudl_out)
    hr_df = compare_heatrate(raw_eia_df, pd.DataFrame, sd_mean_cols=True)
    fig, (coal_ax, gas_ax) = plt.subplots(ncols=2, nrows=1,
                                          figsize=(17, 8))
    xlabel = "Heat Rate"
    cost_range = (0, 100)
    nbins = 300
    pdf = True

    df_coal = hr_df.loc[hr_df['fuel_type_code_pudl'] == 'coal']
    df_gas = hr_df.loc[hr_df['fuel_type_code_pudl'] == 'gas']
    # df_oil = hr_df.loc[hr_df['fuel_type_code_pudl'] == 'oil']
    # df_waste = hr_df.loc[hr_df['fuel_type_code_pudl'] == 'waste']

    x_coal = df_coal.weighted_ave_heat_rate_mmbtu_mwh
    x_gas = df_gas.weighted_ave_heat_rate_mmbtu_mwh
    # x_oil = df_oil.weighted_ave_heat_rate_mmbtu_mwh
    # x_waste = df_waste.weighted_ave_heat_rate_mmbtu_mwh

    coal_ax.hist([x_coal],
                 histtype='bar',
                 range=cost_range,
                 bins=nbins,
                 # weights=df.coal_fraction_mmbtu_ferc1,
                 label=['Coal'],
                 density=pdf,
                 color=['black'],
                 alpha=0.5)
    coal_ax.set_xlabel(xlabel)
    coal_ax.set_title('Coal Heat Rates')
    coal_ax.legend()

    gas_ax.hist([x_gas],
                histtype='bar',
                range=cost_range,
                bins=nbins,
                # weights=df.coal_fraction_mmbtu_ferc1,
                label=['Gas'],
                density=pdf,
                color=['blue'],
                alpha=0.5)
    gas_ax.set_xlabel(xlabel)
    gas_ax.set_title('Gas Heat Rates')
    gas_ax.legend()
    # plt.savefig("ferc1_eia_fuel_pct_check.png")
    plt.tight_layout()
    plt.show()


def create_compatible_df(df, cols):
    """Compare common fields within FERC and EIA datasets.

    Args:
        df (pandas.DataFrame): A DataFrame with relevant FERC and EIA
            columns. Prepped in compare_ferc_eia() function.
        cols (list): A list of the groupby columns.

    Returns:
        pandas.DataFrame: A DataFrame with the same fields for FERC and
            EIA.
    """
    df = (
        df.groupby(cols)[['net_generation_mwh', 'capacity_mw',
                          'opex_fuel', 'total_mmbtu']]
          .agg(sum).reset_index()
          .assign(fuel_cost_per_mwh=lambda x: (x.opex_fuel /
                                               x.net_generation_mwh),
                  fuel_cost_per_mmbtu=lambda x: x.opex_fuel / x.total_mmbtu,
                  heat_rate_mmbtu_mwh=lambda x: (x.total_mmbtu /
                                                 x.net_generation_mwh),
                  # Can't recall what the 8760 is for...might be so that it
                  # shows up in the graph
                  capacity_factor=lambda x: (x.net_generation_mwh /
                                             (8760 * x.capacity_mw))))
    return df


def merge_ferc1_eia_fuel_pcts(pudl_out):
    """Merge FERC Form 1 and EIA fuel percent data at the plant level.

    This function is used in the formation of the histagram in
    plot_fuel_pct_check() to see whether using EIA fuel percent breakdowns as
    a proxy for FERC is prudent.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the tables used for EIA and FERC analysis.
    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 and EIA fuel breakdowns
        by plant.
    """
    eia_pcts = compile_fuel_pcts_eia(date_to_year(pudl_out.mcoe()))
    eia_pcts = (
        eia_pcts.rename(columns={
                    'pct_net_generation_mwh_coal': 'coal_fraction_mmbtu',
                    'pct_net_generation_mwh_gas': 'gas_fraction_mmbtu',
                    'pct_net_generation_mwh_oil': 'oil_fraction_mmbtu',
                    'pct_net_generation_mwh_waste': 'waste_fraction_mmbtu'})
        .drop(['pct_capacity_mw_coal', 'pct_capacity_mw_gas',
               'pct_capacity_mw_oil', 'pct_capacity_mw_waste'], axis=1))
    ferc1_fuel = pudl.transform.ferc1.fuel_by_plant_ferc1(
                 pudl_out.fuel_ferc1())
    steam_ferc1 = pudl_out.plants_steam_ferc1()
    ferc_pcts = (
        pd.merge(ferc1_fuel, steam_ferc1,
                 on=['report_year', 'utility_id_ferc1', 'plant_name_ferc1'],
                 how='inner'))
    # Merge FERC and EIA860
    ferc1_eia_merge = (
        pd.merge(eia_pcts, ferc_pcts[
                 ['report_year', 'plant_id_pudl', 'coal_fraction_mmbtu',
                  'gas_fraction_mmbtu', 'oil_fraction_mmbtu',
                  'waste_fraction_mmbtu', 'coal_fraction_cost',
                  'gas_fraction_cost', 'oil_fraction_cost',
                  'waste_fraction_cost']], suffixes=('_eia', '_ferc1'),
                 on=['report_year', 'plant_id_pudl'], how='inner'))
    return ferc1_eia_merge


def compare_ferc_eia(pudl_out):
    """Gather FERC, EIA Data; create compatible DataFrames; merge.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the tables used for EIA and FERC analysis.
    Returns:
        pandas.DataFrame: A DataFrame with merged FERC Form 1 and EIA data at
            the plant level. For use in graphic comparison.
    """
    steam_ferc1 = pudl_out.plants_steam_ferc1()
    mcoe = pudl_out.mcoe()
    nf = pudl.transform.ferc1.fuel_by_plant_ferc1(pudl_out.fuel_ferc1())
    key_cols = [
        'report_year',
        'utility_id_ferc1',
        'plant_name_ferc1']
    ferc1_plants = (
        pd.merge(nf, steam_ferc1, on=key_cols, how='inner')
        .assign(heat_rate_mmbtu_mwh=lambda x: (x.fuel_mmbtu /
                                               x.net_generation_mwh))
        .merge(steam_ferc1[key_cols+['utility_id_pudl', 'utility_name_ferc1',
                                     'plant_id_pudl', 'plant_id_ferc1']])
        .rename(columns={'fuel_mmbtu': 'total_mmbtu'}))
    eia_plants = (
        mcoe.assign(report_year=lambda x: x.report_date.dt.year)
        .rename(columns={'total_fuel_cost': 'opex_fuel',
                         'fuel_type_code_pudl': 'primary_fuel_by_mmbtu'}))
    pudl_plant_cols = [
        'plant_id_pudl',
        'primary_fuel_by_mmbtu',
        'report_year']
    # Create comprable dfs
    eia_df = create_compatible_df(eia_plants, pudl_plant_cols)
    ferc1_df = create_compatible_df(ferc1_plants, pudl_plant_cols)
    # Merge dfs
    eia_ferc1_merge = pd.merge(ferc1_df, eia_df, suffixes=('_ferc1', '_eia'),
                               on=pudl_plant_cols, how='inner')
    return eia_ferc1_merge


def plot_prep(df, fields_to_plot, xy_limits, scale="linear"):
    """Make plots to compare FERC & EIA reported values for Coal & Gas plants.

    For each of the fields specified in fields_to_plot, create a pair of plots,
    one for 'gas' and one for 'coal' in the same frame, showing the EIA
    quantity vs. the FERC quantity in a scatter plot.

    Args:
        df (pandas.DataFrame): A DataFrame with merged FERC and EIA data
            (the product of the compare_ferc_eia() function).
        fields_to_plot (list): A list of columns to be compared FERC v. EIA.
        xy_limits (dict): A dictionary depicting the plot limits for each
            field_to_plot.
        scale (str): A string defining the plot scale (linear or log)
    """
    for field in fields_to_plot:
        field_eia = field+'_eia'
        field_ferc1 = field+'_ferc1'
        x_coal = df.query("primary_fuel_by_mmbtu=='coal'")[field_eia]
        y_coal = df.query("primary_fuel_by_mmbtu=='coal'")[field_ferc1]
        x_gas = df.query("primary_fuel_by_mmbtu=='gas'")[field_eia]
        y_gas = df.query("primary_fuel_by_mmbtu=='gas'")[field_ferc1]
        fig, (coal_ax, gas_ax) = plt.subplots(
                                        ncols=2, nrows=1, figsize=(17, 8))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_coal,
                                                                       y_coal)
        coal_ax.scatter(x_coal, y_coal, color='black', alpha=0.1, label=field)
        coal_ax.set_xlim(xy_limits[field][0], xy_limits[field][1])
        coal_ax.set_ylim(xy_limits[field][0], xy_limits[field][1])
        coal_ax.set_xlabel('EIA')
        coal_ax.set_yscale(scale)
        coal_ax.set_xscale(scale)
        coal_ax.set_ylabel('FERC Form 1')
        coal_ax.set_title((f"{field} (Coal),\
                            r-squared = {(r_value**2).round(3)}"))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_gas,
                                                                       y_gas)
        gas_ax.scatter(x_gas, y_gas, color='blue', alpha=0.1, label=field)
        gas_ax.set_xlim(xy_limits[field][0], xy_limits[field][1])
        gas_ax.set_ylim(xy_limits[field][0], xy_limits[field][1])
        gas_ax.set_yscale(scale)
        gas_ax.set_xscale(scale)
        gas_ax.set_xlabel('EIA')
        gas_ax.set_ylabel('FERC Form 1')
        gas_ax.set_title((f"{field} (Gas), \
                          r-squared = {(r_value**2).round(3)}"))

        plt.tight_layout()
        plt.savefig(f"{field}_ferc1_vs_eia.png")


def plot_eia_v_ferc(pudl_out):
    """Bring together plot data for FERC EIA comparison.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the tables used for EIA and FERC analysis.
    Returns:
        matplotlib.pyplot: Plots showing the difference between EIA and FERC
            Form 1 values for specified fields.
    """
    log_fields = [
        'capacity_mw', 'opex_fuel', 'total_mmbtu', 'net_generation_mwh',
    ]
    log_limits = {
        'capacity_mw': (1e1, 1e4), 'opex_fuel': (1e6, 1e9),
        'total_mmbtu': (1e5, 1e9), 'net_generation_mwh': (1e4, 1e8),
    }
    linear_fields = [
        'capacity_factor', 'heat_rate_mmbtu_mwh',
        'fuel_cost_per_mwh', 'fuel_cost_per_mmbtu',
    ]
    linear_limits = {
        'capacity_factor': (0, 1.0), 'heat_rate_mmbtu_mwh': (6, 16),
        'fuel_cost_per_mwh': (10, 80), 'fuel_cost_per_mmbtu': (0, 6),
    }
    df = compare_ferc_eia(pudl_out)
    plot_prep(df, log_fields, log_limits, scale="log")
    plot_prep(df, linear_fields, linear_limits, scale="linear")


def plot_fuel_pct_check(df):
    """Create histograms to compare EIA and FERC Form 1 fuel percents.

    Args:
        df (pandas.DataFrame): A DataFrame with merged FERC Form 1 and EIA fuel
            percents. Product of the merge_ferc1_eia_fuel_pcts() function.
    Returns:
        histogram: A histogram depicting the differnce between EIA fuel percent
            breakdown at the plant and fuel level with FERC Form 1 fuel mmbtu
            fraction and fuel cost fraction.
    """
    fig, (net_gen_ax, cost_ax) = plt.subplots(ncols=2, nrows=1,
                                              figsize=(17, 8))
    xlabel = "EIA / FERC Form 1"
    cost_range = (0, 3)
    nbins = 20
    pdf = True

    x_coal_mmbtu = df.coal_fraction_mmbtu_eia/df.coal_fraction_mmbtu_ferc1
    x_gas_mmbtu = df.gas_fraction_mmbtu_eia/df.gas_fraction_mmbtu_ferc1
    x_coal_cost = df.coal_fraction_mmbtu_eia/df.coal_fraction_cost
    x_gas_cost = df.gas_fraction_mmbtu_eia/df.gas_fraction_cost

    net_gen_ax.hist([x_coal_mmbtu, x_gas_mmbtu],
                    histtype='bar',
                    range=cost_range,
                    bins=nbins,
                    # weights=df.coal_fraction_mmbtu_ferc1,
                    label=['Coal', 'Gas'],
                    density=pdf,
                    color=['black', 'blue'],
                    alpha=0.5)
    net_gen_ax.set_title('EIA / FERC Form 1 mmbtu Fractions')
    net_gen_ax.legend()

    cost_ax.hist([x_coal_cost, x_gas_cost],
                 range=cost_range,
                 bins=nbins,
                 # weights=eia_coal_plants.net_generation_mwh,
                 label=['Coal', 'Gas'],
                 density=pdf,
                 color=['black', 'blue'],
                 alpha=0.5)
    cost_ax.set_xlabel(xlabel)
    cost_ax.set_title('EIA / FERC Form 1 Cost Fractions')
    cost_ax.legend()
    # plt.savefig("ferc1_eia_fuel_pct_check.png")
    plt.tight_layout()
    plt.show()


def validate_nems(with_nems_df):
    nems_df = with_nems_df.loc[with_nems_df['report_year'] == 2018]
    nems_pct_df = (
        nems_df.assign(
            nems_fixed_pct_of_ferc1=lambda x: (
                x.fixed_om_mwh_nems / x.fixed_om_mwh_ferc1),
            nems_var_pct_of_ferc1=lambda x: (
                x.variable_om_mwh_nems / x.variable_om_mwh_ferc1)))
    fixed_pct_ave = nems_pct_df['nems_fixed_pct_of_ferc1'].mean()
    var_pct_ave = nems_pct_df['nems_var_pct_of_ferc1'].mean()
    #print(f'On average, NEMS values are')
    # ADD PLOT
    #nems.loc[nems['nems_fixed_pct_of_ferc1'].notna()]
    #nems.loc[nems['nems_var_pct_of_ferc1'].notna()]
