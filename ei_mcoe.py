"""Functions for compiling EI MCOE dataframes."""

# ----------------------------------------------------------
# ---------------------- Package Imports -------------------
# ----------------------------------------------------------

import pandas as pd
import numpy as np
# import sqlalchemy as sa
# import pudl

import logging
logger = logging.getLogger(__name__)

# pudl_settings = pudl.workspace.setup.get_defaults()
# pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
# pudl_out = pudl.output.pudltabl.PudlTabl(pudl_engine, freq='AS', rolling=True)

# mcoe = pudl_out.mcoe()
# ferc1_steam = pudl_out.plants_steam_ferc1()
# ferc1_fuel = pudl_out.fuel_ferc1()

# ----------------------------------------------------------
# -------------------- Global Variables --------------------
# ----------------------------------------------------------

input_dict = {
    'plant_index_cols': ['plant_id_pudl', 'report_year'],
    'fuel_index_cols': ['plant_id_pudl', 'fuel_type_code_pudl', 'report_year'],
    'unit_index_cols': ['plant_id_pudl', 'plant_id_eia', 'unit_id_pudl',
                        'fuel_type_code_pudl', 'report_year'],
    'merge_cols_qual': ['state', 'city', 'latitude', 'longitude'],
    'merge_cols_simple': ['fuel_type_code_pudl'],
    'eia_sum_cols': ['total_fuel_cost', 'net_generation_mwh', 'capacity_mw'],
    'eia_pct_cols': ['net_generation_mwh', 'capacity_mw'],
    'ferc_sum_cols': ['capex_total', 'opex_fuel', 'opex_production_total']
}

eia_wa_col_dict = {
    'generator_age_years': 'capacity_mw',
    'heat_rate_mmbtu_mwh': 'net_generation_mwh'
}

fuel_types = ['coal', 'gas', 'oil', 'waste']

# ----------------------------------------------------------
# -------------------- Fluid Functions ---------------------
# ----------------------------------------------------------

# df = mcoe


def date_to_year(df):
    """Convert report_date to report_year for MCOE table."""
    logger.info('beginning date conversion')
    df = (df.assign(report_year=lambda x: x.report_date.dt.year)
            .drop('report_date', axis=1))
    return df


def add_generator_age(df):
    """Add column for generator age."""
    logger.info('calculating generator age')
    df = df.astype({'operating_date': 'datetime64[ns]'})
    df = df.assign(generator_age_years=(df.report_year -
                                        df.operating_date.dt.year))
    return df


def test_segment(df):
    """Grab a DataFrame chunch pertaining to a single plant id."""
    df = (df.loc[df['plant_id_pudl'] == 32]
            .sort_values('report_year', ascending=False))
    return df


def year_selector(df, start_year, end_year):
    """Define the range of dates represented in final dataframe."""
    logger.info('selecting years')
    df_years = df.loc[df['report_year'].isin(range(start_year, end_year+1))]
    return df_years


def weighted_average(df, wa_col_dict, index_cols):
    """Generate a weighted average for multiple columns at once.

    Args:
        df (pandas.DataFrame): A DataFrame containing, at minimum, the columns
            specified in the other parameters wa_col_dict and by_cols.
        wa_col_dict (dictionary): A dictionary containing keys and values that
            represent the column names for the 'data' and 'weight' values.
        index_cols (list): A list of the columns to group by when calcuating
            the weighted average value.

    Returns:
        pandas.DataFrame: A DataFrame containing weigted average values for
            specified 'data' columns based on specified 'weight' columns.
            Grouped by an indicated set of columns.
    """
    merge_df = df[index_cols]
    for data, weight in wa_col_dict.items():
        logger.info('calculating weighted average for ' + data)
        df['_data_times_weight'] = df[data] * df[weight]
        df['_weight_where_notnull'] = df[weight] * pd.notnull(df[data])
        g = df.groupby(index_cols)
        result = g[
            '_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        result = result.to_frame(name='weighted_ave_'+data).reset_index()
        merge_df = pd.merge(merge_df, result, on=index_cols, how='outer')
    return merge_df


def regroup_data(df, index_cols, merge_cols=[], wa_col_dict=None,
                 sum_cols=None):
    """Regroup data by plant or unit and run aggregation calculations.

    Args:
        df (pandas.DataFrame): A DataFrame containing all relevent columns.
            Likely the pudl_out.mcoe() or equivalent.
        index_cols (list): List of columns to define the groupby.
        merge_cols (list): List of columns to re-incorporate into the final
            DataFrame.
        wa_col_dict (dictionary): Dictionary with keys and values that
            indicate the column names of 'data' and 'weight' values to be
            used in weighted average calculation.
        sum_cols (list): List of columns to be summed during groupby.
        drop_calcs (bool): Boolean to indicate whether to drop calcuated
            columns and just keep the merge_cols and index_cols in new
            groupby form.
        count_col (bool): Boolean to indicate whether to add a column to
            count the number of items compiled per index value in a
            groupby.

    Returns:
        pandas.DataFrame: A DataFrame containing data properly regrouped
            and merged.
    """
    logger.info('regrouping data')
    # Create empty dataframes for merge incase left blank in parameters
    sum_df = df[index_cols]
    wa_df = df[index_cols]

    count = df.groupby(index_cols, as_index=False).size().reset_index(
                                                          name='count')

    if sum_cols is not None:
        sum_df = df.groupby(index_cols, as_index=False)[sum_cols].sum()
    # sum_df = df.groupby(index_cols,as_index=False)[sum_cols].agg(calc_funcs)

    # Find weighted average of generator ages
    if wa_col_dict is not None:
        wa_df = weighted_average(df, wa_col_dict, index_cols)
    # Merge sum and weighted average tables
    wa_sum_merge_df = pd.merge(sum_df, wa_df, on=index_cols, how='outer')
    wa_sum_count = pd.merge(wa_sum_merge_df, count, on=index_cols, how='outer')

    # Merge conglomerate table with final 'merge_cols'
    merge_df = df[index_cols+merge_cols]
    result_df = pd.merge(
        wa_sum_count, merge_df, on=index_cols, how='left').drop_duplicates()
    return result_df


# ----------------------------------------------------------
# --------------------- * P A R T  1 * ---------------------
# ----------------------------------------------------------

"""
Part 1 Functions are mostly neutral and applicable accross eia and ferc
datasets. Their primary purpose is to use groupby and merge functions to
reorient and grab subsets of data. Some of them are used in Part 2 as well.
"""


def part1_main(pudl_out, level, start_yr=None, end_yr=None, drop_calcs=False):
    """Create final data output for Part 1.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the mcoe table used for eia analysis.
        level (str): A string (either 'plant' or 'unit') used to indicate
            the output of the function.
        start_yr (int): An optional integer used to indicate the first year
            of data desired in the output table.
        end_yr (int): An optional integer used to indicate the last year of
            data desired in the output table.

    Returns:
        pandas.DataFrame: A DataFrame that either reflects the plant level
            or unit level eia data separated by fuel type.
    """
    # Prep mcoe table data
    df = pudl_out.mcoe()
    df = add_generator_age(date_to_year(df))

    level_df = regroup_data(df, input_dict[level+'_index_cols'],
                            merge_cols=input_dict['merge_cols_qual'],
                            wa_col_dict=eia_wa_col_dict,
                            sum_cols=input_dict['eia_sum_cols'])

    # if level == 'plant':
    #     level_df = regroup_data(df, input_dict['plant_index_cols'],
    #                             merge_cols=input_dict['merge_cols_qual'],
    #                             wa_col_dict=eia_wa_col_dict,
    #                             sum_cols=input_dict['eia_pct_cols'],)
    #                             # drop_calcs=True)
    # if level == 'unit':
    #     level_df = regroup_data(df, input_dict['unit_index_cols'],
    #                             merge_cols=input_dict['merge_cols_qual'],
    #                             wa_col_dict=eia_wa_col_dict,
    #                             sum_cols=input_dict['eia_sum_cols'])

    # Conditional to comply with EI contract request to have plant-level
    # data bare.
    if drop_calcs is True:
        level_df = level_df[input_dict[level+'_index_cols'] +
                            input_dict['merge_cols_qual']]

    if start_yr is not None:
        level_df = year_selector(level_df, start_yr, end_yr)
    logger.info('Finished Part 1 '+level+' level compilation')
    return level_df


# ----------------------------------------------------------
# --------------------- * P A R T  2 * ---------------------
# ----------------------------------------------------------

"""
Part 2 Functions are primarily used to transform EIA923 and FERC Form 1 data
so that they are compatible with one another. We use EIA936 data broken down
by plant and fuel type to inform the FERC Form 1 data disaggregation in the
same manner. In other words, we calculate the percent that each fuel type
contributes to a given plant-level statistic (in this case capacity, net
generation, or cost) for the EIA data and use those fuel percentages accros
statistics to map disaggregate FERC Form 1 fixed and operating cost data in a
similar manner. We use the combined information from  EIA923 and FERC Form 1 to
calculate an mcoe value for each fuel type within each plant for any given
report year.
"""


def eia_pct_df_maker(df, col):
    """Move fuel type row values to columns.

    Reorient dataframe by plant so that fuel type percentages are columns
    rather than row values. Used for merging with FERC Form 1 data. The
    output DataFrame is ready to merge with another DataFrame from this output
    with a different 'col' specified. It is also ready to merge with FERC Form
    1 plant level data to perform fuel level disagregation.

    Args:
        df (pandas.DataFrame): A DataFrame with percent values for columns
            by fuel type.
        col (str): The name of the percent value column (such as
            'capacity_mw' or 'net_generation_mwh') that will be reoriented
            to suit the calculation of fuel type breakdown of FERC Form 1
            data.

    Returns:
        pandas.DataFrame: A DataFrame with the fuel type percent values
            as columns rather than rows.
    """
    logger.info('turning eia fuel percent values for ' + col + ' into columns')
    pct_df = (df.pivot_table('pct_'+col, input_dict['plant_index_cols'],
                             'fuel_type_code_pudl')
                .reset_index()
                .rename(columns={'coal': 'pct_'+col+'_coal',
                                 'gas': 'pct_'+col+'_gas',
                                 'oil': 'pct_'+col+'_oil',
                                 'waste': 'pct_'+col+'_waste'}))
    return pct_df


def calc_eia_fuel_percentages(df, pct_col1, pct_col2):
    """Calculate the fuel type breakdown for input columsn.

    For use specifically in the eia_fuel_pcts() function where a DataFrame
    containing eia data aggragated by fuel type and plant totals is created.

    Args:
        df (pandas.DataFrame): A DataFrame containing eia data containing
            values aggregated by plant fuel type AND plant total.
        pct_col1 (str): A single column name to be broken down by fuel
            type.
        pct_col2 (str): A single column name to be broken down by fuel
            type.

    Returns:
        pandas.DataFrame: A DataFrame containing the percent breakdown by
            fuel type of pct_col1. Merged with that of pct_col2.
    """
    logger.info('calculating eia fuel type percentages')
    # Calculate percent that each fuel contributes to input cols
    # (capcity and net gen in this case)
    df['pct_'+pct_col1] = df[pct_col1] / df[pct_col1+'_plant_level']
    df['pct_'+pct_col2] = df[pct_col2] / df[pct_col2+'_plant_level']
    # Reorient table so that fuel type percents become columns
    # (makes it easier to run calculations on FERC1 data)
    pct_df1 = eia_pct_df_maker(df, pct_col1)
    pct_df2 = eia_pct_df_maker(df, pct_col2)
    # Merge percent dfs so that they are both included.
    # pd.merge will not take a df LIST -- need to fix this.
    eia_pct_merge = pd.merge(
        pct_df1, pct_df2, on=input_dict['plant_index_cols'], how='outer')
    return eia_pct_merge


def prep_eia_data(df):
    """Group eia data by plant and fuel type.

    Args:
        df (pandas.DataFrame): A DataFrame with relevent eia data to be
            aggregated for integration with FERC Form 1 data.

    Returns:
        pandas.DataFrame: A DataFrame with plant level data disaggregated
            by fuel type ready to merge with FERC Form 1 data once it's
            been similarly disaggregated with eia percent data.
    """
    logger.info('building eia table broken down by plant and fuel type')
    eia_plant_fuel_df = regroup_data(df, input_dict['fuel_index_cols'],
                                     sum_cols=input_dict['eia_sum_cols'])
    return eia_plant_fuel_df


def eia_fuel_pcts(df):
    """Extract fuel type percents for use with FERC Form 1 Data.

    This function uses other functions to find the breakdown of mcoe variable
    columns (such as capacity and net generation) by plant and fuel type.
    The DataFrame output is used to disaggretate Ferc Form 1 data in the same
    manner.

    Args:
        df (pandas.DataFrame): A DataFrame with relevent eia data to be
            aggregated for integration with FERC Form 1 data.

    Returns:
        pandas.DataFrame: A DataFrame with weight columns (capacity
            and net generation) broken down by fuel type at the plant level
            for use in calculating FERC Form 1 data breakdown by the same
            means.
    """
    logger.info('readying eia fuel pct data to merge with ferc')
    eia_plant_fuel_df = prep_eia_data(df)
    # Create df that finds the plant level totals (combines fuel types) for the
    # aggregated mcoe data
    eia_plant_totals_df = regroup_data(
        df, input_dict['plant_index_cols'],
        merge_cols=input_dict['merge_cols_simple'],
        sum_cols=input_dict['eia_sum_cols'])
    # Rename fields to differentiate fuel type level vs. plant level.
    eia_plant_totals_df = eia_plant_totals_df.rename(
        columns={'total_fuel_cost': 'total_fuel_cost_plant_level',
                 'net_generation_mwh': 'net_generation_mwh_plant_level',
                 'capacity_mw': 'capacity_mw_plant_level'})
    # Merge with eia_plant_fuel_df --- having a hard time doing this in the
    # regroup_data() function. Should show plant totals AND fuel type totals
    eia_plant_fuel_df = pd.merge(
        eia_plant_fuel_df, eia_plant_totals_df,
        on=input_dict['fuel_index_cols'], how='left')
    # Calculate the percentage that each fuel type (coal, oil, gas, waste)
    # accounts for for the specified columns (net gen & capacity)
    # **NOTE** cannot feed this function a list of col names beacuse merge
    # function does not take a list.
    eia_pct_df = calc_eia_fuel_percentages(
        eia_plant_fuel_df, 'net_generation_mwh', 'capacity_mw')
    # Return table needed for ferc fuel type delineation and final FERC1 merge.
    return eia_pct_df


def ferc1_plant_level_prep(df):
    """Ready FERC Form 1 data for merging with EIA-932 fuel pct breakdown.

    The output DataFrame for this function will be ready to merge with the EIA
    percent breakdowns by plant and fuel type.

    Args:
        df (pandas.DataFrame): A DataFrame with raw FERC Form 1 Data.

    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 Data aggregated by
            plant.

    """
    logger.info('building FERC table broken down by plant')
    # **NOTE** Does not include 'COUNT' field -- had trouble adding it into
    # the regroup_data() function
    ferc1_plant_df = regroup_data(
        df, input_dict['plant_index_cols'],
        sum_cols=input_dict['ferc_sum_cols'])
    ferc1_plant_df['opex_nofuel_ferc1'] = ferc1_plant_df[
        'opex_production_total'] - ferc1_plant_df['opex_fuel']
    # Rename cols to indication origin of FERC1
    ferc1_plant_df = ferc1_plant_df.rename(
        columns={'count': 'count_ferc1',
                 'capex_total': 'capex_total_ferc1',
                 'opex_fuel': 'opex_fuel_ferc1',
                 'opex_production_total': 'opex_production_total_ferc1'})
    return ferc1_plant_df


def ferc_cost_pct_breakdown(df):
    """Calculate FERC Form 1 cost breakdowns from EIA-923 fuel percentages.

    For specific use within the merge_ferc_with_eia_pcts() function.

    Args:
        df (pandas.DataFrame): A DataFrame with raw ferc plant values
            merged with eia percents.

    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 Data disaggregated
            by plant and fuel type based on EIA percent values.

    """
    logger.info('building FERC table broken down by plant and fuel type')
    for fuel in fuel_types:
        df['capex_'+fuel] = df[
            'capex_total_ferc1'] * df['pct_capacity_mw_'+fuel]
        df['opex_nofuel_'+fuel] = df[
            'opex_nofuel_ferc1'] * df['pct_net_generation_mwh_'+fuel]
    return df


def cost_subtable_maker(df, cost):
    """Rearange cost breakdown data.

    This function takes the FERC Form 1 data broken down by plant and fuel type
    based on EIA percentages and melts it so that the fuel type columns become
    row values again rather than columns. This function must be executed once
    for every FERC value that has been disaggregated. (i.e. once for capex and
    once for opex).

    Args:
        df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            disaggregated by plant and fuel type.
        cost (str): A string with the name of the cost column to subdivide by.

    Returns:
        pandas.DataFrame: A DataFrame with disaggregated FERC Form 1 data
            melted so that columns become row values.
    """
    logger.info('melting FERC pct data back to row values')
    # apply EIA fuel percents to specified FERC cost data.
    df = df[input_dict['plant_index_cols'] + [cost+'_coal', cost+'_gas',
                                              cost+'_oil', cost+'_waste']]
    df = df.rename(
        columns={cost+'_coal': 'coal',
                 cost+'_gas': 'gas',
                 cost+'_oil': 'oil',
                 cost+'_waste': 'waste'})
    df_melt = pd.melt(df, input_dict['plant_index_cols']).rename(
        columns={'value': cost, 'variable': 'fuel_type_code_pudl'})

    df_melt = df_melt.dropna(subset=[cost])
    return df_melt


def merge_ferc_with_eia_pcts(eia_pct_df, ferc_df):
    """Merge EIA fuel percents with FERC Form 1 data.

    This function makes use of several of the other functions defined here.

    Args:
        eia_pct_df (pandas.DataFrame): A DataFrame with EIA aggregated by
            plant and fuel type.
        ferc_df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            aggregated by plant.

    Returns:
        pandas.DataFrame: A DataFrame with FERC Form 1 data disaggregated
            by the fuel percentage breakdowns depicted in the EIA data.
    """
    logger.info('merging FERC data with EIA pct data')
    # Merge prepped EIA923 percent data with FERC1 cost data
    ferc_eia_pcts = pd.merge(
        eia_pct_df, ferc_df, on=input_dict['plant_index_cols'], how='outer')
    ferc_eia_pcts = ferc_cost_pct_breakdown(ferc_eia_pcts)
    capex_melt = cost_subtable_maker(ferc_eia_pcts, 'capex')
    opex_melt = cost_subtable_maker(ferc_eia_pcts, 'opex_nofuel')
    # Merge capex and opex FERC1 tables
    ferc_cap_op = pd.merge(
        capex_melt, opex_melt, on=input_dict['fuel_index_cols'], how='outer')
    return ferc_cap_op


def merge_ferc_eia_mcoe(eia_fuel_df, ferc_fuel_df):
    """Produce final EIA and FERC Form 1 Merge and calculate MCOE value.

    Args:
        eia_fuel_df (pandas.DataFrame): A DataFrame with EIA data broken
            down by plant and fuel type.
        ferc_fuel_df (pandas.DataFrame): A DataFrame with FERC Form 1 data
            broken down by plant and fuel type.

    Returns:
        pandas.DataFrame: A DataFrame with EIA and FERC Form 1 data broken
            down by plant and fuel type. MCOE values calculated.
    """
    logger.info('merging FERC and EIA data on plant and fuel type')
    # Merge FERC1 and EIA923 on plant, fuel, and year using prep_eia_data()
    # output associated with key 'plant_fuel_ag'
    eia_ferc_merge = pd.merge(
        eia_fuel_df, ferc_fuel_df, on=input_dict['fuel_index_cols'],
        how='outer')
    # Calculate MCOE and
    eia_ferc_merge = eia_ferc_merge.assign(
        mcoe=((eia_ferc_merge['total_fuel_cost'] +
               eia_ferc_merge['opex_nofuel']) +
              eia_ferc_merge['capex'] *
              eia_ferc_merge['capacity_mw']) /
        eia_ferc_merge['net_generation_mwh'],
        fuel_cost_mwh_eia923=lambda x: (x.total_fuel_cost /
                                        x.net_generation_mwh),
        variable_om_mwh_ferc1=lambda x: x.opex_nofuel / x.net_generation_mwh,
        fixed_om_mwh_ferc1=lambda x: x.capex / x.net_generation_mwh)
    # Rearrange columns
    eia_ferc_merge = eia_ferc_merge[[
        'plant_id_pudl',
        'fuel_type_code_pudl',
        'report_year',
        'fuel_cost_mwh_eia923',
        'variable_om_mwh_ferc1',
        'fixed_om_mwh_ferc1',
        'mcoe']]
    return eia_ferc_merge


def part2_main(pudl_out, start_yr=None, end_yr=None):
    """Create final data output for Part 2.

    A function that calls the other functions for Part 2 and outputs the mcoe
    DataFrame.

    Args:
        pudl_out (pudl.output.pudltabl.PudlTabl): An object used to create
            the mcoe table used for eia analysis.
        level (str): A string (either 'plant' or 'unit') used to indicate
            the output of the function.
        start_yr (int): An optional integer used to indicate the first year
            of data desired in the output table.
        end_yr (int): An optional integer used to indicate the last year of
            data desired in the output table.
    Returns:
        pandas.DataFrame: A DataFrame with the MCOE calculations.
    """
    eia_raw = pudl_out.mcoe()
    eia_raw = add_generator_age(date_to_year(eia_raw))
    ferc_raw = pudl_out.plants_steam_ferc1()
    ferc_prep = merge_ferc_with_eia_pcts(eia_fuel_pcts(eia_raw),
                                         ferc1_plant_level_prep(ferc_raw))
    eia_prep = prep_eia_data(eia_raw)
    mcoe_df = merge_ferc_eia_mcoe(eia_prep, ferc_prep)
    # Add heat rate comparison
    hr_df = compare_heatrate(part1_main(pudl_out, 'unit'), 'plant')
    # Make sure dfs are the same length before merging
    logger.info('checking df length compatability')
    if len(mcoe_df) != len(hr_df):
        print('dfs not the same length')
    mcoe_hr_df = pd.merge(mcoe_df, hr_df, on=input_dict['fuel_index_cols'])
    if start_yr is not None:
        mcoe_hr_df = year_selector(mcoe_df, start_yr, end_yr)
    logger.info('Finished Part 2 Compilation')
    return mcoe_hr_df


# ----------------------------------------------------------
# ------------------- Data Validation ----------------------
# ----------------------------------------------------------

def compare_costs(df):
    """Compare costs within plants to find outliers"""
    logger.info('comparing costs internally')
    plant_cost_df = regroup_data(df, input_dict['fuel_index_cols'],
                                 merge_cols=['unit_id_pudl', 'plant_id_eia'],
                                 sum_cols=[''])


def compare_heatrate(df, output_level):
    """Compare heatrates within plants to find outliers.

    Args:
        df (pandas.DataFrame): A DataFrame with plant and fuel type data
        produced by running part1_main('unit').
    Returns:
        pandas.DataFrame: A Dataframe with a boolean column to show whether
        the heat rate of a given unit supercedes the total plant heat rate
        divided by the number of units per that plant.
    """
    logger.info('comparing heat rates internally')
    # Create create plant/fuel wahr sums and merge back with unit level
    plant_hr_df = regroup_data(df, input_dict['fuel_index_cols'],
                               merge_cols=['unit_id_pudl', 'plant_id_eia'],
                               sum_cols=['weighted_ave_heat_rate_mmbtu_mwh'])
    # Merge wahr plant/fuel with wahr for unit level
    plant_hr_df = (pd.merge(plant_hr_df, df[input_dict['unit_index_cols'] +
                            ['weighted_ave_heat_rate_mmbtu_mwh']],
                            on=input_dict['unit_index_cols'], how='outer')
                     .rename(columns={'weighted_ave_heat_rate_mmbtu_mwh_x':
                                      'wei_ave_hr_plant',
                                      'weighted_ave_heat_rate_mmbtu_mwh_y':
                                      'wei_ave_hr_unit'}))
    # Create bool column to indicate significant hr of one unit per plant
    plant_hr_df = plant_hr_df.assign(sig_hr=((plant_hr_df['wei_ave_hr_plant'] /
                                             plant_hr_df['count']) <
                                             plant_hr_df['wei_ave_hr_unit']))
    # Merge back with info of input df
    merge_df = pd.merge(df, plant_hr_df[input_dict['unit_index_cols'] +
                                                  ['sig_hr']],
                        on=input_dict['unit_index_cols'], how='outer')
    # Group by plant level - if any unit was True, plant is also True
    if output_level == 'plant':
        merge_df = plant_hr_df.groupby(
            input_dict['fuel_index_cols'])['sig_hr'].any().reset_index()

    return merge_df
