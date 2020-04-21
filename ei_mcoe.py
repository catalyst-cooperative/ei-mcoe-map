"""Functions for compiling EI MCOE dataframes."""

# ----------------------------------------------------------
# ---------------------- Package Imports -------------------
# ----------------------------------------------------------

import pandas as pd
# import sqlalchemy as sa
import pudl  # is it bad that this is imported here as well? (used for graphs)
import matplotlib.pyplot as plt
from scipy import stats

import logging
logger = logging.getLogger(__name__)

# pudl_settings = pudl.workspace.setup.get_defaults()
# pudl_engine = sa.create_engine(pudl_settings["pudl_db"])
# pudl_out = pudl.output.pudltabl.PudlTabl(pudl_engine, freq='AS',rolling=True)

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


def date_to_year(df):
    """Convert report_date to report_year for MCOE table."""
    logger.info('beginning date conversion')
    df = (
        df.assign(report_year=lambda x: x.report_date.dt.year)
          .drop('report_date', axis=1)
    )
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
    df = (
        df.loc[df['plant_id_pudl'] == 32]
          .sort_values('report_year', ascending=False)
    )
    return df


def year_selector(df, start_year, end_year):
    """Define the range of dates represented in final dataframe."""
    logger.info('selecting years')
    df_years = df.loc[df['report_year'].isin(range(start_year, end_year+1))]
    return df_years


def weighted_average(df, wa_col_dict, idx_cols):
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
    merge_df = df[idx_cols]
    for data, weight in wa_col_dict.items():
        logger.info('calculating weighted average for ' + data)
        df['_data_times_weight'] = df[data] * df[weight]
        df['_weight_where_notnull'] = df[weight] * pd.notnull(df[data])
        g = df.groupby(idx_cols)
        result = g[
            '_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        result = result.to_frame(name='weighted_ave_'+data).reset_index()
        merge_df = pd.merge(merge_df, result, on=idx_cols, how='outer')
    return merge_df


def regroup_data(df, idx_cols, merge_cols=[], wa_col_dict=None, sum_cols=None):
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
    sum_df = df[idx_cols]
    wa_df = df[idx_cols]
    count = df.groupby(idx_cols, as_index=False).size().reset_index(
                                                        name='count')
    if sum_cols is not None:
        sum_df = df.groupby(idx_cols, as_index=False)[sum_cols].sum()
    # Find weighted average of generator ages and heat rate
    if wa_col_dict is not None:
        wa_df = weighted_average(df, wa_col_dict, idx_cols)
    # Merge sum and weighted average tables
    wa_sum_merge_df = pd.merge(sum_df, wa_df, on=idx_cols, how='outer')
    wa_sum_count = pd.merge(wa_sum_merge_df, count, on=idx_cols, how='outer')
    # Merge conglomerate table with final 'merge_cols'
    merge_df = df[idx_cols+merge_cols]
    result_df = pd.merge(
        wa_sum_count, merge_df, on=idx_cols, how='left').drop_duplicates()
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
    df = add_generator_age(date_to_year(pudl_out.mcoe()))

    if level == 'plant':
        level = 'fuel'

    level_df = regroup_data(df, input_dict[level+'_index_cols'],
                            merge_cols=input_dict['merge_cols_qual'],
                            wa_col_dict=eia_wa_col_dict,
                            sum_cols=input_dict['eia_sum_cols'])
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
    """Calculate the fuel type breakdown for input columns.

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
    # pd.merge will not take a df list.
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
        sum_cols=input_dict['eia_sum_cols']).rename(
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
    ferc1_plant_df = (
        regroup_data(df, input_dict['plant_index_cols'],
                     sum_cols=input_dict['ferc_sum_cols'])
        .assign(opex_nofuel_ferc1=lambda x: (x.opex_production_total -
                                             x.opex_fuel))
        .rename(
            columns={'count': 'count_ferc1',
                     'capex_total': 'capex_total_ferc1',
                     'opex_fuel': 'opex_fuel_ferc1',
                     'opex_production_total': 'opex_production_total_ferc1'})
    )
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
    # Did not use .assign here because need to integrate local variables.
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
    df = (
        df[(input_dict['plant_index_cols'] +
           [cost+'_coal', cost+'_gas', cost+'_oil', cost+'_waste'])]
        .rename(
            columns={cost+'_coal': 'coal',
                     cost+'_gas': 'gas',
                     cost+'_oil': 'oil',
                     cost+'_waste': 'waste'})
    )
    df_melt = (
        pd.melt(df, input_dict['plant_index_cols'])
          .rename(columns={'value': cost, 'variable': 'fuel_type_code_pudl'})
          .dropna(subset=[cost])
    )
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
    eia_ferc_merge = (
        pd.merge(eia_fuel_df, ferc_fuel_df, on=input_dict['fuel_index_cols'],
                 how='outer')
          .assign(
            mcoe=(lambda x: ((x.total_fuel_cost + x.opex_nofuel) +
                             x.capex * x.capacity_mw) / x.net_generation_mwh),
            fuel_cost_mwh_eia923=lambda x: (x.total_fuel_cost /
                                            x.net_generation_mwh),
            variable_om_mwh_ferc1=lambda x: (x.opex_nofuel /
                                             x.net_generation_mwh),
            fixed_om_mwh_ferc1=lambda x: x.capex / x.net_generation_mwh)
    )
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
    hr_df = compare_heatrate(pudl_out)
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


def compare_heatrate(pudl_out):
    """Compare heatrates within plants to find outliers.

    Outputs a pandas DataFrame containing information about whether unit level
    heat rates differ significantly from plant level heatrates. Each is
    already calculated by way of weighted average. A significant difference in
    this case is defined by a unit differing by more than 1 or -1 from the
    plant average. This is derived from eia data on heatrates. The final output
    shows TRUE if the plant contains a unit with a heatrate differing by the
    aforementioned amount.

    Args:
        df (pandas.DataFrame): A DataFrame with plant and fuel type data
            produced by running part1_main('unit').
    Returns:
        pandas.DataFrame: A Dataframe with a boolean column to show whether
            the heat rate of a given unit supercedes the total plant heat rate
            divided by the number of units associated with that plant.
    """
    logger.info('comparing heat rates internally')
    # Get plant and unit level wahr then merge for comparison.
    plant_level = part1_main(pudl_out, 'plant')
    unit_level = part1_main(pudl_out, 'unit')
    # Calculate significant difference in heat rate. I chose one based
    # on approximate differences here:
    # https://www.eia.gov/electricity/annual/html/epa_08_01.html
    plant_unit_merge = (
        pd.merge(unit_level, plant_level, suffixes=['_unit', '_plant'],
                 on=input_dict['fuel_index_cols'], how='outer')
          .assign(
                 sig_hr=lambda x: (
                    abs(x.weighted_ave_heat_rate_mmbtu_mwh_plant -
                        x.weighted_ave_heat_rate_mmbtu_mwh_unit)) > 1)
          .groupby(input_dict['fuel_index_cols'])['sig_hr'].any()
          .reset_index()
    )
    return plant_unit_merge


def create_compatible_df(df, cols):
    """Arrange FERC and EIA dataframes with common fields.

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
                  capacity_factor=lambda x: (x.net_generation_mwh /
                                             (8760*x.capacity_mw)))
     )
    return df


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
          .rename(columns={'fuel_mmbtu': 'total_mmbtu'})
    )
    eia_plants = (
        mcoe.assign(report_year=lambda x: x.report_date.dt.year)
            .rename(columns={'total_fuel_cost': 'opex_fuel',
                             'fuel_type_code_pudl': 'primary_fuel_by_mmbtu'})
    )
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
        xy_limits (dictionary): A dictionary depicting the plot limits for each
            field_to_plot.
        scale (string): A string defining the plot scale (linear or log)
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
    eia_pcts = eia_fuel_pcts(date_to_year(pudl_out.mcoe()))
    eia_pcts = (eia_pcts
                .rename(columns={
                    'pct_net_generation_mwh_coal': 'coal_fraction_mmbtu',
                    'pct_net_generation_mwh_gas': 'gas_fraction_mmbtu',
                    'pct_net_generation_mwh_oil': 'oil_fraction_mmbtu',
                    'pct_net_generation_mwh_waste': 'waste_fraction_mmbtu'
                 })
                .drop(['pct_capacity_mw_coal', 'pct_capacity_mw_gas',
                       'pct_capacity_mw_oil', 'pct_capacity_mw_waste'],
                      axis=1))

    ferc1_fuel = pudl.transform.ferc1.fuel_by_plant_ferc1(
                 pudl_out.fuel_ferc1())
    steam_ferc1 = pudl_out.plants_steam_ferc1()
    ferc_pcts = pd.merge(
        ferc1_fuel, steam_ferc1,
        on=['report_year', 'utility_id_ferc1', 'plant_name_ferc1'],
        how='inner')

    # Merge FERC and EIA860
    ferc1_eia_merge = (
        pd.merge(
              eia_pcts, ferc_pcts[
                ['report_year', 'plant_id_pudl', 'coal_fraction_mmbtu',
                 'gas_fraction_mmbtu', 'oil_fraction_mmbtu',
                 'waste_fraction_mmbtu', 'coal_fraction_cost',
                 'gas_fraction_cost', 'oil_fraction_cost',
                 'waste_fraction_cost']], suffixes=('_eia', '_ferc1'),
              on=['report_year', 'plant_id_pudl'], how='inner'))
    return ferc1_eia_merge


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
    #plt.savefig("ferc1_eia_fuel_pct_check.png")
    plt.tight_layout()
    plt.show()
