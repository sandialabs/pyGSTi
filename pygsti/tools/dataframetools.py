"""
Tools for working with Pandas dataframes
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


def _drop_constant_cols(df):
    to_drop = [col for col in df.columns if len(df[col].unique()) == 1]
    return df.drop(columns=to_drop)


def _reset_index(df):
    '''Returns DataFrame with index as columns - works with Categorical indices unlike DataFrame.reset_index'''
    import pandas as _pd
    index_df = df.index.to_frame(index=False)
    df = df.reset_index(drop=True)
    # In merge is important the order in which you pass the dataframes
    # if the index contains a Categorical. I.e.,
    # pd.merge(df, index_df, left_index=True, right_index=True) does not work.
    return _pd.merge(index_df, df, left_index=True, right_index=True)


def _process_dataframe(df, pivot_valuename, pivot_value, drop_columns, preserve_order=False):
    """ See to_dataframe docstrings for argument descriptions. """
    if drop_columns:
        if drop_columns is True: drop_columns = (True,)
        for col in drop_columns:
            df = _drop_constant_cols(df) if (col is True) else df.drop(columns=col)

    if pivot_valuename is not None or pivot_value is not None:
        if pivot_valuename is None: pivot_valuename = "ValueName"
        if pivot_value is None: pivot_value = "Value"
        index_columns = list(df.columns)
        index_columns.remove(pivot_valuename)
        index_columns.remove(pivot_value)
        df_all_index_but_value = df.set_index(index_columns + [pivot_valuename])
        df_unstacked = df_all_index_but_value[pivot_value].unstack()
        if preserve_order:  # a documented bug in pandas is unstack sorts - this tries to fix (HACK)
            #df_unstacked = df_unstacked.reindex(df_all_index_but_value.index.get_level_values(0))
            df_unstacked = df_unstacked.reindex(df_all_index_but_value.index.get_level_values(0).unique())
        df = _reset_index(df_unstacked)

    return df